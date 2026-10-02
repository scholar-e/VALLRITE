"""Distill soft candidate rankings from a Qwen teacher into a smaller adapter."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import sys
import time

LOGGER = logging.getLogger("competitive_word_decoder.distill")


def configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.handlers.clear()
    file_handler = logging.FileHandler(path, delay=False)
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(console_handler)


def load_teacher_scores(path: Path) -> dict[tuple[str, int], float]:
    scores: dict[tuple[str, int], float] = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        key = (row["clip_id"], int(row["candidate_index"]))
        score = float(row["score"])
        if key in scores or not math.isfinite(score):
            raise ValueError(f"duplicate or non-finite teacher score: {key}")
        scores[key] = score
    if not scores:
        raise ValueError("teacher score file is empty")
    return scores


def prepare_records(examples_path: Path, score_path: Path, score_window: float) -> list[dict]:
    from PhonemeDecoder.train_qwen import candidate_prompt

    scores = load_teacher_scores(score_path)
    records = []
    seen = set()
    for line in examples_path.read_text().splitlines():
        record = json.loads(line)
        clip_id = record["clip_id"]
        if clip_id in seen:
            raise ValueError(f"duplicate clip id: {clip_id}")
        seen.add(clip_id)
        candidates = record["candidates"]
        if not candidates:
            continue
        cutoff = candidates[0]["log_score"] - score_window
        indexes = [index for index, candidate in enumerate(candidates)
                   if candidate["log_score"] >= cutoff and (clip_id, index) in scores]
        if not indexes:
            continue
        reference_indexes = [index for index in indexes
                             if candidates[index]["words"] == record["reference_words"]]
        records.append({
            "clip_id": clip_id,
            "prompt": candidate_prompt(record),
            "candidate_texts": [candidates[index]["text"] for index in indexes],
            "teacher_scores": [scores[(clip_id, index)] for index in indexes],
            "reference_index": (indexes.index(reference_indexes[0]) if reference_indexes else None),
        })
    if not records:
        raise ValueError("no examples overlap the teacher scores")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--teacher-scores", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("checkpoints/Qwen3.5-0.8B"))
    parser.add_argument("--initial-adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-entries", type=int, default=5685)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--hard-weight", type=float, default=0.5)
    parser.add_argument("--token-weight", type=float, default=0.1)
    parser.add_argument("--visual-score-window", type=float, default=5.0)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pause-file", type=Path, default=Path("PAUSE"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    paths = [args.examples, args.teacher_scores, args.model]
    if not all(path.exists() for path in paths):
        parser.error("examples, teacher scores, and model must exist")
    if args.initial_adapter is not None and not args.initial_adapter.exists():
        parser.error("initial adapter does not exist")
    if min(args.max_steps, args.max_entries, args.max_length, args.save_steps) < 1:
        parser.error("step, entry, length, and save settings must be positive")
    if args.learning_rate <= 0 or args.temperature <= 0:
        parser.error("learning rate and temperature must be positive")
    if args.hard_weight < 0 or args.token_weight < 0 or args.visual_score_window < 0:
        parser.error("loss weights and score window cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    configure_logging(args.output / "train.log")
    records = prepare_records(args.examples, args.teacher_scores, args.visual_score_window)
    random.Random(args.seed).shuffle(records)
    records = records[:args.max_entries]
    manifest = {
        "format": "competitive-word-decoder-distillation-0.1",
        "objective": "teacher KL plus hard reference ranking and reference token NLL",
        "examples_sha256": hashlib.sha256(args.examples.read_bytes()).hexdigest(),
        "teacher_scores_sha256": hashlib.sha256(args.teacher_scores.read_bytes()).hexdigest(),
        "records": len(records),
        "records_with_reference": sum(row["reference_index"] is not None for row in records),
        "selected_clip_ids": [row["clip_id"] for row in records],
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
    }
    manifest_path = args.output / "manifest.json"
    if args.resume:
        previous = json.loads(manifest_path.read_text())
        for key in ("examples_sha256", "teacher_scores_sha256", "selected_clip_ids"):
            if previous[key] != manifest[key]:
                raise ValueError(f"resume manifest mismatch: {key}")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    import torch
    import torch.nn.functional as functional
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer, set_seed

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but is unavailable")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    if args.initial_adapter is not None:
        model = PeftModel.from_pretrained(
            model, args.initial_adapter, local_files_only=True, is_trainable=True,
        )
    else:
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05,
            bias="none", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ))
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model = model.to(device).train()
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters()
                                   if parameter.requires_grad), lr=args.learning_rate)
    start_step = 0
    state_path = args.output / "training-state.pt"
    if args.resume:
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        LOGGER.info("resumed at step=%d", start_step)

    def save(step: int) -> None:
        adapter_path = args.output / "adapter"
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        torch.save({"step": step, "optimizer": optimizer.state_dict()}, state_path)
        LOGGER.info("checkpoint step=%d adapter=%s", step, adapter_path)

    LOGGER.info("device=%s records=%d reference_records=%d start_step=%d max_steps=%d",
                device, len(records), manifest["records_with_reference"], start_step,
                args.max_steps)
    for step in range(start_step, args.max_steps):
        while args.pause_file.exists():
            LOGGER.warning("paused by %s at step=%d", args.pause_file, step)
            for handler in LOGGER.handlers:
                handler.flush()
            time.sleep(5)
        record = records[step % len(records)]
        prefix = tokenizer.encode(record["prompt"], add_special_tokens=False)
        sequences, labels = [], []
        for text in record["candidate_texts"]:
            target = tokenizer.encode(" " + text, add_special_tokens=False)
            target.append(tokenizer.eos_token_id)
            complete = prefix + target
            if len(complete) > args.max_length:
                raise ValueError(f"sequence exceeds max length for {record['clip_id']}")
            sequences.append(complete)
            labels.append([-100] * len(prefix) + target)
        width = max(map(len, sequences))
        input_ids = torch.tensor([row + [tokenizer.pad_token_id] * (width - len(row))
                                  for row in sequences], device=device)
        attention = input_ids.ne(tokenizer.pad_token_id)
        target_ids = torch.tensor([row + [-100] * (width - len(row)) for row in labels],
                                  device=device)
        logits = model(input_ids=input_ids, attention_mask=attention).logits[:, :-1].float()
        shifted = target_ids[:, 1:]
        mask = shifted.ne(-100)
        log_probs = functional.log_softmax(logits, dim=-1)
        gathered = log_probs.gather(-1, shifted.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        student_scores = (gathered * mask).sum(1) / mask.sum(1)
        teacher_scores = torch.tensor(record["teacher_scores"], device=device)
        teacher_distribution = functional.softmax(teacher_scores / args.temperature, dim=0)
        distill_loss = functional.kl_div(
            functional.log_softmax(student_scores / args.temperature, dim=0),
            teacher_distribution, reduction="sum",
        ) * args.temperature**2
        hard_loss = torch.zeros((), device=device)
        token_loss = torch.zeros((), device=device)
        if record["reference_index"] is not None:
            reference_index = torch.tensor([record["reference_index"]], device=device)
            hard_loss = functional.cross_entropy(student_scores.unsqueeze(0), reference_index)
            token_loss = -student_scores[record["reference_index"]]
        loss = distill_loss + args.hard_weight * hard_loss + args.token_weight * token_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        completed = step + 1
        if completed == 1 or completed % 10 == 0:
            LOGGER.info("step=%d loss=%.5f kl=%.5f hard=%.5f token=%.5f candidates=%d",
                        completed, loss.item(), distill_loss.item(), hard_loss.item(),
                        token_loss.item(), len(sequences))
        if completed % args.save_steps == 0:
            save(completed)
    save(args.max_steps)
    LOGGER.info("training complete")
    for handler in LOGGER.handlers:
        handler.flush()


if __name__ == "__main__":
    main()

