"""Experimental supervised LoRA warm-up for candidate-only Qwen scoring.

Only train-speaker records whose reference already occurs in the beam are used.
This is not a validation benchmark or the production reranking integration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from pathlib import Path

LOGGER = logging.getLogger(__name__)
PROMPT_VERSION = "phoneme-candidate-sft-1"


def candidate_prompt(record):
    # No reference phones/words or oracle annotations may enter the prompt.
    return (
        "Select the English transcript supported by the observed ARPAbet phonemes.\n"
        "Return one of the candidate transcripts.\nObserved phonemes: "
        + " ".join(record["greedy_visual_phones"])
        + "\nCandidates: "
        + json.dumps([c["text"] for c in record["candidates"]], ensure_ascii=False)
        + "\nAnswer:"
    )


def training_rows(path):
    rows = []
    seen = set()
    skipped = 0
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record["split"] != "train" or record["speaker_id"] not in range(1, 9):
            raise ValueError("Training input must contain only train speakers 1–8")
        if record["clip_id"] in seen:
            raise ValueError("Duplicate training clip")
        seen.add(record["clip_id"])
        matches = [c for c in record["candidates"]
                   if c["words"] == record["reference_words"]]
        if not matches:
            skipped += 1
            continue
        rows.append({"prompt": candidate_prompt(record), "target": matches[0]["text"],
                     "clip_id": record["clip_id"]})
    if not rows:
        raise ValueError("No references available in the candidate beam")
    return rows, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("checkpoints/Qwen3.5-2B"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-entries", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    if min(args.max_steps, args.max_entries, args.max_length) <= 0:
        parser.error("steps, entries, and length must be positive")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s",
        handlers=[logging.FileHandler(args.output / "train.log"), logging.StreamHandler()])
    rows, skipped = training_rows(args.examples)
    eligible = len(rows)
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.max_entries]
    manifest = {
        "scope": "in-sample supervised candidate-scoring warm-up; not held-out accuracy",
        "prompt_version": PROMPT_VERSION,
        "objective": "target-only token cross entropy, including exactly one EOS",
        "scoring_convention": "mean target token log likelihood including EOS",
        "examples_sha256": hashlib.sha256(args.examples.read_bytes()).hexdigest(),
        "eligible_records": eligible, "skipped_reference_absent": skipped,
        "selected_clip_ids": [r["clip_id"] for r in rows],
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    manifest_path = args.output / "manifest.json"
    if args.resume:
        previous = json.loads(manifest_path.read_text())
        for key in ("examples_sha256", "selected_clip_ids", "prompt_version"):
            if previous[key] != manifest[key]:
                raise ValueError(f"Resume manifest mismatch: {key}")
        for key in ("model", "max_steps", "max_length", "seed"):
            if previous["arguments"][key] != manifest["arguments"][key]:
                raise ValueError(f"Resume argument mismatch: {key}")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    LOGGER.info("Selected %d of %d eligible records; skipped %d absent references",
                len(rows), eligible, skipped)
    if args.prepare_only:
        return

    import torch
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but is unavailable; refusing CPU fallback")
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (AutoModelForImageTextToText, AutoTokenizer,
                              DataCollatorForSeq2Seq, Trainer, TrainerCallback,
                              TrainingArguments, set_seed)
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = []
    for row in rows:
        prefix = tokenizer.encode(row["prompt"], add_special_tokens=False)
        target = tokenizer.encode(" " + row["target"], add_special_tokens=False)
        target.append(tokenizer.eos_token_id)
        if len(prefix) + len(target) > args.max_length:
            raise ValueError("Example exceeds max length; increase --max-length (no truncation)")
        encoded.append({"input_ids": prefix + target,
                        "attention_mask": [1] * (len(prefix) + len(target)),
                        "labels": [-100] * len(prefix) + target})
    LOGGER.info("Loading local model; device=%s", "cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM,
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))

    class ProgressLog(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            LOGGER.info("step=%d metrics=%s", state.global_step, logs)

    trainer = Trainer(model=model, args=TrainingArguments(
        output_dir=str(args.output), max_steps=args.max_steps,
        per_device_train_batch_size=1, learning_rate=2e-4,
        logging_steps=1, save_strategy="steps", save_steps=5, save_total_limit=2,
        report_to="none", remove_unused_columns=False, seed=args.seed,
        dataloader_num_workers=0, dataloader_pin_memory=torch.cuda.is_available(),
        disable_tqdm=True),
        train_dataset=Dataset.from_list(encoded), processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100),
        callbacks=[ProgressLog()])
    LOGGER.info("Training device=%s; resume=%s", trainer.args.device, args.resume)
    result = trainer.train(resume_from_checkpoint=True if args.resume else None)
    trainer.save_model(str(args.output / "adapter"))
    tokenizer.save_pretrained(args.output / "adapter")
    (args.output / "metrics.json").write_text(json.dumps(result.metrics, indent=2) + "\n")
    LOGGER.info("Training complete; adapter saved to %s", args.output / "adapter")


if __name__ == "__main__":
    main()
