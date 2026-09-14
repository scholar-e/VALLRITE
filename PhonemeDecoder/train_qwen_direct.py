"""Train a Qwen LoRA to generate words directly from an ARPAbet sequence."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from pathlib import Path
import sys

from PhonemeDecoder.evaluate_qwen_direct import prompt

LOGGER = logging.getLogger("phoneme_decoder.train_qwen_direct")
PROMPT_VERSION = "direct-arpabet-to-words-1"


def configure_logging(path: Path) -> None:
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.handlers.clear()
    for handler in (logging.FileHandler(path), logging.StreamHandler(sys.stdout)):
        handler.setLevel(logging.DEBUG if isinstance(handler, logging.FileHandler) else logging.INFO)
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def load_rows(path: Path) -> list[dict]:
    rows = []
    seen = set()
    for line in path.read_text().splitlines():
        record = json.loads(line)
        clip_id = record["clip_id"]
        if clip_id in seen:
            raise ValueError(f"duplicate clip_id: {clip_id}")
        seen.add(clip_id)
        if not record.get("phonemes") or not record.get("transcript", "").strip():
            continue
        rows.append({"clip_id": clip_id, "phones": record["phonemes"],
                     "target": record["transcript"].strip()})
    if not rows:
        raise ValueError("no usable phoneme/transcript records")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("checkpoints/Qwen3.5-2B"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--max-entries", type=int, default=7895)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    if (not args.manifest.is_file() or not args.model.is_dir()
            or min(args.max_steps, args.max_entries, args.max_length, args.batch_size,
                   args.gradient_accumulation, args.save_steps) < 1
            or args.learning_rate <= 0):
        parser.error("invalid paths or training settings")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    configure_logging(args.output / "train.log")

    rows = load_rows(args.manifest)
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.max_entries]
    manifest = {
        "scope": "direct ARPAbet-to-transcript LoRA training",
        "prompt_version": PROMPT_VERSION,
        "objective": "target-only token cross entropy; no lexicon or word candidates",
        "source_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "selected_clip_ids": [row["clip_id"] for row in rows],
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
    }
    manifest_path = args.output / "manifest.json"
    if args.resume:
        previous = json.loads(manifest_path.read_text())
        for key in ("source_sha256", "selected_clip_ids", "prompt_version"):
            if previous[key] != manifest[key]:
                raise ValueError(f"resume manifest mismatch: {key}")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    import torch
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but is unavailable")
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
        messages = [{"role": "user", "content": prompt(row["phones"])}]
        prefix = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
        )
        if hasattr(prefix, "input_ids"):
            prefix = prefix.input_ids
        target = tokenizer.encode(row["target"], add_special_tokens=False)
        target.append(tokenizer.eos_token_id)
        complete = prefix + target
        if len(complete) > args.max_length:
            continue
        encoded.append({"input_ids": complete, "attention_mask": [1] * len(complete),
                        "labels": [-100] * len(prefix) + target})
    if not encoded:
        raise ValueError("all examples exceeded max length")
    LOGGER.info("encoded=%d selected=%d device=%s", len(encoded), len(rows),
                "cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05,
        bias="none", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    ))

    class ProgressLog(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            LOGGER.info("step=%d metrics=%s", state.global_step, logs)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output), max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate, bf16=torch.cuda.is_available(),
            logging_steps=10, save_strategy="steps", save_steps=args.save_steps,
            save_total_limit=3, report_to="none", remove_unused_columns=False,
            seed=args.seed, dataloader_num_workers=2,
            dataloader_pin_memory=torch.cuda.is_available(), disable_tqdm=True,
        ),
        train_dataset=Dataset.from_list(encoded), processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100),
        callbacks=[ProgressLog()],
    )
    result = trainer.train(resume_from_checkpoint=True if args.resume else None)
    trainer.save_model(str(args.output / "adapter"))
    tokenizer.save_pretrained(args.output / "adapter")
    (args.output / "metrics.json").write_text(json.dumps(result.metrics, indent=2) + "\n")
    LOGGER.info("complete adapter=%s", args.output / "adapter")


if __name__ == "__main__":
    main()
