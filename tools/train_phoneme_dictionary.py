"""LoRA-tune Qwen on CMUdict ARPAbet-to-word mappings."""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import cmudict
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForImageTextToText,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "checkpoints" / "Qwen3.5-2B"
DEFAULT_OUTPUT = PROJECT_ROOT / "checkpoints" / "qwen-phoneme-dictionary-lora"


def normalize_word(word: str) -> str:
    """Remove CMUdict's alternate-pronunciation suffix."""
    return re.sub(r"\(\d+\)$", "", word).lower()


def normalize_phones(phones: list[str]) -> str:
    return " ".join(re.sub(r"\d", "", phone) for phone in phones)


def prompt(phones: str) -> str:
    return (
        "Convert the visual speech phonemes into the most likely English word.\n"
        "Return only the word.\n"
        f"Top phoneme sequence: {phones}\n"
        "Answer:"
    )


def build_dataset(seed: int, max_entries: int | None) -> Dataset:
    entries = [
        {"prompt": prompt(normalize_phones(phones)), "target": normalize_word(word)}
        for word, phones in cmudict.entries()
        if word and phones
    ]
    random.Random(seed).shuffle(entries)
    if max_entries is not None:
        entries = entries[:max_entries]
    return Dataset.from_list(entries)


def tokenize(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    def encode(example):
        prompt_ids = tokenizer(example["prompt"], add_special_tokens=False).input_ids
        target_ids = tokenizer(
            " " + example["target"] + tokenizer.eos_token,
            add_special_tokens=False,
        ).input_ids
        prompt_ids = prompt_ids[: max_length - len(target_ids)]
        input_ids = (prompt_ids + target_ids)[:max_length]
        labels = ([-100] * len(prompt_ids) + target_ids)[:max_length]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    return dataset.map(encode, remove_columns=dataset.column_names)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-entries", type=int)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = tokenize(
        build_dataset(args.seed, args.max_entries), tokenizer, args.max_length
    )

    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output),
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=1,
            learning_rate=args.learning_rate,
            warmup_steps=0,
            logging_steps=1,
            save_strategy="no",
            bf16=False,
            fp16=False,
            report_to="none",
            remove_unused_columns=False,
            seed=args.seed,
        ),
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, label_pad_token_id=-100
        ),
        processing_class=tokenizer,
    )
    trainer.train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved phoneme dictionary adapter to {args.output}")


if __name__ == "__main__":
    main()
