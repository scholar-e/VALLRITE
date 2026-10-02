"""Evaluate the Qwen phoneme-to-text recipe found in VALLR/Models/Llama.py.

The upstream repository contains a training recipe, but no trained adapter.  With
no ``--adapter`` this program therefore measures the unadapted Qwen base model;
it must not be reported as the performance of a released VALLR word decoder.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import sys
import time

LOGGER = logging.getLogger("competitive_word_decoder.vallr_qwen")
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)*")


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


def vallr_prompt(phones: list[str]) -> str:
    """Reproduce the prompt format used by the upstream VALLR training recipe."""
    return "<S2S>\n<PHONEMES>\n" + " ".join(phones) + "\n</PHONEMES>\n<TEXT>\n"


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, 1):
        current = [row]
        for column, actual in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (expected != actual)))
        previous = current
    return previous[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypotheses", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("checkpoints/Qwen3.5-2B"))
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    if not args.hypotheses.is_file() or not args.model.is_dir():
        parser.error("hypotheses and local model must exist")
    if args.adapter is not None and not args.adapter.exists():
        parser.error("adapter does not exist")
    if args.batch_size < 1 or args.max_new_tokens < 1 or (args.limit is not None and args.limit < 1):
        parser.error("batch size, token count, and limit must be positive")
    return args


def main() -> None:
    args = parse_args()
    configure_logging(args.output.with_suffix(".log"))
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but is unavailable")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    if args.adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True)
    model = model.to(device).eval()

    records = [json.loads(line) for line in args.hypotheses.read_text().splitlines() if line.strip()]
    if args.limit is not None:
        records = records[:args.limit]
    items = []
    for record_index, record in enumerate(records):
        for hypothesis_index, hypothesis in enumerate(record["phone_hypotheses"]):
            items.append((record_index, hypothesis_index, vallr_prompt(hypothesis["phones"])))
    generated: list[list[str | None]] = [
        [None] * len(record["phone_hypotheses"]) for record in records
    ]
    baseline_kind = "local-adapter" if args.adapter else "unadapted-base-diagnostic"
    LOGGER.info("device=%s records=%d generations=%d baseline=%s", device, len(records),
                len(items), baseline_kind)
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(items), args.batch_size):
            batch = items[start:start + args.batch_size]
            inputs = tokenizer([item[2] for item in batch], return_tensors="pt", padding=True).to(device)
            output_ids = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
            continuations = output_ids[:, inputs["input_ids"].shape[1]:]
            texts = tokenizer.batch_decode(continuations, skip_special_tokens=True)
            for (record_index, hypothesis_index, _), text in zip(batch, texts, strict=True):
                generated[record_index][hypothesis_index] = text.strip()
            LOGGER.info("generated=%d/%d", min(start + len(batch), len(items)), len(items))

    total_words = top_errors = oracle_errors = exact_hits = 0
    details = []
    for record, texts in zip(records, generated, strict=True):
        reference = record["reference_words"]
        hypotheses = []
        errors = []
        for phone_hypothesis, text in zip(record["phone_hypotheses"], texts, strict=True):
            if text is None:
                raise RuntimeError("generation missing")
            decoded = words(text)
            error = edit_distance(reference, decoded)
            errors.append(error)
            hypotheses.append({"phone_rank": phone_hypothesis["rank"], "text": text,
                               "words": decoded, "word_errors": error})
        total_words += len(reference)
        top_errors += errors[0]
        oracle_errors += min(errors)
        exact_hits += int(min(errors) == 0)
        details.append({"clip_id": record["clip_id"], "reference_words": reference,
                        "hypotheses": hypotheses})
    elapsed = time.perf_counter() - started
    n = max((len(row["phone_hypotheses"]) for row in records), default=0)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "format": "competitive-word-decoder-vallr-qwen-eval-0.1",
        "baseline_kind": baseline_kind,
        "warning": (None if args.adapter else
                    "VALLR did not provide its trained LoRA; this is the unadapted base model, not a released decoder."),
        "prompt_source": "VALLR/Models/Llama.py",
        "model": str(args.model.resolve()),
        "adapter": str(args.adapter.resolve()) if args.adapter else None,
        "records": len(records), "reference_words": total_words,
        "top_1": {"word_errors": top_errors, "wer": top_errors / total_words},
        "oracle_at_n": {"n": n, "word_errors": oracle_errors,
                        "wer": oracle_errors / total_words,
                        "exact_hits": exact_hits,
                        "exact_accuracy": exact_hits / len(records)},
        "elapsed_seconds": elapsed, "records_detail": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    LOGGER.info("complete top1_WER=%.2f%% oracle_WER@%d=%.2f%% elapsed=%.1fs",
                100 * report["top_1"]["wer"], n, 100 * report["oracle_at_n"]["wer"], elapsed)
    for handler in LOGGER.handlers:
        handler.flush()


if __name__ == "__main__":
    main()

