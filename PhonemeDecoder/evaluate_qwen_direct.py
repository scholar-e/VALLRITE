"""Directly generate text from phoneme N-best sequences and measure WER@N."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import sys
import time

LOGGER = logging.getLogger("phoneme_decoder.evaluate_qwen_direct")


def configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.handlers.clear()
    for handler in (logging.FileHandler(path, delay=False), logging.StreamHandler(sys.stdout)):
        handler.setLevel(logging.DEBUG if isinstance(handler, logging.FileHandler)
                         else logging.INFO)
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, 1):
        current = [row]
        for column, actual in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (expected != actual)))
        previous = current
    return previous[-1]


def prompt(phones: list[str]) -> str:
    return (
        "Convert the visual speech phonemes into the most likely English sentence.\n"
        "Return only the sentence.\n"
        "Top phoneme sequence: " + " ".join(phones) + "\nAnswer:"
    )


def format_prompt(tokenizer, phones: list[str]) -> str:
    """Render an instruction-tuned prompt while suppressing Qwen's reasoning trace."""
    messages = [{"role": "user", "content": prompt(phones)}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


def words(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)*", text.lower())


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
    if (not args.hypotheses.is_file() or not args.model.is_dir()
            or (args.adapter is not None and not args.adapter.exists())
            or args.batch_size < 1 or args.max_new_tokens < 1
            or (args.limit is not None and args.limit < 1)):
        parser.error("invalid hypotheses, model, adapter, batch, token, or limit setting")
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
        args.model, local_files_only=True, dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    if args.adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True)
    model = model.to(device).eval()
    records = [json.loads(line) for line in args.hypotheses.read_text().splitlines()
               if line.strip()]
    if args.limit:
        records = records[:args.limit]
    items = []
    for record_index, record in enumerate(records):
        for hypothesis_index, hypothesis in enumerate(record["phone_hypotheses"]):
            items.append((record_index, hypothesis_index,
                          format_prompt(tokenizer, hypothesis["phones"])))
    generated_by_record: list[list[str | None]] = [
        [None] * len(record["phone_hypotheses"]) for record in records
    ]
    LOGGER.info("device=%s records=%d phone_hypotheses=%d model=%s adapter=%s",
                device, len(records), len(items), args.model, args.adapter)
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(items), args.batch_size):
            batch = items[start:start + args.batch_size]
            inputs = tokenizer([item[2] for item in batch], return_tensors="pt",
                               padding=True).to(device)
            generated = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            continuation = generated[:, inputs["input_ids"].shape[1]:]
            texts = tokenizer.batch_decode(continuation, skip_special_tokens=True)
            for (record_index, hypothesis_index, _), text in zip(batch, texts, strict=True):
                generated_by_record[record_index][hypothesis_index] = text.strip()
            if start == 0 or (start // args.batch_size) % 20 == 0:
                LOGGER.info("generated=%d/%d", min(start + len(batch), len(items)),
                            len(items))
    total_words = top_errors = oracle_errors = exact_hits = 0
    output_records = []
    for record, generated_texts in zip(records, generated_by_record, strict=True):
        reference = record["reference_words"]
        hypotheses = []
        errors = []
        for phone_hypothesis, text in zip(record["phone_hypotheses"],
                                          generated_texts, strict=True):
            if text is None:
                raise RuntimeError("missing generated hypothesis")
            decoded_words = words(text)
            error = edit_distance(reference, decoded_words)
            errors.append(error)
            hypotheses.append({
                "phone_rank": phone_hypothesis["rank"],
                "phones": phone_hypothesis["phones"],
                "text": text,
                "words": decoded_words,
                "word_errors": error,
            })
        top_errors += errors[0]
        oracle_errors += min(errors)
        exact_hits += min(errors) == 0
        total_words += len(reference)
        output_records.append({
            "clip_id": record["clip_id"],
            "reference_words": reference,
            "hypotheses": hypotheses,
        })
    elapsed = time.perf_counter() - started
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "format": "direct-phoneme-qwen-wer-0.1",
        "scope": "LRS3 development validation",
        "model": str(args.model.resolve()),
        "adapter": str(args.adapter.resolve()) if args.adapter else None,
        "decoder_input": "five CTC phoneme sequences; no lexical candidate stage",
        "records": len(records),
        "reference_words": total_words,
        "top_1": {"word_errors": top_errors, "wer": top_errors / total_words},
        "oracle_at_n": {"n": max(len(row["phone_hypotheses"]) for row in records),
                        "word_errors": oracle_errors,
                        "wer": oracle_errors / total_words,
                        "exact_hits": exact_hits,
                        "exact_accuracy": exact_hits / len(records)},
        "elapsed_seconds": elapsed,
        "records_detail": output_records,
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    LOGGER.info("complete top1_WER=%.2f%% oracle_WER@%d=%.2f%% exact=%.2f%% seconds=%.1f",
                100 * report["top_1"]["wer"], report["oracle_at_n"]["n"],
                100 * report["oracle_at_n"]["wer"],
                100 * report["oracle_at_n"]["exact_accuracy"], elapsed)
    for handler in LOGGER.handlers:
        handler.flush()


if __name__ == "__main__":
    main()
