"""Generate transcripts from visual-phone N-best records and report WER."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time

from CompetitiveWordDecoder.evaluate_vallr_qwen import edit_distance, words
from CompetitiveWordDecoder.noisy_teacher import PROMPT_VERSION, load_nbest, nbest_prompt

LOGGER = logging.getLogger("competitive_word_decoder.noisy_teacher_eval")


def configure_logging(path: Path) -> None:
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.handlers.clear()
    for handler in (logging.FileHandler(path, delay=False), logging.StreamHandler(sys.stdout)):
        handler.setLevel(logging.DEBUG if isinstance(handler, logging.FileHandler) else logging.INFO)
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypotheses", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("checkpoints/Qwen3.5-2B"))
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    if not args.hypotheses.is_file() or not args.model.is_dir() or not args.adapter.exists():
        parser.error("hypotheses, model, and adapter must exist")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    configure_logging(args.output.with_suffix(".log"))
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but is unavailable")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForImageTextToText.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter, local_files_only=True).to(device).eval()
    rows = load_nbest(args.hypotheses)
    if args.limit:
        rows = rows[:args.limit]
    prompts = []
    for row in rows:
        prompts.append(tokenizer.apply_chat_template(
            [{"role": "user", "content": nbest_prompt(row)}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        ))
    details = []
    total_words = total_errors = exact = 0
    started = time.perf_counter()
    LOGGER.info("device=%s records=%d", device, len(rows))
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start:start + args.batch_size]
            inputs = tokenizer(prompts[start:start + args.batch_size], return_tensors="pt",
                               padding=True).to(device)
            output = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
            texts = tokenizer.batch_decode(output[:, inputs["input_ids"].shape[1]:],
                                           skip_special_tokens=True)
            for row, text in zip(batch_rows, texts, strict=True):
                hypothesis = words(text)
                error = edit_distance(row["reference_words"], hypothesis)
                total_words += len(row["reference_words"])
                total_errors += error
                exact += int(error == 0)
                details.append({"clip_id": row["clip_id"], "reference_words": row["reference_words"],
                                "text": text.strip(), "words": hypothesis, "word_errors": error})
            if start == 0 or (start // args.batch_size) % 20 == 0:
                LOGGER.info("generated=%d/%d", min(start + len(batch_rows), len(rows)), len(rows))
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "format": "competitive-word-decoder-noisy-teacher-eval-0.1",
        "prompt_version": PROMPT_VERSION, "records": len(rows),
        "reference_words": total_words, "word_errors": total_errors,
        "wer": total_errors / total_words, "exact_hits": exact,
        "exact_accuracy": exact / len(rows), "elapsed_seconds": time.perf_counter() - started,
        "model": str(args.model.resolve()), "adapter": str(args.adapter.resolve()),
        "records_detail": details,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    LOGGER.info("complete WER=%.2f%% exact=%.2f%%", 100 * report["wer"],
                100 * report["exact_accuracy"])
    for handler in LOGGER.handlers:
        handler.flush()


if __name__ == "__main__":
    main()

