"""Evaluate released VALLR on forced-aligned LRS3 validation word segments."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import random
import re
import sys
import time

import pronouncing
import torch
from torch.nn import functional as F

from tools.evaluate_original_vallr_per import (ctc_collapse, edit_counts, load_model,
                                                load_segment, normalize_phone,
                                                phone_names)

LOGGER = logging.getLogger("original_vallr_lrs3")


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


def instances(data_root: Path, teacher_labels: Path) -> list[dict]:
    validation_sources = {
        str((data_root / json.loads(line)["video"]).resolve())
        for line in (data_root / "manifests/validation.jsonl").read_text().splitlines()
    }
    result = []
    for line in (teacher_labels / "labels.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record.get("source") not in validation_sources or not record.get("alignment"):
            continue
        document = json.loads(Path(record["alignment"]).read_text())
        if document.get("label_status") != "accepted":
            continue
        phone_entries = document["tiers"]["phones"]["entries"]
        for word_index, (start, end, word) in enumerate(document["tiers"]["words"]["entries"]):
            reference = []
            for phone_start, phone_end, raw_phone in phone_entries:
                midpoint = (float(phone_start) + float(phone_end)) / 2
                phone = normalize_phone(str(raw_phone))
                if float(start) <= midpoint <= float(end) and phone is not None:
                    from tools.evaluate_original_vallr_per import PHONE_TO_ID
                    reference.append(PHONE_TO_ID[phone])
            if reference:
                result.append({"source": record["source"], "word_index": word_index,
                               "word": str(word).lower(), "start": float(start),
                               "end": float(end), "reference": reference})
    return result


def pronunciation_matches(word: str, hypothesis: list[int]) -> bool:
    from tools.evaluate_original_vallr_per import PHONE_TO_ID
    for raw in pronouncing.phones_for_word(re.sub(r"[^a-z']", "", word.lower())):
        phones = [normalize_phone(phone) for phone in raw.split()]
        sequence = [PHONE_TO_ID[phone] for phone in phones if phone is not None]
        if sequence == hypothesis:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/lrs3"))
    parser.add_argument("--teacher-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-words", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--crop", choices=("face", "mouth", "full"), default="face")
    parser.add_argument("--normalization", choices=("raw", "zero-one", "minus-one-one"),
                        default="raw")
    parser.add_argument("--context-seconds", type=float, default=0.0)
    parser.add_argument("--comparison-checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if not args.checkpoint.is_file() or args.max_words < 1:
        parser.error("invalid checkpoint or sample size")
    configure_logging(args.output.with_suffix(".log"))
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    comparison = comparison_checkpoint = None
    if args.comparison_checkpoint:
        from VisualPhoneme.evaluate_nbest import load_model as load_visual_model
        comparison_checkpoint = torch.load(args.comparison_checkpoint, map_location="cpu",
                                           weights_only=True)
        _, comparison = load_visual_model(comparison_checkpoint, device)
    samples = instances(args.data_root, args.teacher_labels)
    random.Random(args.seed).shuffle(samples)
    samples = samples[:args.max_words]
    substitutions = deletions = insertions = phones = exact_words = 0
    original_group_errors = comparison_group_errors = group_tokens = 0
    records = []
    started = time.perf_counter()
    with torch.inference_mode():
        for index, item in enumerate(samples, 1):
            video = load_segment(Path(item["source"]), item["start"], item["end"],
                                 args.crop, args.normalization, args.context_seconds).to(device)
            logits, _ = model(video)
            hypothesis = ctc_collapse(logits.argmax(-1)[0].tolist())
            sub, delete, insert = edit_counts(item["reference"], hypothesis)
            substitutions += sub; deletions += delete; insertions += insert
            phones += len(item["reference"])
            matched = pronunciation_matches(item["word"], hypothesis)
            exact_words += int(matched)
            if comparison is not None:
                from VisualPhoneme.data import greedy_decode
                from VisualPhoneme.train import upsample_ctc_logits
                from VisualPhoneme.visemes import (PHONE_ID_TO_VISUAL_GROUP_ID,
                                                   REST_GROUP_ID)
                current_video = load_segment(
                    Path(item["source"]), item["start"], item["end"], "mouth",
                    "zero-one", args.context_seconds).to(device)
                current_video = (0.2989 * current_video[:, :, 0]
                                 + 0.5870 * current_video[:, :, 1]
                                 + 0.1140 * current_video[:, :, 2]).unsqueeze(2)
                current_video = F.interpolate(
                    current_video.flatten(0, 1), size=(96, 96), mode="bilinear",
                    align_corners=False).reshape(1, 16, 1, 96, 96)
                current_logits = comparison(current_video)
                current_logits, current_lengths = upsample_ctc_logits(
                    current_logits, torch.tensor([16], device=device),
                    int(comparison_checkpoint.get("ctc_upsample_factor", 1)))
                current_hypothesis = greedy_decode(current_logits, current_lengths)[0]
                current_hypothesis = [token for token in current_hypothesis
                                      if token != REST_GROUP_ID]
                group_reference = [PHONE_ID_TO_VISUAL_GROUP_ID[token]
                                   for token in item["reference"]]
                original_groups = [PHONE_ID_TO_VISUAL_GROUP_ID[token]
                                   for token in hypothesis]
                original_group_errors += sum(edit_counts(group_reference, original_groups))
                comparison_group_errors += sum(edit_counts(
                    group_reference, current_hypothesis))
                group_tokens += len(group_reference)
            records.append({"word": item["word"], "reference": phone_names(item["reference"]),
                            "hypothesis": phone_names(hypothesis),
                            "pronunciation_exact": matched})
            if index % 20 == 0:
                LOGGER.info("evaluated words=%d/%d", index, len(samples))
    errors = substitutions + deletions + insertions
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "LRS3 validation forced-aligned isolated word segments",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "words": len(samples), "reference_phones": phones,
        "substitutions": substitutions, "deletions": deletions, "insertions": insertions,
        "per": errors / phones, "pronunciation_exact_words": exact_words,
        "pronunciation_exact_word_accuracy": exact_words / len(samples),
        "pronunciation_exact_wer": 1 - exact_words / len(samples),
        "original_visual_group_per": (original_group_errors / group_tokens
                                      if group_tokens else None),
        "comparison_checkpoint": (str(args.comparison_checkpoint.resolve())
                                  if args.comparison_checkpoint else None),
        "comparison_visual_group_per": (comparison_group_errors / group_tokens
                                        if group_tokens else None),
        "preprocessing": {"frames": 16, "size": 224, "crop": args.crop,
                          "normalization": args.normalization,
                          "context_seconds_each_side": args.context_seconds},
        "limitations": "Oracle word boundaries; pronunciation-exact WER is not sentence WER.",
        "seconds": time.perf_counter() - started, "records": records,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    LOGGER.info("PER=%.2f%% pronunciation-exact WER=%.2f%% words=%d seconds=%.1f",
                100 * report["per"], 100 * report["pronunciation_exact_wer"],
                len(samples), report["seconds"])
    for handler in LOGGER.handlers:
        handler.flush()


if __name__ == "__main__":
    main()
