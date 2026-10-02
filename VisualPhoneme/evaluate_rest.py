"""Compare a REST-aware checkpoint with its baseline on aligned LRS3 clips."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

from VisualPhoneme.data import collate_aligned_clips, greedy_decode
from VisualPhoneme.evaluate_nbest import load_model
from VisualPhoneme.lrs3_data import Lrs3Clips
from VisualPhoneme.train import edit_totals, upsample_ctc_logits
from VisualPhoneme.visemes import (PHONE_ID_TO_VISUAL_GROUP_ID,
                                   PHONE_ID_TO_VISUAL_GROUP_ID_WITH_REST,
                                   REST_GROUP_ID, REST_PHONE_ID)

LOGGER = logging.getLogger("visual_phoneme.evaluate_rest")


def configure_logging(path: Path) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--rest-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/lrs3"))
    parser.add_argument("--teacher-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--min-rest-seconds", type=float, default=0.08)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if (not args.baseline.is_file() or not args.rest_checkpoint.is_file()
            or args.batch_size < 1 or args.workers < 0 or args.min_rest_seconds < 0):
        parser.error("invalid checkpoint, batch, workers, or REST threshold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    configure_logging(args.output.with_suffix(".log"))
    device = torch.device(args.device)
    baseline_checkpoint = torch.load(args.baseline, map_location="cpu", weights_only=True)
    rest_checkpoint = torch.load(args.rest_checkpoint, map_location="cpu", weights_only=True)
    if (baseline_checkpoint.get("target_inventory") != "visual-groups"
            or rest_checkpoint.get("target_inventory") != "visual-groups-rest"):
        raise ValueError("expected visual-groups baseline and visual-groups-rest checkpoint")
    _, baseline = load_model(baseline_checkpoint, device)
    _, rest_model = load_model(rest_checkpoint, device)
    dataset = Lrs3Clips(
        args.data_root, "validation", int(rest_checkpoint["image_size"]),
        rest_checkpoint["crop"], None, False, False, True, False, "eye-normalized",
        None, "position", teacher_labels_dir=args.teacher_labels, max_chunk_phones=40,
        min_teacher_transcript_agreement=0.8, include_frame_targets=True,
        include_rest_targets=True, min_rest_seconds=args.min_rest_seconds,
    )
    loader = DataLoader(dataset, args.batch_size, num_workers=args.workers,
                        collate_fn=collate_aligned_clips, pin_memory=device.type == "cuda",
                        persistent_workers=args.workers > 0)
    totals = {"phones": 0, "baseline_errors": 0, "rest_stripped_errors": 0,
              "rest_inclusive_tokens": 0, "rest_inclusive_errors": 0,
              "tp": 0, "fp": 0, "fn": 0, "tn": 0}
    phone_map = PHONE_ID_TO_VISUAL_GROUP_ID
    rest_map = PHONE_ID_TO_VISUAL_GROUP_ID_WITH_REST
    with torch.inference_mode():
        for batch_index, (video, packed, lengths, target_lengths,
                          frame_targets, _) in enumerate(loader, 1):
            video = video.to(device, non_blocking=True)
            baseline_logits = baseline(video)
            rest_logits = rest_model(video)
            frame_predictions = rest_logits.argmax(-1).T.cpu()
            valid_frames = frame_targets >= 0
            mapped_frames = torch.full_like(frame_targets, -100)
            mapped_frames[valid_frames] = torch.tensor(rest_map)[frame_targets[valid_frames]]
            actual_rest = mapped_frames == REST_GROUP_ID
            predicted_rest = frame_predictions == REST_GROUP_ID
            totals["tp"] += int((actual_rest & predicted_rest).sum())
            totals["fp"] += int((~actual_rest & predicted_rest & valid_frames).sum())
            totals["fn"] += int((actual_rest & ~predicted_rest).sum())
            totals["tn"] += int((~actual_rest & ~predicted_rest & valid_frames).sum())
            baseline_logits, baseline_lengths = upsample_ctc_logits(
                baseline_logits, lengths,
                int(baseline_checkpoint.get("ctc_upsample_factor", 1)))
            rest_logits, rest_lengths = upsample_ctc_logits(
                rest_logits, lengths, int(rest_checkpoint.get("ctc_upsample_factor", 1)))
            baseline_hypotheses = greedy_decode(baseline_logits, baseline_lengths)
            rest_hypotheses = greedy_decode(rest_logits, rest_lengths)
            offset = 0
            for index, target_length in enumerate(target_lengths.tolist()):
                raw = packed[offset:offset + target_length].tolist()
                offset += target_length
                reference = [phone_map[token] for token in raw if token != REST_PHONE_ID]
                reference_with_rest = [rest_map[token] for token in raw]
                rest_stripped = [token for token in rest_hypotheses[index]
                                 if token != REST_GROUP_ID]
                totals["baseline_errors"] += edit_totals(
                    reference, baseline_hypotheses[index])[0]
                totals["rest_stripped_errors"] += edit_totals(reference, rest_stripped)[0]
                totals["rest_inclusive_errors"] += edit_totals(
                    reference_with_rest, rest_hypotheses[index])[0]
                totals["phones"] += len(reference)
                totals["rest_inclusive_tokens"] += len(reference_with_rest)
            if batch_index % 5 == 0:
                LOGGER.info("processed clips=%d/%d", min(batch_index * args.batch_size,
                                                         len(dataset)), len(dataset))
    precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
    recall = totals["tp"] / max(totals["tp"] + totals["fn"], 1)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "clips": len(dataset),
        "baseline": str(args.baseline.resolve()),
        "rest_checkpoint": str(args.rest_checkpoint.resolve()),
        "min_rest_seconds": args.min_rest_seconds,
        "baseline_stripped_greedy_per": totals["baseline_errors"] / totals["phones"],
        "rest_model_stripped_greedy_per": totals["rest_stripped_errors"] / totals["phones"],
        "rest_model_inclusive_greedy_per": (
            totals["rest_inclusive_errors"] / totals["rest_inclusive_tokens"]),
        "rest_frame_precision": precision, "rest_frame_recall": recall,
        "rest_frame_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "counts": totals,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    LOGGER.info("baseline stripped PER=%.2f%% REST-model stripped PER=%.2f%% REST F1=%.2f%%",
                100 * report["baseline_stripped_greedy_per"],
                100 * report["rest_model_stripped_greedy_per"],
                100 * report["rest_frame_f1"])
    for handler in LOGGER.handlers:
        handler.flush()


if __name__ == "__main__":
    main()
