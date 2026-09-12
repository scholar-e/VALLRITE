"""Evaluate oracle PER and exact recall across CTC candidate-list depths."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader

from VisualPhoneme.data import (
    GridClips, PHONEMES, collate_clips, collate_fusion_clips,
    collate_landmark_clips, ctc_prefix_beam_search, fit_bigram_log_probs,
    greedy_decode, landmark_feature_dimensions, phoneme_target,
)
from VisualPhoneme.model import (
    CompactFusionVisualPhoneme, CompactGatedFusionVisualPhoneme,
    CompactLandmarkPhoneme, CompactVisualPhoneme,
)
from VisualPhoneme.train import edit_totals

LOGGER = logging.getLogger("visual_phoneme.nbest")


def configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.handlers.clear()
    file_handler = logging.FileHandler(path, delay=False)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(console_handler)


def load_model(checkpoint: dict, device: torch.device):
    architecture = checkpoint.get("architecture", "image")
    classes = {"image": CompactVisualPhoneme, "coordinates": CompactLandmarkPhoneme,
               "fusion": CompactFusionVisualPhoneme,
               "gated-fusion": CompactGatedFusionVisualPhoneme}
    arguments = {"classes": len(PHONEMES) + 1}
    if architecture != "image":
        arguments.update({"landmark_points": int(checkpoint["landmark_points"]),
                          "coordinate_dimensions": int(checkpoint.get("coordinate_dimensions", 2))})
    if architecture in {"coordinates", "gated-fusion"}:
        arguments["landmark_bottleneck"] = checkpoint.get("landmark_bottleneck")
    model = classes[architecture](**arguments).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return architecture, model


def training_bigram(data_root: Path, smoothing: float) -> torch.Tensor:
    rows = [json.loads(line) for line in (data_root / "clips.jsonl").read_text().splitlines()]
    sequences = []
    for row in rows:
        if row["split"] != "train":
            continue
        stem = Path(row["video"]).stem
        alignment = (data_root / "landmark-experiment" / "aligned"
                     / f"s{row['speaker_id']}" / f"{stem}.json")
        sequences.append(phoneme_target(str(alignment)))
    return fit_bigram_log_probs(sequences, len(PHONEMES) + 1, smoothing)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/grid-pilot"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--n-values", default="1,3,5,10,20")
    parser.add_argument("--lm-weights", default="0")
    parser.add_argument("--lm-smoothing", type=float, default=1.0)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--beam-token-top-k", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--frame-cache-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    n_values = sorted({int(value) for value in args.n_values.split(",")})
    lm_weights = [float(value) for value in args.lm_weights.split(",")]
    if (not args.checkpoint.is_file() or not n_values or n_values[0] < 1
            or args.beam_width < n_values[-1] or args.beam_token_top_k < 1
            or any(weight < 0 for weight in lm_weights) or args.lm_smoothing <= 0):
        parser.error("invalid checkpoint, N values, beam, or language-model setting")
    output = args.output or args.checkpoint.with_name("oracle-curve.json")
    configure_logging(output.with_suffix(".log"))
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device("cpu" if device_name == "auto" else device_name)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if tuple(checkpoint["phones"]) != PHONEMES:
        raise ValueError("checkpoint phoneme vocabulary does not match this code")
    architecture, model = load_model(checkpoint, device)
    coordinate_mode = checkpoint.get("coordinate_mode", "eye-normalized")
    coordinate_features = checkpoint.get("coordinate_features", "position")
    if landmark_feature_dimensions(coordinate_features) != int(
            checkpoint.get("coordinate_dimensions", 2)):
        raise ValueError("checkpoint coordinate dimensions are inconsistent")
    include_landmarks = architecture != "image"
    include_video = architecture in {"image", "fusion", "gated-fusion"}
    validation = GridClips(
        args.data_root, "validation", int(checkpoint["image_size"]), checkpoint["crop"],
        None, False, include_landmarks, include_video, False, coordinate_mode,
        args.frame_cache_dir, coordinate_features,
    )
    collate = {"image": collate_clips, "coordinates": collate_landmark_clips,
               "fusion": collate_fusion_clips, "gated-fusion": collate_fusion_clips}
    loader = DataLoader(validation, batch_size=args.batch_size, num_workers=args.workers,
                        collate_fn=collate[architecture], pin_memory=device.type == "cuda",
                        persistent_workers=args.workers > 0)
    records = []
    greedy_errors = phones = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, 1):
            if architecture in {"fusion", "gated-fusion"}:
                video, landmarks, mask, targets, lengths, target_lengths, clip_ids = batch
                logits = model(video.to(device), landmarks.to(device), mask.to(device))
            elif architecture == "coordinates":
                landmarks, mask, targets, lengths, target_lengths, clip_ids = batch
                logits = model(landmarks.to(device), mask.to(device))
            else:
                video, targets, lengths, target_lengths, clip_ids = batch
                logits = model(video.to(device))
            log_probabilities = logits.log_softmax(-1).transpose(0, 1).cpu()
            hypotheses = greedy_decode(logits, lengths)
            offset = 0
            for index, target_length in enumerate(target_lengths):
                reference = targets[offset:offset + int(target_length)].tolist()
                offset += int(target_length)
                errors, count = edit_totals(reference, hypotheses[index])
                greedy_errors += errors
                phones += count
                records.append((clip_ids[index], reference,
                                log_probabilities[index, :int(lengths[index])]))
            if batch_index % 10 == 0:
                LOGGER.info("encoded clips=%d/%d", len(records), len(validation))
    transitions = training_bigram(args.data_root, args.lm_smoothing)
    results = []
    for lm_weight in lm_weights:
        totals = {n: {"errors": 0, "hits": 0} for n in n_values}
        for index, (_, reference, scores) in enumerate(records, 1):
            candidates = ctc_prefix_beam_search(
                scores, args.beam_width, n_values[-1], args.beam_token_top_k,
                transitions if lm_weight else None, lm_weight,
            )
            sequences = [sequence for sequence, _ in candidates]
            for n in n_values:
                subset = sequences[:n]
                totals[n]["hits"] += int(reference in subset)
                totals[n]["errors"] += min(edit_totals(reference, sequence)[0]
                                           for sequence in subset)
            if index % 500 == 0:
                LOGGER.info("decoded lm_weight=%.3f clips=%d/%d", lm_weight, index,
                            len(records))
        curve = [{"n": n, "oracle_per": totals[n]["errors"] / phones,
                  "exact_hits": totals[n]["hits"],
                  "exact_accuracy": totals[n]["hits"] / len(records)} for n in n_values]
        results.append({"lm_weight": lm_weight, "curve": curve})
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_epoch": checkpoint["epoch"],
        "architecture": architecture, "coordinate_mode": coordinate_mode,
        "coordinate_features": coordinate_features, "validation_clips": len(records),
        "validation_phones": phones, "greedy_per": greedy_errors / phones,
        "beam_width": args.beam_width, "beam_token_top_k": args.beam_token_top_k,
        "lm_smoothing": args.lm_smoothing, "results": results,
        "seconds": time.perf_counter() - started,
        "test_speakers_evaluated": False,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    LOGGER.info("report=%s seconds=%.1f", output, report["seconds"])
    for handler in LOGGER.handlers:
        handler.flush()


if __name__ == "__main__":
    main()
