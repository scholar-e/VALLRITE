"""Generate realistic GRID decoder examples and measure word error rate."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import sys
import time

import torch
from torch.utils.data import DataLoader

from PhonemeDecoder.decoder import Lexicon, PHONES, StreamingDecoder, probabilities_from_logits
from VisualPhoneme.data import (GridClips, collate_clips, collate_fusion_clips,
                                collate_landmark_clips, greedy_decode)
from VisualPhoneme.model import (CompactFusionVisualPhoneme,
                                 CompactGatedFusionVisualPhoneme,
                                 CompactLandmarkPhoneme,
                                 CompactTongueGatedFusionVisualPhoneme,
                                 CompactVisualPhoneme)

LOGGER = logging.getLogger("phoneme_decoder.evaluate_grid")


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.handlers.clear()
    file_handler = logging.FileHandler(log_path, delay=False)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(console_handler)


def build_lexicon(path: Path) -> dict:
    """Convert the stress-marked GRID dictionary into the runtime JSON contract."""
    pronunciations: dict[str, set[tuple[str, ...]]] = {}
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        fields = raw.split()
        if len(fields) < 2:
            if raw.strip():
                raise ValueError(f"invalid dictionary row {line_number}")
            continue
        word = fields[0].lower()
        phones = tuple(re.sub(r"\d", "", phone).upper() for phone in fields[1:])
        if any(phone not in PHONES for phone in phones):
            raise ValueError(f"unknown phone on dictionary row {line_number}")
        pronunciations.setdefault(word, set()).add(phones)
    words = []
    for word, variants_set in sorted(pronunciations.items()):
        variants = sorted(variants_set)
        prior = 1.0 / len(variants)
        words.append({
            "id": word,
            "text": word.removeprefix("letter") if word.startswith("letter") else word,
            "pronunciations": [
                {"id": f"{word}-{index}", "phones": list(phones), "prior": prior}
                for index, phones in enumerate(variants, 1)
            ],
        })
    document = {"phone_inventory": list(PHONES), "words": words}
    Lexicon(document)
    return document


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, 1):
        current = [row]
        for column, actual in enumerate(hypothesis, 1):
            current.append(min(
                previous[column] + 1,
                current[column - 1] + 1,
                previous[column - 1] + (expected != actual),
            ))
        previous = current
    return previous[-1]


def reference_for(dataset: GridClips, clip_id: str) -> tuple[list[str], list[str]]:
    speaker, stem = clip_id.split(":", 1)
    path = dataset.root / "landmark-experiment" / "aligned" / speaker / f"{stem}.json"
    record = json.loads(path.read_text())
    words = [entry[2].lower() for entry in record["tiers"]["words"]["entries"]]
    phones = [re.sub(r"\d", "", entry[2]).upper()
              for entry in record["tiers"]["phones"]["entries"]]
    return words, phones


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if tuple(checkpoint["phones"]) != PHONES:
        raise ValueError("checkpoint and decoder phone inventories differ")
    architecture = checkpoint.get("architecture", "image")
    classes = {
        "image": CompactVisualPhoneme,
        "coordinates": CompactLandmarkPhoneme,
        "fusion": CompactFusionVisualPhoneme,
        "gated-fusion": CompactGatedFusionVisualPhoneme,
        "tongue-gated-fusion": CompactTongueGatedFusionVisualPhoneme,
    }
    arguments = {"classes": len(PHONES) + 1}
    if architecture in {"coordinates", "fusion", "gated-fusion", "tongue-gated-fusion"}:
        arguments.update(
            landmark_points=int(checkpoint["landmark_points"]),
            coordinate_dimensions=int(checkpoint.get("coordinate_dimensions", 2)),
        )
    if architecture in {"coordinates", "gated-fusion", "tongue-gated-fusion"}:
        arguments["landmark_bottleneck"] = checkpoint.get("landmark_bottleneck")
    if architecture in {"gated-fusion", "tongue-gated-fusion"}:
        arguments["image_gate_probability"] = checkpoint.get(
            "image_gate_initial_probability", 0.002472623
        )
    if architecture == "tongue-gated-fusion":
        arguments["inner_mouth_gate_probability"] = checkpoint.get(
            "inner_mouth_gate_initial_probability", 0.05
        )
    model = classes[architecture](**arguments).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return checkpoint, architecture, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/grid-pilot"))
    parser.add_argument("--dictionary", type=Path,
                        default=Path("datasets/grid-pilot/landmark-experiment/grid.dict"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--lexical-beam", type=int, default=64)
    parser.add_argument("--nbest", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--frame-cache-dir", type=Path,
                        default=Path("datasets/grid-pilot/frame-cache"))
    args = parser.parse_args()
    if (not args.checkpoint.is_file() or not args.dictionary.is_file()
            or args.batch_size < 1 or args.workers < 0 or args.beam_width < 1
            or args.lexical_beam < 1 or args.nbest < 1
            or (args.limit is not None and args.limit < 1)):
        parser.error("invalid path, limit, batch, worker, beam, or n-best setting")

    configure_logging(args.output_dir / "generate.log")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device("cpu" if device_name == "auto" else device_name)
    checkpoint, architecture, model = load_model(args.checkpoint, device)
    include_video = architecture in {"image", "fusion", "gated-fusion",
                                     "tongue-gated-fusion"}
    include_landmarks = architecture in {"coordinates", "fusion", "gated-fusion",
                                         "tongue-gated-fusion"}
    dataset = GridClips(
        args.data_root, args.split, int(checkpoint["image_size"]), checkpoint["crop"],
        args.limit, False, include_landmarks, include_video, False,
        checkpoint.get("coordinate_mode", "eye-normalized"), args.frame_cache_dir,
        checkpoint.get("coordinate_features", "position"),
    )
    collators = {"image": collate_clips, "coordinates": collate_landmark_clips,
                 "fusion": collate_fusion_clips, "gated-fusion": collate_fusion_clips,
                 "tongue-gated-fusion": collate_fusion_clips}
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        collate_fn=collators[architecture], pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    lexicon_document = build_lexicon(args.dictionary)
    lexicon = Lexicon(lexicon_document)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "lexicon.json").write_text(
        json.dumps(lexicon_document, indent=2, allow_nan=False) + "\n"
    )
    output_path = args.output_dir / "examples.jsonl"
    total_words = top_errors = oracle_errors = exact_hits = 0
    status_counts: Counter[str] = Counter()
    started = time.perf_counter()
    processed = 0
    LOGGER.info("device=%s architecture=%s split=%s clips=%d beam=%d nbest=%d",
                device, architecture, args.split, len(dataset), args.beam_width, args.nbest)
    with output_path.open("w", buffering=1) as output, torch.inference_mode():
        for batch_number, batch in enumerate(loader, 1):
            if architecture in {"fusion", "gated-fusion", "tongue-gated-fusion"}:
                video, landmarks, mask, _, lengths, _, clip_ids = batch
                logits = model(video.to(device, non_blocking=True),
                               landmarks.to(device, non_blocking=True),
                               mask.to(device, non_blocking=True))
            elif architecture == "coordinates":
                landmarks, mask, _, lengths, _, clip_ids = batch
                logits = model(landmarks.to(device, non_blocking=True),
                               mask.to(device, non_blocking=True))
            else:
                video, _, lengths, _, clip_ids = batch
                logits = model(video.to(device, non_blocking=True))
            greedy = greedy_decode(logits, lengths)
            for index, (clip_id, length) in enumerate(zip(clip_ids, lengths, strict=True)):
                decoder = StreamingDecoder(lexicon, beam_width=args.beam_width,
                                           lexical_beam=args.lexical_beam)
                decoder.accept(probabilities_from_logits(
                    logits, checkpoint["phones"], int(length), index
                ))
                result = decoder.result(args.nbest)
                reference_words, reference_phones = reference_for(dataset, clip_id)
                candidates = result["candidates"]
                errors = [edit_distance(reference_words, candidate["words"])
                          for candidate in candidates]
                top_error = errors[0] if errors else len(reference_words)
                oracle_error = min(errors, default=len(reference_words))
                exact = oracle_error == 0
                record = {
                    "format": "grid-decoder-training-example-0.1",
                    "clip_id": clip_id,
                    "speaker_id": int(clip_id.split(":", 1)[0].removeprefix("s")),
                    "split": args.split,
                    "reference_words": reference_words,
                    "reference_phones": reference_phones,
                    "greedy_visual_phones": [PHONES[token - 1] for token in greedy[index]],
                    "decoder_status": result["status"],
                    "candidates": candidates,
                    "top_1_word_errors": top_error,
                    "oracle_word_errors_at_n": oracle_error,
                    "reference_in_nbest": exact,
                }
                output.write(json.dumps(record, allow_nan=False) + "\n")
                total_words += len(reference_words)
                top_errors += top_error
                oracle_errors += oracle_error
                exact_hits += exact
                status_counts[result["status"]] += 1
                processed += 1
            output.flush()
            if batch_number == 1 or batch_number % 20 == 0:
                LOGGER.info("processed=%d top1_WER=%.2f%% oracle_WER@%d=%.2f%% exact@%d=%.2f%%",
                            processed, 100 * top_errors / total_words, args.nbest,
                            100 * oracle_errors / total_words, args.nbest,
                            100 * exact_hits / processed)

    elapsed = time.perf_counter() - started
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "format": "grid-decoder-training-summary-0.1",
        "scope": "in-sample diagnostic" if args.split == "train" else "development validation",
        "warning": ("The visual producer was fit on these speakers; do not report this as "
                    "held-out performance." if args.split == "train" else None),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint["epoch"],
        "architecture": architecture,
        "split": args.split,
        "clips": processed,
        "reference_words": total_words,
        "decoder_statuses": dict(status_counts),
        "top_1_word_errors": top_errors,
        "top_1_wer": top_errors / total_words,
        "oracle_word_errors_at_n": oracle_errors,
        "oracle_wer_at_n": oracle_errors / total_words,
        "nbest": args.nbest,
        "reference_in_nbest": exact_hits,
        "reference_in_nbest_rate": exact_hits / processed,
        "elapsed_seconds": elapsed,
        "clips_per_second": processed / elapsed,
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    LOGGER.info("complete clips=%d top1_WER=%.2f%% oracle_WER@%d=%.2f%% exact@%d=%.2f%% seconds=%.1f",
                processed, 100 * summary["top_1_wer"], args.nbest,
                100 * summary["oracle_wer_at_n"], args.nbest,
                100 * summary["reference_in_nbest_rate"], elapsed)
    for handler in LOGGER.handlers:
        handler.flush()


if __name__ == "__main__":
    main()
