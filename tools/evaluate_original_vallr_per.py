"""Measure the released VALLR checkpoint's PER on aligned GRID word segments.

This is an out-of-domain diagnostic. The released checkpoint emits eight CTC
steps, so sentence-length GRID references are segmented at aligned word bounds.
"""
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

import cv2
import numpy as np
import torch
from transformers import VideoMAEConfig, Wav2Vec2Config


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "VALLR"))

from Models.VALLR import VALLR  # noqa: E402


LOGGER = logging.getLogger("original_vallr_per")
PHONEMES = tuple(
    "AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG "
    "OW OY P R S SH T TH UH UW V W Y Z ZH".split()
)
PHONE_TO_ID = {phone: index + 1 for index, phone in enumerate(PHONEMES)}
ID_TO_PHONE = {value: key for key, value in PHONE_TO_ID.items()}
CROPS = {
    "face": (0.18, 0.02, 0.82, 0.94),
    "mouth": (0.27, 0.43, 0.73, 0.81),
    "full": (0.0, 0.0, 1.0, 1.0),
}


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


def normalize_phone(raw: str) -> str | None:
    phone = re.sub(r"\d", "", raw).upper()
    return phone if phone in PHONE_TO_ID else None


def edit_counts(reference: list[int], hypothesis: list[int]) -> tuple[int, int, int]:
    """Return substitution, deletion, and insertion counts."""
    rows: list[list[tuple[int, int, int]]] = [
        [(0, 0, column) for column in range(len(hypothesis) + 1)]
    ]
    for row, ref in enumerate(reference, 1):
        current = [(0, row, 0)]
        for column, hyp in enumerate(hypothesis, 1):
            if ref == hyp:
                current.append(rows[row - 1][column - 1])
                continue
            substitution = rows[row - 1][column - 1]
            deletion = rows[row - 1][column]
            insertion = current[column - 1]
            current.append(min(
                (substitution[0] + 1, substitution[1], substitution[2]),
                (deletion[0], deletion[1] + 1, deletion[2]),
                (insertion[0], insertion[1], insertion[2] + 1),
                key=lambda counts: (sum(counts), counts[1], counts[2], counts[0]),
            ))
        rows.append(current)
    return rows[-1][-1]


def ctc_collapse(indices: list[int]) -> list[int]:
    result: list[int] = []
    previous: int | None = None
    for token in indices:
        if token != 0 and token != previous:
            result.append(token)
        previous = token
    return result


def load_model(checkpoint: Path, device: torch.device) -> VALLR:
    video_config = VideoMAEConfig()
    audio_config = Wav2Vec2Config(vocab_size=len(PHONEMES) + 1)
    model = VALLR(video_config, audio_config, adapter_dim=256)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def aligned_instances(data_root: Path, split: str) -> list[dict]:
    rows = [json.loads(line) for line in (data_root / "clips.jsonl").read_text().splitlines()]
    instances = []
    for row in rows:
        if row["split"] != split:
            continue
        stem = Path(row["video"]).stem
        alignment_path = (
            data_root / "landmark-experiment" / "aligned"
            / f"s{row['speaker_id']}" / f"{stem}.json"
        )
        alignment = json.loads(alignment_path.read_text())
        phones = alignment["tiers"]["phones"]["entries"]
        for word_index, (start, end, word) in enumerate(alignment["tiers"]["words"]["entries"]):
            reference = []
            for phone_start, phone_end, raw_phone in phones:
                midpoint = (phone_start + phone_end) / 2
                phone = normalize_phone(raw_phone)
                if start <= midpoint <= end and phone is not None:
                    reference.append(PHONE_TO_ID[phone])
            if reference:
                instances.append({
                    "clip_id": row["clip_id"], "video": row["video"],
                    "word_index": word_index, "word": word,
                    "start": start, "end": end, "reference": reference,
                })
    return instances


def load_segment(path: Path, start: float, end: float, crop_name: str,
                 normalization: str, context_seconds: float,
                 frames: int = 16) -> torch.Tensor:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"cannot open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    first = max(0, min(total - 1, round((start - context_seconds) * fps)))
    last = max(first, min(total - 1, round((end + context_seconds) * fps) - 1))
    indices = np.linspace(first, last, frames).round().astype(int)
    result = []
    try:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                raise OSError(f"cannot decode frame {index}: {path}")
            height, width = frame.shape[:2]
            x0, y0, x1, y1 = CROPS[crop_name]
            crop = frame[
                round(y0 * height):round(y1 * height),
                round(x0 * width):round(x1 * width),
            ]
            crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
            result.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    array = np.stack(result).transpose(0, 3, 1, 2)
    tensor = torch.from_numpy(np.ascontiguousarray(array)).float().unsqueeze(0)
    if normalization == "zero-one":
        tensor.div_(255)
    elif normalization == "minus-one-one":
        tensor.div_(127.5).sub_(1)
    return tensor


def phone_names(sequence: list[int]) -> list[str]:
    return [ID_TO_PHONE[token] for token in sequence]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "datasets/grid-pilot")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "evaluation/original-vallr-grid-word-per.json")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--max-words", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--crop", choices=tuple(CROPS), default="face")
    parser.add_argument("--normalization", choices=("raw", "zero-one", "minus-one-one"),
                        default="raw")
    parser.add_argument("--context-seconds", type=float, default=0.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if not args.checkpoint.is_file() or args.max_words < 1 or args.context_seconds < 0:
        parser.error("checkpoint/max-words/context-seconds are invalid")
    configure_logging(args.output.with_suffix(".log"))
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device("cpu" if device_name == "auto" else device_name)
    LOGGER.info("loading checkpoint=%s device=%s", args.checkpoint, device)
    model = load_model(args.checkpoint, device)
    instances = aligned_instances(args.data_root, args.split)
    random.Random(args.seed).shuffle(instances)
    instances = instances[:args.max_words]
    substitutions = deletions = insertions = reference_phones = exact = 0
    records = []
    started = time.perf_counter()
    with torch.inference_mode():
        for index, instance in enumerate(instances, 1):
            video = load_segment(
                args.data_root / instance["video"], instance["start"], instance["end"],
                args.crop, args.normalization, args.context_seconds,
            ).to(device)
            logits, _ = model(video)
            hypothesis = ctc_collapse(logits.argmax(dim=-1)[0].tolist())
            reference = instance["reference"]
            sub, delete, insert = edit_counts(reference, hypothesis)
            substitutions += sub
            deletions += delete
            insertions += insert
            reference_phones += len(reference)
            exact += int(reference == hypothesis)
            records.append({
                "clip_id": instance["clip_id"], "word_index": instance["word_index"],
                "word": instance["word"], "reference": phone_names(reference),
                "hypothesis": phone_names(hypothesis), "substitutions": sub,
                "deletions": delete, "insertions": insert,
            })
            if index % 10 == 0 or index == len(instances):
                LOGGER.info("evaluated words=%d/%d", index, len(instances))
                for handler in LOGGER.handlers:
                    handler.flush()
    errors = substitutions + deletions + insertions
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Out-of-domain held-out GRID word segments; not the in-domain LRS3 test PER",
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": None,
        "dataset": "GRID", "split": args.split, "seed": args.seed,
        "sampling": "deterministic shuffled aligned words",
        "preprocessing": {
            "frames": 16, "color": "RGB", "crop": args.crop, "size": 224,
            "normalization": args.normalization, "context_seconds_each_side": args.context_seconds,
        },
        "decode": "greedy argmax with standard CTC repeat collapse and blank removal",
        "words": len(instances), "reference_phones": reference_phones,
        "substitutions": substitutions, "deletions": deletions, "insertions": insertions,
        "errors": errors, "per": errors / reference_phones,
        "exact_words": exact, "exact_word_accuracy": exact / len(instances),
        "seconds": time.perf_counter() - started, "records": records,
    }
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    report["checkpoint_sha256"] = digest
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    LOGGER.info(
        "PER=%.2f%% errors=%d phones=%d exact_words=%d/%d seconds=%.1f report=%s",
        report["per"] * 100, errors, reference_phones, exact, len(instances),
        report["seconds"], args.output,
    )
    for handler in LOGGER.handlers:
        handler.flush()


if __name__ == "__main__":
    main()
