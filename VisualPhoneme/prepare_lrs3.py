"""Create VALLRITE manifests and unpacked test arrays from downloaded LRS3."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
import logging
from pathlib import Path
import re
import sys

import numpy as np
import pyarrow.parquet as pq
import pronouncing

from AudioPhonemeLabeler.core import normalize_word, strip_stress
from VisualPhoneme.data import PHONE_TO_ID
from VisualPhoneme.lrs3_data import load_lrs3_landmarks


LOGGER = logging.getLogger("visual_phoneme.prepare_lrs3")


def cmudict_pronunciation(word: str) -> list[str]:
    pronunciations = pronouncing.phones_for_word(word)
    return pronunciations[0].split() if pronunciations else []


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


def transcript_phones(transcript: str) -> tuple[list[str], list[str], float]:
    transcript = html.unescape(transcript).replace("’", "'").replace("‘", "'")
    words = [normalize_word(value) for value in re.findall(r"[A-Za-z']+", transcript)]
    phones = []
    unknown = []
    for word in words:
        pronunciation = [strip_stress(phone) for phone in cmudict_pronunciation(word)]
        pronunciation = [phone for phone in pronunciation if phone in PHONE_TO_ID]
        if pronunciation:
            phones.extend(pronunciation)
        else:
            unknown.append(word)
    return phones, unknown, (len(words) - len(unknown)) / max(len(words), 1)


def read_transcript(path: Path) -> str:
    for line in path.read_text(errors="replace").splitlines():
        if line.lower().startswith("text:"):
            return line.split(":", 1)[1].strip().lower()
    raise ValueError(f"missing Text field: {path}")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))


def prepare_trainval(root: Path, validation_ids: set[str]) -> tuple[list[dict], list[dict]]:
    source = root / "raw" / "trainval"
    train = []
    validation = []
    videos = sorted(source.glob("*/*.mp4"))
    for index, video in enumerate(videos, 1):
        relative_id = f"trainval/{video.parent.name}/{video.stem}"
        transcript = read_transcript(video.with_suffix(".txt"))
        phonemes, unknown, coverage = transcript_phones(transcript)
        landmark = root / "LRS3_landmarks" / "trainval" / video.parent.name / f"{video.stem}.pkl"
        if not landmark.is_file():
            raise FileNotFoundError(f"missing trainval landmarks: {landmark}")
        frame_count = len(load_lrs3_landmarks(landmark))
        row = {
            "clip_id": f"lrs3:{relative_id}", "dataset": "LRS3",
            "split": "validation" if relative_id in validation_ids else "train",
            "source_type": "video", "video": str(video.relative_to(root)),
            "landmarks": str(landmark.relative_to(root)), "frames": frame_count,
            "fps": 25, "transcript": transcript, "phonemes": phonemes,
            "pronunciation_coverage": coverage, "unknown_words": unknown,
            "label_source": "LRS3 transcript + CMUdict first pronunciation",
        }
        (validation if relative_id in validation_ids else train).append(row)
        if index % 5000 == 0:
            LOGGER.info("indexed trainval clips=%d/%d", index, len(videos))
    return train, validation


def prepare_test(root: Path) -> list[dict]:
    output = root / "test-video"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    shards = sorted((root / "downloads").glob("train-*.parquet"))
    for shard in shards:
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(batch_size=8, columns=["idx", "video", "label"]):
            for record in batch.to_pylist():
                item_id = int(record["idx"])
                video = np.asarray(record["video"], dtype=np.uint8)
                video_path = output / f"{item_id:05d}.npy"
                with video_path.open("wb") as destination:
                    np.save(destination, video, allow_pickle=False)
                transcript = str(record["label"]).strip().lower()
                phonemes, unknown, coverage = transcript_phones(transcript)
                rows.append({
                    "clip_id": f"lrs3:test/{item_id:05d}", "dataset": "LRS3",
                    "split": "test", "source_type": "npy-mouth",
                    "video": str(video_path.relative_to(root)), "landmarks": None,
                    "frames": len(video), "fps": 25, "transcript": transcript,
                    "phonemes": phonemes, "pronunciation_coverage": coverage,
                    "unknown_words": unknown,
                    "label_source": "LRS3 transcript + CMUdict first pronunciation",
                    "source_parquet": shard.name,
                })
        LOGGER.info("converted test shard=%s accumulated_clips=%d", shard.name, len(rows))
    return sorted(rows, key=lambda row: row["clip_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("datasets/lrs3"))
    args = parser.parse_args()
    configure_logging(args.root / "prepare.log")
    validation_path = args.root / "manifests" / "lrs3-valid.id"
    validation_ids = set(validation_path.read_text().splitlines())
    if len(validation_ids) != 1200:
        raise ValueError(f"expected 1,200 AV-HuBERT validation IDs: {validation_path}")
    train, validation = prepare_trainval(args.root, validation_ids)
    test = prepare_test(args.root)
    manifests = args.root / "manifests"
    write_jsonl(manifests / "train.jsonl", train)
    write_jsonl(manifests / "validation.jsonl", validation)
    write_jsonl(manifests / "test.jsonl", test)
    all_rows = train + validation + test
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "splits": {"train": len(train), "validation": len(validation), "test": len(test)},
        "frames": {name: sum(row["frames"] for row in rows)
                   for name, rows in (("train", train), ("validation", validation), ("test", test))},
        "phonemes": {name: sum(len(row["phonemes"]) for row in rows)
                     for name, rows in (("train", train), ("validation", validation), ("test", test))},
        "mean_pronunciation_coverage": (
            sum(row["pronunciation_coverage"] for row in all_rows) / len(all_rows)
        ),
        "minimum_training_pronunciation_coverage": 0.95,
        "quality_gated_splits": {
            name: sum(row["pronunciation_coverage"] >= 0.95 for row in rows)
            for name, rows in (("train", train), ("validation", validation), ("test", test))
        },
        "rows_with_unknown_words": sum(bool(row["unknown_words"]) for row in all_rows),
        "validation_protocol": "official AV-HuBERT 1,200-ID validation list",
        "test_landmarks_joined": False,
        "test_landmark_note": (
            "The supplied test parquet retains only numeric idx; no verified mapping to "
            "the separately supplied landmark path IDs is available."
        ),
    }
    (manifests / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    LOGGER.info("prepared splits train=%d validation=%d test=%d coverage=%.5f",
                len(train), len(validation), len(test),
                summary["mean_pronunciation_coverage"])
    for handler in LOGGER.handlers:
        handler.flush()


if __name__ == "__main__":
    main()
