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

import cv2
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


def read_timed_words(path: Path) -> list[tuple[str, float, float]]:
    """Read the word timing table included with an LRS3 pretrain clip."""
    words = []
    in_table = False
    for line in path.read_text(errors="replace").splitlines():
        if line.strip().upper() == "WORD START END ASDSCORE":
            in_table = True
            continue
        if not in_table or not line.strip():
            continue
        fields = line.split()
        if len(fields) < 3:
            continue
        try:
            words.append((fields[0], float(fields[1]), float(fields[2])))
        except ValueError:
            continue
    return words


def pretrain_chunks(video: Path, root: Path, max_phones: int,
                    max_seconds: float = 6.0, max_gap: float = 0.5) -> list[dict]:
    """Split one long pretrain recording at supplied word boundaries."""
    capture = cv2.VideoCapture(str(video))
    try:
        source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
    finally:
        capture.release()
    if source_frames <= 0:
        raise OSError(f"cannot read frame count: {video}")
    groups = []
    unknown_words = []
    for word, start, end in read_timed_words(video.with_suffix(".txt")):
        normalized = normalize_word(word)
        phones = [strip_stress(phone) for phone in cmudict_pronunciation(normalized)]
        phones = [phone for phone in phones if phone in PHONE_TO_ID]
        if phones:
            groups.append((normalized, start, end, phones))
        else:
            unknown_words.append(normalized)
    packed = []
    current = []
    current_phones = 0
    for group in groups:
        exceeds_phones = current_phones + len(group[3]) > max_phones
        exceeds_span = bool(current and group[2] - current[0][1] > max_seconds)
        crosses_gap = bool(current and group[1] - current[-1][2] > max_gap)
        if current and (exceeds_phones or exceeds_span or crosses_gap):
            packed.append(current)
            current = []
            current_phones = 0
        current.append(group)
        current_phones += len(group[3])
    if current:
        packed.append(current)
    relative = video.relative_to(root)
    relative_id = relative.with_suffix("").as_posix()
    rows = []
    for index, chunk in enumerate(packed):
        frame_start = max(0, int(np.floor((chunk[0][1] - 0.04) * fps)))
        frame_end = min(source_frames, int(np.ceil((chunk[-1][2] + 0.04) * fps)))
        phones = [phone for group in chunk for phone in group[3]]
        rows.append({
            "clip_id": f"lrs3:{relative_id}@{frame_start}:{frame_end}",
            "source_clip_id": f"lrs3:{relative_id}",
            "dataset": "LRS3", "split": "train", "source_type": "video",
            "video": str(relative), "landmarks": None,
            "frame_start": frame_start, "frame_end": frame_end,
            "frames": frame_end - frame_start, "fps": fps,
            "transcript": " ".join(group[0] for group in chunk),
            "phonemes": phones, "pronunciation_coverage": 1.0,
            "unknown_words": unknown_words,
            "label_source": "LRS3 pretrain word timings + CMUdict first pronunciation",
            "pretrain_chunk_index": index,
        })
    return rows


def prepare_pretrain(root: Path, max_phones: int) -> list[dict]:
    source = root / "raw" / "pretrain"
    videos = sorted(source.glob("*/*.mp4"))
    if not videos:
        raise FileNotFoundError(f"missing extracted LRS3 pretrain videos: {source}")
    rows = []
    failures = 0
    for index, video in enumerate(videos, 1):
        try:
            rows.extend(pretrain_chunks(video, root, max_phones))
        except (OSError, ValueError) as error:
            failures += 1
            LOGGER.warning("skipping pretrain clip %s: %s", video, error)
        if index % 5000 == 0:
            LOGGER.info("indexed pretrain videos=%d/%d chunks=%d failures=%d",
                        index, len(videos), len(rows), failures)
    LOGGER.info("indexed pretrain videos=%d chunks=%d failures=%d",
                len(videos), len(rows), failures)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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
    parser.add_argument("--include-pretrain", action="store_true")
    parser.add_argument("--pretrain-max-phones", type=int, default=16)
    args = parser.parse_args()
    if args.pretrain_max_phones < 1:
        parser.error("--pretrain-max-phones must be positive")
    configure_logging(args.root / "prepare.log")
    validation_path = args.root / "manifests" / "lrs3-valid.id"
    validation_ids = set(validation_path.read_text().splitlines())
    if len(validation_ids) != 1200:
        raise ValueError(f"expected 1,200 AV-HuBERT validation IDs: {validation_path}")
    manifests = args.root / "manifests"
    existing = all((manifests / f"{split}.jsonl").is_file()
                   for split in ("train", "validation", "test"))
    if args.include_pretrain and existing:
        train = [row for row in read_jsonl(manifests / "train.jsonl")
                 if "/pretrain/" not in f"/{row['video']}" ]
        validation = read_jsonl(manifests / "validation.jsonl")
        test = read_jsonl(manifests / "test.jsonl")
        LOGGER.info("reusing prepared trainval/validation/test manifests")
    else:
        train, validation = prepare_trainval(args.root, validation_ids)
        test = prepare_test(args.root)
    pretrain = prepare_pretrain(args.root, args.pretrain_max_phones) if args.include_pretrain else []
    train = pretrain + train
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
        "pretrain_chunks": len(pretrain),
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
