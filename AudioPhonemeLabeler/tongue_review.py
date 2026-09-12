"""Create a manual tongue-visibility review queue from audio-phone intervals."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import re
import sys

LOGGER = logging.getLogger("audio_phoneme_labeler.tongue_review")
TONGUE_RELEVANT_PHONES = frozenset({"TH", "DH", "L"})


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


def review_records(document: dict, alignment_path: Path) -> list[dict]:
    phones = document["tiers"]["phones"]["entries"]
    records = []
    for index, (start, end, raw_phone) in enumerate(phones):
        phone = re.sub(r"\d", "", raw_phone).upper()
        if phone not in TONGUE_RELEVANT_PHONES:
            continue
        records.append({
            "format": "tongue-visibility-review-0.1",
            "source": document["provenance"]["source"],
            "alignment": str(alignment_path.resolve()),
            "source_label_status": document["label_status"],
            "phone": phone,
            "start_seconds": float(start),
            "end_seconds": float(end),
            "left_phone": (re.sub(r"\d", "", phones[index - 1][2]).upper()
                           if index else None),
            "right_phone": (re.sub(r"\d", "", phones[index + 1][2]).upper()
                            if index + 1 < len(phones) else None),
            "review_status": "pending",
            "tongue_visibility": None,
            "tongue_keypoints": None,
            "admit_to_training": False,
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accepted-only", action="store_true")
    args = parser.parse_args()
    if not args.label_dir.is_dir():
        parser.error("label_dir must exist")
    configure_logging(args.output.with_suffix(".log"))
    records = []
    alignments = 0
    for path in sorted(args.label_dir.glob("*.json")):
        if path.name in {"labels.json", "summary.json"}:
            continue
        document = json.loads(path.read_text())
        if "tiers" not in document or "provenance" not in document:
            continue
        alignments += 1
        if args.accepted_only and document.get("label_status") != "accepted":
            continue
        records.extend(review_records(document, path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(record, allow_nan=False) + "\n"
                                   for record in records))
    LOGGER.info("alignments=%d candidates=%d accepted_only=%s output=%s",
                alignments, len(records), args.accepted_only, args.output)
    for handler in LOGGER.handlers:
        handler.flush()


if __name__ == "__main__":
    main()
