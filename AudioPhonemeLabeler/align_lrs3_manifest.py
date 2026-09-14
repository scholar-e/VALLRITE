"""Force-align official LRS3 transcripts without running an ASR transcript teacher."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import re
import sys
import tempfile

from .__main__ import (atomic_json, atomic_jsonl, cmudict_pronunciation,
                       content_sha256, output_name)
from .alignment import normalize_audio
from .core import (TeacherTranscript, Word, build_alignment,
                   phone_intervals_to_alignment, use_forced_alignment)
from .phones import Wav2Vec2PhoneTeacher

LOGGER = logging.getLogger("audio_phoneme_labeler.align_lrs3_manifest")


def configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.handlers.clear()
    for handler in (logging.FileHandler(path, delay=False), logging.StreamHandler(sys.stdout)):
        handler.setLevel(logging.DEBUG if isinstance(handler, logging.FileHandler) else logging.INFO)
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def transcript_words(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)*", text.lower())


def provisional_document(row: dict, source: Path) -> dict:
    words = transcript_words(row["transcript"])
    weighted = [(word, cmudict_pronunciation(word)) for word in words]
    total_phones = sum(len(phones) for _, phones in weighted)
    duration = float(row["frames"]) / float(row["fps"])
    offset = 0
    entries = []
    for word, phones in weighted:
        start = duration * offset / max(total_phones, 1)
        offset += max(len(phones), 1)
        end = duration * offset / max(total_phones, 1)
        entries.append(Word(word, start, end, 1.0))
    teacher = TeacherTranscript(
        teacher="official-lrs3-transcript", language="en", language_probability=1.0,
        words=tuple(entries), duration=duration,
    )
    document = build_alignment(
        teacher, 1.0, cmudict_pronunciation,
        {"source": str(source.resolve()), "source_sha256": content_sha256(source),
         "teacher_transcripts": [words], "labeler": "LRS3 manifest CTC aligner/0.1"},
    )
    expected = [re.sub(r"\d", "", str(phone)).upper() for phone in row["phonemes"]]
    actual = [re.sub(r"\d", "", str(entry[2])).upper()
              for entry in document["tiers"]["phones"]["entries"]]
    if actual != expected:
        raise ValueError("manifest and reconstructed CMUdict phone sequences differ")
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-labels", type=Path,
                        help="reuse accepted/rejected records from an earlier labels.jsonl")
    parser.add_argument("--phone-teacher-model",
                        default="facebook/wav2vec2-lv-60-espeak-cv-ft")
    parser.add_argument("--model-cache-dir", type=Path,
                        default=Path("datasets/audio-teacher-smoke/.model-cache"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--flush-every", type=int, default=25)
    args = parser.parse_args()
    if (not args.manifest.is_file() or not args.data_root.is_dir()
            or (args.seed_labels and not args.seed_labels.is_file())
            or (args.limit is not None and args.limit < 1) or args.flush_every < 1):
        parser.error("invalid manifest, root, seed labels, limit, or flush interval")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(args.output_dir / "align.log")

    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    records_by_source = {}
    if args.seed_labels:
        for line in args.seed_labels.read_text().splitlines():
            record = json.loads(line)
            records_by_source[str(Path(record["source"]).resolve())] = record
    output_manifest = args.output_dir / "labels.jsonl"
    if output_manifest.is_file():
        for line in output_manifest.read_text().splitlines():
            record = json.loads(line)
            records_by_source[str(Path(record["source"]).resolve())] = record
    pending = []
    for row in rows:
        source = (args.data_root / row["video"]).resolve()
        if str(source) not in records_by_source:
            pending.append((row, source))
    if args.limit:
        pending = pending[:args.limit]
    LOGGER.info("manifest=%d reused=%d pending=%d device=%s",
                len(rows), len(records_by_source), len(pending), args.device)
    teacher = Wav2Vec2PhoneTeacher(
        args.phone_teacher_model, args.device, args.local_files_only,
        args.model_cache_dir,
    )
    accepted = rejected = failed = 0
    with tempfile.TemporaryDirectory(prefix="lrs3-ctc-align-") as temporary:
        work = Path(temporary)
        for index, (row, source) in enumerate(pending, 1):
            destination = args.output_dir / output_name(source)
            try:
                document = provisional_document(row, source)
                if document["label_status"] == "rejected":
                    status = "rejected"
                    rejected += 1
                else:
                    wav = work / "audio.wav"
                    normalize_audio(source, wav)
                    _, forced = teacher.recognize_and_align(wav, list(row["phonemes"]))
                    alignment = phone_intervals_to_alignment(
                        document, forced.entries, forced.path_confidence)
                    use_forced_alignment(document, alignment,
                                         f"CTC Viterbi {forced.model}")
                    document["quality"]["forced_alignment_score"] = forced.path_confidence
                    document["quality"]["forced_alignment_frame_seconds"] = forced.frame_duration
                    status = "accepted"
                    accepted += 1
                atomic_json(destination, document)
                records_by_source[str(source)] = {
                    "source": str(source), "alignment": str(destination), "status": status,
                }
            except Exception as error:
                failed += 1
                records_by_source[str(source)] = {
                    "source": str(source), "alignment": None, "status": "failed",
                    "error": str(error),
                }
                LOGGER.error("failed clip=%s error=%s", row["clip_id"], error, exc_info=True)
            if index % args.flush_every == 0 or index == len(pending):
                atomic_jsonl(output_manifest, sorted(records_by_source.values(),
                                                     key=lambda value: value["source"]))
                LOGGER.info("processed=%d/%d accepted=%d rejected=%d failed=%d",
                            index, len(pending), accepted, rejected, failed)
                for handler in LOGGER.handlers:
                    handler.flush()


if __name__ == "__main__":
    main()
