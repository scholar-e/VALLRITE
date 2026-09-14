"""Label media with audibly supervised ARPAbet sequences from large ASR teachers."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import re
import sys
import tempfile
from typing import Iterable

from .alignment import MontrealForcedAligner, normalize_audio
from .core import (add_acoustic_evidence, build_alignment, select_consensus,
                   phone_intervals_to_alignment, use_forced_alignment)
from .phones import Wav2Vec2PhoneTeacher
from .teachers import FasterWhisperTeacher

LOGGER = logging.getLogger("audio_phoneme_labeler")
MEDIA_SUFFIXES = {".aac", ".flac", ".m4a", ".mkv", ".mov", ".mp3", ".mp4",
                  ".mpeg", ".mpg", ".ogg", ".wav", ".webm"}
ARPABET_PHONES = {
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH",
    "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N",
    "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V",
    "W", "Y", "Z", "ZH",
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


def media_files(inputs: Iterable[Path]) -> list[Path]:
    paths = []
    for source in inputs:
        if source.is_dir():
            paths.extend(path for path in source.rglob("*")
                         if path.is_file() and path.suffix.lower() in MEDIA_SUFFIXES)
        elif source.is_file() and source.suffix.lower() in MEDIA_SUFFIXES:
            paths.append(source)
    return sorted(set(path.resolve() for path in paths))


def content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def output_name(path: Path) -> str:
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", path.stem).strip("-") or "media"
    identity = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]
    return f"{safe_stem}-{identity}.json"


def cmudict_pronunciation(word: str) -> list[str]:
    try:
        import pronouncing
    except ImportError as error:
        raise RuntimeError(
            "pronouncing is required; install requirements-labeler.txt"
        ) from error
    pronunciations = pronouncing.phones_for_word(word)
    if not pronunciations:
        return []
    return [phone for phone in pronunciations[0].split()
            if re.sub(r"\d", "", phone) in ARPABET_PHONES]


def atomic_json(path: Path, document: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def atomic_jsonl(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text("".join(json.dumps(record, allow_nan=False) + "\n"
                                 for record in records), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="media files or directories")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("datasets/audio-teacher-labels"))
    parser.add_argument("--teacher", action="append", dest="teachers",
                        help="repeat for an ensemble (default: large-v3-turbo)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--compute-type", default="default",
                        help="CTranslate2 type such as float16, int8_float16, or int8")
    parser.add_argument("--language", default="en")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--min-quality", type=float, default=0.65)
    parser.add_argument("--min-coverage", type=float, default=0.95)
    parser.add_argument("--min-agreement", type=float, default=0.75)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--model-cache-dir", type=Path,
                        help="shared writable teacher cache (default: OUTPUT_DIR/.model-cache)")
    parser.add_argument("--phone-teacher-mode", choices=("required", "preferred", "off"),
                        default="preferred")
    parser.add_argument("--phone-teacher-model",
                        default="facebook/wav2vec2-lv-60-espeak-cv-ft")
    parser.add_argument("--min-phone-agreement", type=float, default=0.60)
    parser.add_argument("--mfa-mode", choices=("required", "preferred", "off"),
                        default="preferred")
    parser.add_argument("--mfa-dictionary", default="english_us_arpa")
    parser.add_argument("--mfa-acoustic-model", default="english_us_arpa")
    parser.add_argument("--ctc-align-mode", choices=("required", "preferred", "off"),
                        default="preferred",
                        help="transcript-constrained Wav2Vec2 phone alignment fallback")
    args = parser.parse_args()
    thresholds = (args.min_quality, args.min_coverage, args.min_agreement,
                  args.min_phone_agreement)
    if (args.beam_size < 1 or (args.limit is not None and args.limit < 1)
            or any(not 0 <= value <= 1 for value in thresholds)):
        parser.error("invalid beam, limit, or quality threshold")
    sources = media_files(args.inputs)
    if args.limit:
        sources = sources[:args.limit]
    if not sources:
        parser.error("no supported media files found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(args.output_dir / "labeler.log")
    model_cache_dir = args.model_cache_dir or args.output_dir / ".model-cache"
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    aligner = MontrealForcedAligner(args.mfa_dictionary, args.mfa_acoustic_model)
    if args.mfa_mode == "required" and not aligner.available:
        raise RuntimeError("--mfa-mode required but the mfa executable is unavailable")
    if args.mfa_mode == "preferred" and not aligner.available:
        LOGGER.warning("MFA unavailable; provisional within-word timings will be retained")
    teacher_names = args.teachers or ["large-v3-turbo"]
    LOGGER.info("loading teachers=%s device=%s compute_type=%s", teacher_names,
                args.device, args.compute_type)
    teachers = [FasterWhisperTeacher(name, args.device, args.compute_type,
                                     args.local_files_only, model_cache_dir)
                for name in teacher_names]
    phone_teacher = None
    if args.phone_teacher_mode != "off" or args.ctc_align_mode != "off":
        try:
            LOGGER.info("loading phone_teacher=%s", args.phone_teacher_model)
            phone_teacher = Wav2Vec2PhoneTeacher(
                args.phone_teacher_model, args.device, args.local_files_only,
                model_cache_dir,
            )
        except Exception:
            if (args.phone_teacher_mode == "required"
                    or args.ctc_align_mode == "required"):
                raise
            LOGGER.warning("direct phone teacher unavailable; continuing in preferred mode",
                           exc_info=True)
    manifest_path = args.output_dir / "labels.jsonl"
    records_by_source = {}
    if manifest_path.is_file():
        records_by_source = {
            record["source"]: record
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and (record := json.loads(line)).get("source")
        }
    accepted = rejected = failures = skipped = 0
    for index, source in enumerate(sources, 1):
        destination = args.output_dir / output_name(source)
        if destination.exists() and not args.overwrite:
            skipped += 1
            records_by_source[str(source)] = {
                "source": str(source), "alignment": str(destination), "status": "cached"
            }
            LOGGER.info("clip=%d/%d status=cached source=%s", index, len(sources), source)
            continue
        try:
            transcripts = [teacher.transcribe(source, args.language, args.beam_size)
                           for teacher in teachers]
            consensus, agreement = select_consensus(transcripts)
            source_metadata = {
                "source": str(source),
                "source_sha256": content_sha256(source),
                "teachers_run": [transcript.teacher for transcript in transcripts],
                "teacher_transcripts": [list(transcript.tokens) for transcript in transcripts],
                "labeler": "AudioPhonemeLabeler/0.1",
                "asr_options": {"beam_size": args.beam_size, "language": args.language,
                                "compute_type": args.compute_type},
                "hybrid_options": {
                    "phone_teacher_mode": args.phone_teacher_mode,
                    "phone_teacher_model": args.phone_teacher_model,
                    "min_phone_agreement": args.min_phone_agreement,
                    "mfa_mode": args.mfa_mode,
                    "mfa_dictionary": args.mfa_dictionary,
                    "mfa_acoustic_model": args.mfa_acoustic_model,
                    "ctc_align_mode": args.ctc_align_mode,
                    "model_cache_dir": str(model_cache_dir),
                },
            }
            document = build_alignment(
                consensus, agreement, cmudict_pronunciation, source_metadata,
                args.min_quality, args.min_coverage, args.min_agreement,
            )
            document["quality"]["direct_phone_status"] = (
                "off" if args.phone_teacher_mode == "off" else "unavailable"
            )
            document["quality"]["forced_alignment"] = (
                "off" if args.mfa_mode == "off" and args.ctc_align_mode == "off"
                else "fallback"
            )
            if args.mfa_mode != "off" and not aligner.available:
                document["quality"]["forced_alignment_error"] = (
                    "Montreal Forced Aligner executable is not installed"
                )
            needs_wav = phone_teacher is not None or (
                args.mfa_mode != "off" and aligner.available
            )
            if needs_wav:
                with tempfile.TemporaryDirectory(prefix="vallr-audio-label-") as temporary:
                    work_dir = Path(temporary)
                    wav_path = work_dir / "audio.wav"
                    normalize_audio(source, wav_path)
                    ctc_alignment_complete = False
                    if phone_teacher is not None:
                        try:
                            targets = [str(entry[2])
                                       for entry in document["tiers"]["phones"]["entries"]]
                            if args.ctc_align_mode != "off":
                                direct, forced_phones = phone_teacher.recognize_and_align(
                                    wav_path, targets
                                )
                                forced = phone_intervals_to_alignment(
                                    document, forced_phones.entries,
                                    forced_phones.path_confidence,
                                )
                                use_forced_alignment(
                                    document, forced,
                                    f"CTC Viterbi {forced_phones.model}",
                                )
                                document["quality"]["forced_alignment_score"] = (
                                    forced_phones.path_confidence
                                )
                                document["quality"]["forced_alignment_frame_seconds"] = (
                                    forced_phones.frame_duration
                                )
                                ctc_alignment_complete = True
                            else:
                                direct = phone_teacher.recognize(wav_path)
                            if args.phone_teacher_mode != "off":
                                add_acoustic_evidence(
                                    document, direct.arpabet, direct.mapping_coverage,
                                    direct.raw, direct.model, args.min_phone_agreement,
                                    args.min_quality,
                                )
                                document["quality"]["direct_phone_status"] = "complete"
                        except Exception as error:
                            if args.phone_teacher_mode != "off":
                                document["quality"]["direct_phone_status"] = "failed"
                                document["quality"]["direct_phone_error"] = str(error)
                            if args.phone_teacher_mode == "required":
                                document["quality"]["rejection_reasons"].append(
                                    "direct_phone_teacher_required"
                                )
                            if args.ctc_align_mode != "off":
                                document["quality"]["ctc_forced_alignment_error"] = str(error)
                            if args.ctc_align_mode == "required":
                                document["quality"]["rejection_reasons"].append(
                                    "ctc_forced_alignment_required"
                                )
                            LOGGER.warning("direct phone teacher failed for %s", source,
                                           exc_info=True)
                    if args.mfa_mode != "off" and aligner.available:
                        try:
                            forced = aligner.align(wav_path, list(consensus.tokens), work_dir)
                            use_forced_alignment(
                                document, forced,
                                f"MFA {args.mfa_acoustic_model}/{args.mfa_dictionary}",
                            )
                        except Exception as error:
                            document["quality"]["forced_alignment"] = "fallback"
                            document["quality"]["forced_alignment_error"] = str(error)
                            if args.mfa_mode == "required":
                                document["quality"]["rejection_reasons"].append(
                                    "forced_alignment_required"
                                )
                            LOGGER.warning("forced alignment unavailable for %s", source,
                                           exc_info=True)
                    if (args.mfa_mode == "off" and args.ctc_align_mode != "off"
                            and not ctc_alignment_complete
                            and args.ctc_align_mode == "required"
                            and "ctc_forced_alignment_required" not in
                            document["quality"]["rejection_reasons"]):
                        document["quality"]["rejection_reasons"].append(
                            "ctc_forced_alignment_required"
                        )
            reasons = document["quality"]["rejection_reasons"]
            document["label_status"] = "accepted" if not reasons else "rejected"
            atomic_json(destination, document)
            status = document["label_status"]
            accepted += status == "accepted"
            rejected += status == "rejected"
            records_by_source[str(source)] = {
                "source": str(source), "alignment": str(destination),
                "status": status, "quality": document["quality"]
            }
            LOGGER.info("clip=%d/%d status=%s quality=%.3f phones=%d source=%s",
                        index, len(sources), status, document["quality"]["score"],
                        len(document["tiers"]["phones"]["entries"]), source)
        except Exception as error:
            failures += 1
            records_by_source[str(source)] = {
                "source": str(source), "alignment": None,
                "status": "failed", "error": str(error)
            }
            LOGGER.error("clip=%d/%d status=failed source=%s", index, len(sources),
                         source, exc_info=True)
        records = [records_by_source[key] for key in sorted(records_by_source)]
        atomic_json(manifest_path.with_suffix(".json"), {"records": records})
        atomic_jsonl(manifest_path, records)
        for handler in LOGGER.handlers:
            handler.flush()
    LOGGER.info("complete total=%d accepted=%d rejected=%d failed=%d cached=%d",
                len(sources), accepted, rejected, failures, skipped)
    for handler in LOGGER.handlers:
        handler.flush()
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
