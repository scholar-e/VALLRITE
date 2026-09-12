"""Pure data conversion and quality scoring for audio-teacher labels."""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Callable, Sequence


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float
    probability: float


@dataclass(frozen=True)
class TeacherTranscript:
    teacher: str
    language: str
    language_probability: float
    words: tuple[Word, ...]
    duration: float | None = None

    @property
    def tokens(self) -> tuple[str, ...]:
        return tuple(normalize_word(word.text) for word in self.words
                     if normalize_word(word.text))


def normalize_word(value: str) -> str:
    """Normalize an English ASR word while preserving internal apostrophes."""
    return re.sub(r"(^'+|'+$)", "", re.sub(r"[^a-z']", "", value.lower()))


def edit_distance(left: Sequence[str], right: Sequence[str]) -> int:
    previous = list(range(len(right) + 1))
    for row, left_value in enumerate(left, 1):
        current = [row]
        for column, right_value in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (left_value != right_value)))
        previous = current
    return previous[-1]


def normalized_distance(left: Sequence[str], right: Sequence[str]) -> float:
    return edit_distance(left, right) / max(len(left), len(right), 1)


def strip_stress(phone: str) -> str:
    return re.sub(r"\d", "", phone)


def add_acoustic_evidence(document: dict, direct_phones: Sequence[str],
                          mapping_coverage: float, raw_phones: Sequence[str],
                          model: str, min_agreement: float = 0.60,
                          min_quality: float = 0.65) -> None:
    """Score a transcript-derived target against an independent phone recognizer."""
    reference = [strip_stress(str(entry[2]))
                 for entry in document["tiers"]["phones"]["entries"]]
    acoustic_agreement = max(
        0.0, 1.0 - normalized_distance(reference, direct_phones)
    ) * mapping_coverage
    quality = document["quality"]
    quality["acoustic_phone_agreement"] = acoustic_agreement
    quality["direct_phone_mapping_coverage"] = mapping_coverage
    quality["direct_arpabet_phones"] = list(direct_phones)
    quality["direct_raw_phones"] = list(raw_phones)
    quality["direct_phone_teacher"] = model
    quality["score"] = math.sqrt(quality["score"] * acoustic_agreement)
    if acoustic_agreement < min_agreement:
        quality["rejection_reasons"].append("low_acoustic_phone_agreement")
    if quality["score"] < min_quality and "low_quality_score" not in quality["rejection_reasons"]:
        quality["rejection_reasons"].append("low_quality_score")
    document["label_status"] = (
        "accepted" if not quality["rejection_reasons"] else "rejected"
    )


def use_forced_alignment(document: dict, forced: dict, aligner: str) -> None:
    """Replace provisional intervals with validated forced-alignment tiers."""
    tiers = forced.get("tiers", {})
    words = tiers.get("words", {}).get("entries")
    phones = tiers.get("phones", {}).get("entries")
    if not isinstance(words, list) or not isinstance(phones, list) or not phones:
        raise ValueError("forced alignment has no word/phone interval tiers")
    document["start"] = float(forced.get("start", document["start"]))
    document["end"] = float(forced.get("end", document["end"]))
    document["tiers"] = {
        "words": {"type": "interval", "entries": words},
        "phones": {"type": "interval", "entries": phones},
    }
    document["provenance"]["phone_timing"] = aligner
    document["quality"]["forced_alignment"] = "complete"


def select_consensus(transcripts: Sequence[TeacherTranscript]) -> tuple[TeacherTranscript, float]:
    """Return the transcript medoid and mean pairwise agreement with it."""
    if not transcripts:
        raise ValueError("at least one teacher transcript is required")
    if len(transcripts) == 1:
        return transcripts[0], 1.0
    distances = []
    for candidate in transcripts:
        distances.append(sum(normalized_distance(candidate.tokens, other.tokens)
                             for other in transcripts if other is not candidate)
                         / (len(transcripts) - 1))
    best_index = min(range(len(transcripts)), key=lambda index: (
        distances[index], -mean_word_confidence(transcripts[index]),
        transcripts[index].teacher,
    ))
    return transcripts[best_index], max(0.0, 1.0 - distances[best_index])


def mean_word_confidence(transcript: TeacherTranscript) -> float:
    if not transcript.words:
        return 0.0
    return sum(min(1.0, max(0., word.probability))
               for word in transcript.words) / len(transcript.words)


def build_alignment(transcript: TeacherTranscript, agreement: float,
                    pronunciation: Callable[[str], Sequence[str]], source: dict,
                    min_quality: float = 0.65, min_coverage: float = 0.95,
                    min_agreement: float = 0.75) -> dict:
    """Convert word timestamps to a VALLR alignment document with audit metadata."""
    word_entries: list[list[float | str]] = []
    phone_entries: list[list[float | str]] = []
    unknown_words: list[str] = []
    covered_words = 0
    for word in transcript.words:
        normalized = normalize_word(word.text)
        if not normalized or word.end <= word.start:
            continue
        word_entries.append([word.start, word.end, normalized])
        phones = list(pronunciation(normalized))
        if not phones:
            unknown_words.append(normalized)
            continue
        covered_words += 1
        phone_duration = (word.end - word.start) / len(phones)
        for index, phone in enumerate(phones):
            start = word.start + index * phone_duration
            phone_entries.append([start, word.start + (index + 1) * phone_duration, phone])
    coverage = covered_words / max(len(word_entries), 1)
    asr_confidence = mean_word_confidence(transcript)
    language_confidence = min(1.0, max(0.0, transcript.language_probability))
    quality = math.prod((coverage, agreement, asr_confidence, language_confidence)) ** 0.25
    reasons = []
    if not phone_entries:
        reasons.append("no_phonemes")
    if coverage < min_coverage:
        reasons.append("low_pronunciation_coverage")
    if agreement < min_agreement:
        reasons.append("low_teacher_agreement")
    if quality < min_quality:
        reasons.append("low_quality_score")
    end = max(transcript.duration or 0.0,
              max((float(entry[1]) for entry in word_entries), default=0.0))
    return {
        "start": 0.0,
        "end": end,
        "tiers": {
            "words": {"type": "interval", "entries": word_entries},
            "phones": {"type": "interval", "entries": phone_entries},
        },
        "label_status": "accepted" if not reasons else "rejected",
        "quality": {
            "score": quality,
            "teacher_agreement": agreement,
            "mean_word_probability": asr_confidence,
            "language_probability": language_confidence,
            "pronunciation_coverage": coverage,
            "unknown_words": unknown_words,
            "rejection_reasons": reasons,
        },
        "provenance": {
            **source,
            "teacher": transcript.teacher,
            "language": transcript.language,
            "phone_inventory": "CMUdict ARPAbet",
            "phone_timing": "uniform subdivision of teacher word timestamps",
            "machine_generated": True,
        },
    }
