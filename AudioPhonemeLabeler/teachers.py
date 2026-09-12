"""Lazy-loaded audio teacher implementations."""
from __future__ import annotations

import os
from pathlib import Path

from .core import TeacherTranscript, Word


class FasterWhisperTeacher:
    """Large Whisper teacher with word timestamps through CTranslate2."""

    def __init__(self, model_name: str, device: str = "auto", compute_type: str = "default",
                 local_files_only: bool = False, cache_dir: Path | None = None):
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["HF_HOME"] = str(cache_dir.resolve())
            os.environ["HF_HUB_CACHE"] = str((cache_dir / "hub").resolve())
            os.environ["HF_XET_CACHE"] = str((cache_dir / "xet").resolve())
        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise RuntimeError(
                "faster-whisper is required; install requirements-labeler.txt"
            ) from error
        self.name = f"faster-whisper/{model_name}"
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type,
                                  local_files_only=local_files_only,
                                  download_root=str(cache_dir) if cache_dir else None)

    def transcribe(self, media: Path, language: str | None = "en",
                   beam_size: int = 5) -> TeacherTranscript:
        segments, information = self.model.transcribe(
            str(media), language=language, beam_size=beam_size, word_timestamps=True,
            vad_filter=True, condition_on_previous_text=False,
        )
        words = []
        for segment in segments:
            for word in segment.words or ():
                words.append(Word(word.word, float(word.start), float(word.end),
                                  float(word.probability)))
        return TeacherTranscript(
            teacher=self.name,
            language=information.language,
            language_probability=float(information.language_probability),
            words=tuple(words),
            duration=float(information.duration),
        )
