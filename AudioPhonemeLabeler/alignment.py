"""Audio normalization and optional Montreal Forced Aligner integration."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess


def normalize_audio(source: Path, destination: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for hybrid audio labelling")
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(source),
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-y",
         str(destination)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode or not destination.is_file():
        raise RuntimeError(result.stderr.strip() or f"ffmpeg exited {result.returncode}")


class MontrealForcedAligner:
    def __init__(self, dictionary: str = "english_us_arpa",
                 acoustic_model: str = "english_us_arpa"):
        self.executable = shutil.which("mfa")
        self.dictionary = dictionary
        self.acoustic_model = acoustic_model

    @property
    def available(self) -> bool:
        return self.executable is not None

    def align(self, wav_path: Path, words: list[str], work_dir: Path) -> dict:
        if not self.executable:
            raise RuntimeError("Montreal Forced Aligner executable is not installed")
        transcript = work_dir / "transcript.lab"
        output = work_dir / "forced-alignment.json"
        transcript.write_text(" ".join(words) + "\n", encoding="utf-8")
        result = subprocess.run(
            [self.executable, "align_one", str(wav_path), str(transcript),
             self.dictionary, self.acoustic_model, str(output),
             "--output_format", "json", "--clean"],
            capture_output=True, text=True, check=False,
        )
        if result.returncode or not output.is_file():
            message = (result.stderr or result.stdout).strip()
            raise RuntimeError(message or f"mfa align_one exited {result.returncode}")
        return json.loads(output.read_text(encoding="utf-8"))
