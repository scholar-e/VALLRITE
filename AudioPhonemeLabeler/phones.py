"""Direct acoustic phone recognition and explicit IPA-to-ARPAbet mapping."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


# Deliberately conservative: unmapped multilingual tokens lower confidence.
IPA_TO_ARPABET = {
    "p": "P", "b": "B", "t": "T", "d": "D", "k": "K", "ɡ": "G",
    "g": "G", "f": "F", "v": "V", "θ": "TH", "ð": "DH", "s": "S",
    "z": "Z", "ʃ": "SH", "ʒ": "ZH", "h": "HH", "tʃ": "CH", "dʒ": "JH",
    "m": "M", "n": "N", "ŋ": "NG", "l": "L", "ɫ": "L", "ɹ": "R",
    "r": "R", "ɾ": "T", "j": "Y", "w": "W", "i": "IY", "iː": "IY",
    "ɪ": "IH", "ᵻ": "IH", "e": "EY", "eː": "EY", "eɪ": "EY",
    "ɛ": "EH", "æ": "AE", "a": "AE", "ɑ": "AA", "ɑː": "AA",
    "ɒ": "AA", "ɔ": "AO", "ɔː": "AO", "o": "OW", "oː": "OW",
    "oʊ": "OW", "əʊ": "OW", "ʊ": "UH", "u": "UW", "uː": "UW",
    "ʌ": "AH", "ɐ": "AH", "ə": "AH", "ɜ": "ER", "ɜː": "ER",
    "ɚ": "ER", "ɝ": "ER", "aɪ": "AY", "aʊ": "AW", "ɔɪ": "OY",
    "oɪ": "OY", "ɑːɹ": "AA", "ɔːɹ": "AO", "ɪɹ": "IH", "ɛɹ": "EH",
    "ʊɹ": "UH", "əl": "AH",
}


@dataclass(frozen=True)
class DirectPhones:
    model: str
    raw: tuple[str, ...]
    arpabet: tuple[str, ...]
    mapping_coverage: float


def map_ipa_tokens(tokens: list[str]) -> DirectPhones:
    mapped = tuple(IPA_TO_ARPABET[token] for token in tokens if token in IPA_TO_ARPABET)
    return DirectPhones(
        model="",
        raw=tuple(tokens),
        arpabet=mapped,
        mapping_coverage=len(mapped) / max(len(tokens), 1),
    )


class Wav2Vec2PhoneTeacher:
    """Direct CTC phone recognizer, independent of the Whisper transcript."""

    def __init__(self, model_name: str, device: str = "auto",
                 local_files_only: bool = False, cache_dir: Path | None = None):
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["HF_HOME"] = str(cache_dir.resolve())
            os.environ["HF_HUB_CACHE"] = str((cache_dir / "hub").resolve())
            os.environ["HF_XET_CACHE"] = str((cache_dir / "xet").resolve())
        try:
            import torch
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        except ImportError as error:
            raise RuntimeError(
                "transformers is required; install requirements-labeler.txt"
            ) from error
        selected_device = ("cuda" if device == "auto" and torch.cuda.is_available()
                           else "cpu" if device == "auto" else device)
        self.name = model_name
        self.device = torch.device(selected_device)
        self.processor = Wav2Vec2Processor.from_pretrained(
            model_name, local_files_only=local_files_only, cache_dir=cache_dir
        )
        self.model = Wav2Vec2ForCTC.from_pretrained(
            model_name, local_files_only=local_files_only, cache_dir=cache_dir
        ).to(self.device).eval()

    def recognize(self, wav_path: Path) -> DirectPhones:
        import wave

        import numpy as np
        import torch

        with wave.open(str(wav_path), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getframerate() != 16000 \
                    or wav.getsampwidth() != 2:
                raise ValueError("phone teacher requires mono 16 kHz PCM16 WAV")
            audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        inputs = self.processor(audio.astype(np.float32) / 32768.0,
                                sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.to(self.device)
        with torch.inference_mode():
            predicted = self.model(input_values).logits.argmax(dim=-1)
        text = self.processor.batch_decode(predicted.cpu())[0]
        result = map_ipa_tokens(text.split())
        return DirectPhones(self.name, result.raw, result.arpabet, result.mapping_coverage)
