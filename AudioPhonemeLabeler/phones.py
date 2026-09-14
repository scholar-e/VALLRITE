"""Direct acoustic phone recognition and explicit IPA-to-ARPAbet mapping."""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import re


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


@dataclass(frozen=True)
class ForcedPhones:
    """Transcript-constrained phone intervals from a CTC acoustic model."""

    model: str
    entries: tuple[tuple[float, float, str], ...]
    path_confidence: float
    frame_duration: float


def ctc_viterbi_align(log_probs, target_token_ids: list[list[int]],
                      target_phones: list[str], blank_id: int,
                      duration: float) -> ForcedPhones:
    """Find the highest-scoring CTC path through a fixed phone sequence.

    Each target phone may be represented by several IPA vocabulary tokens. Their
    probabilities are summed before alignment, while blank absorbs silence and
    non-phone frames. This function is model-independent and directly testable.
    """
    import torch

    if log_probs.ndim != 2 or log_probs.shape[0] < 1:
        raise ValueError("log_probs must have shape [frames,vocabulary]")
    if len(target_token_ids) != len(target_phones) or not target_phones:
        raise ValueError("target phone/token groups must be nonempty and equal length")
    if not 0 <= blank_id < log_probs.shape[1]:
        raise ValueError("blank token is outside the acoustic vocabulary")
    if duration <= 0 or log_probs.shape[0] < len(target_phones):
        raise ValueError("audio has insufficient frames for the target phone sequence")
    vocabulary = log_probs.shape[1]
    if any(not ids or any(token < 0 or token >= vocabulary for token in ids)
           for ids in target_token_ids):
        raise ValueError("every target phone requires valid acoustic token IDs")

    scores = log_probs.detach().to(device="cpu", dtype=torch.float32)
    phone_scores = torch.stack([
        torch.logsumexp(scores[:, ids], dim=1) for ids in target_token_ids
    ], dim=1)
    frames = scores.shape[0]
    states = 2 * len(target_phones) + 1
    emissions = scores.new_empty((frames, states))
    emissions[:, 0::2] = scores[:, blank_id].unsqueeze(1)
    emissions[:, 1::2] = phone_scores

    negative_infinity = torch.tensor(float("-inf"), dtype=scores.dtype)
    previous = torch.full((states,), negative_infinity, dtype=scores.dtype)
    previous[0] = emissions[0, 0]
    if states > 1:
        previous[1] = emissions[0, 1]
    backpointers = torch.zeros((frames, states), dtype=torch.int8)
    for frame in range(1, frames):
        stay = previous
        advance = torch.cat((negative_infinity.view(1), previous[:-1]))
        skip = torch.cat((negative_infinity.repeat(2), previous[:-2]))
        # A two-state skip enters a phone state. CTC forbids that transition
        # when it would join two identical adjacent labels without a blank.
        skip_allowed = torch.zeros(states, dtype=torch.bool)
        for phone_index in range(1, len(target_phones)):
            state = 2 * phone_index + 1
            if target_phones[phone_index] != target_phones[phone_index - 1]:
                skip_allowed[state] = True
        skip = torch.where(skip_allowed, skip, negative_infinity)
        candidates = torch.stack((stay, advance, skip))
        best_scores, transitions = candidates.max(dim=0)
        previous = best_scores + emissions[frame]
        backpointers[frame] = transitions.to(torch.int8)

    end_candidates = [states - 1]
    if states > 1:
        end_candidates.append(states - 2)
    end_state = max(end_candidates, key=lambda state: float(previous[state]))
    final_score = float(previous[end_state])
    if not math.isfinite(final_score):
        raise ValueError("no valid CTC alignment path for the target phones")
    path = [end_state]
    for frame in range(frames - 1, 0, -1):
        end_state -= int(backpointers[frame, end_state])
        path.append(end_state)
    path.reverse()

    seconds_per_frame = duration / frames
    entries = []
    for phone_index, phone in enumerate(target_phones):
        state = 2 * phone_index + 1
        occupied = [frame for frame, path_state in enumerate(path)
                    if path_state == state]
        if not occupied:
            raise ValueError(f"CTC path omitted target phone {phone_index}: {phone}")
        entries.append((occupied[0] * seconds_per_frame,
                        (occupied[-1] + 1) * seconds_per_frame, phone))
    return ForcedPhones(
        model="",
        entries=tuple(entries),
        path_confidence=math.exp(final_score / frames),
        frame_duration=seconds_per_frame,
    )


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

    def _infer(self, wav_path: Path):
        """Return CPU log probabilities and duration for normalized PCM audio."""
        import wave

        import numpy as np
        import torch

        with wave.open(str(wav_path), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getframerate() != 16000 \
                    or wav.getsampwidth() != 2:
                raise ValueError("phone teacher requires mono 16 kHz PCM16 WAV")
            frames = wav.getnframes()
            audio = np.frombuffer(wav.readframes(frames), dtype="<i2")
        inputs = self.processor(audio.astype(np.float32) / 32768.0,
                                sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            logits = self.model(inputs.input_values.to(self.device)).logits
            log_probs = logits.log_softmax(dim=-1).cpu()[0]
        return log_probs, frames / 16000.0

    def recognize(self, wav_path: Path) -> DirectPhones:
        log_probs, _ = self._infer(wav_path)
        predicted = log_probs.argmax(dim=-1).unsqueeze(0)
        text = self.processor.batch_decode(predicted.cpu())[0]
        result = map_ipa_tokens(text.split())
        return DirectPhones(self.name, result.raw, result.arpabet, result.mapping_coverage)

    def recognize_and_align(self, wav_path: Path, target_phones: list[str]
                            ) -> tuple[DirectPhones, ForcedPhones]:
        """Recognize freely and force-align a known ARPAbet sequence in one pass."""
        log_probs, duration = self._infer(wav_path)
        predicted = log_probs.argmax(dim=-1).unsqueeze(0)
        text = self.processor.batch_decode(predicted)[0]
        mapped = map_ipa_tokens(text.split())
        direct = DirectPhones(self.name, mapped.raw, mapped.arpabet,
                              mapped.mapping_coverage)
        vocabulary = self.processor.tokenizer.get_vocab()
        ids_by_phone: dict[str, list[int]] = {}
        for token, token_id in vocabulary.items():
            arpabet = IPA_TO_ARPABET.get(token)
            if arpabet:
                ids_by_phone.setdefault(arpabet, []).append(int(token_id))
        normalized = [re.sub(r"\d", "", phone).upper() for phone in target_phones]
        missing = sorted({phone for phone in normalized if phone not in ids_by_phone})
        if missing:
            raise ValueError(f"acoustic vocabulary cannot represent phones: {missing}")
        forced = ctc_viterbi_align(
            log_probs, [ids_by_phone[phone] for phone in normalized],
            normalized, int(self.processor.tokenizer.pad_token_id), duration,
        )
        stress_preserving_entries = tuple(
            (start, end, original)
            for (start, end, _), original in zip(forced.entries, target_phones)
        )
        return direct, ForcedPhones(self.name, stress_preserving_entries,
                                    forced.path_confidence, forced.frame_duration)
