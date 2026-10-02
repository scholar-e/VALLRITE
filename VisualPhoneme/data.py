"""GRID clip loading and transcript-level phoneme targets."""
from __future__ import annotations

from functools import lru_cache
import json
import math
import os
from pathlib import Path
import random
import re

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

PHONEMES = tuple(
    "AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG "
    "OW OY P R S SH T TH UH UW V W Y Z ZH".split()
)
PHONE_TO_ID = {phone: index + 1 for index, phone in enumerate(PHONEMES)}
BLANK_ID = 0

CROPS = {
    # GRID is a centered, fixed-camera corpus. Fractions are configurable in the CLI.
    "face": (0.18, 0.02, 0.82, 0.94),
    "mouth": (0.27, 0.43, 0.73, 0.81),
    "full": (0.0, 0.0, 1.0, 1.0),
}


def normalize_phone(raw: str) -> str | None:
    phone = re.sub(r"\d", "", raw).upper()
    return phone if phone in PHONE_TO_ID else None


@lru_cache(maxsize=16384)
def phoneme_target(alignment: str) -> tuple[int, ...]:
    record = json.loads(Path(alignment).read_text())
    phones = [normalize_phone(entry[2]) for entry in record["tiers"]["phones"]["entries"]]
    return tuple(PHONE_TO_ID[phone] for phone in phones if phone is not None)


def letterbox(frame: np.ndarray, size: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(size / width, size / height)
    resized = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((size, size), dtype=np.uint8)
    y = (size - resized.shape[0]) // 2
    x = (size - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def decode_video(path: Path, crop: tuple[float, float, float, float], size: int) -> torch.Tensor:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"cannot open video: {path}")
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            x0, y0, x1, y1 = crop
            bounds = (round(x0 * width), round(y0 * height), round(x1 * width), round(y1 * height))
            left, top, right, bottom = bounds
            gray = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
            frames.append(letterbox(gray, size))
    finally:
        capture.release()
    if not frames:
        raise OSError(f"video contains no decodable frames: {path}")
    return torch.from_numpy(np.stack(frames)).unsqueeze(1).float().div_(255.0)


def load_video(path: Path, crop: tuple[float, float, float, float], size: int,
               cache_path: Path | None = None) -> torch.Tensor:
    """Load a video, optionally persisting its resized uint8 frames after first decode."""
    if cache_path is not None and cache_path.is_file():
        frames = np.load(cache_path, allow_pickle=False)
        if frames.ndim != 3 or frames.shape[1:] != (size, size) or frames.dtype != np.uint8:
            raise ValueError(f"invalid frame cache: {cache_path}")
        return torch.from_numpy(np.asarray(frames)).unsqueeze(1).float().div_(255.0)
    video = decode_video(path, crop, size)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as output:
            np.save(output, video.squeeze(1).mul(255).round().byte().numpy(), allow_pickle=False)
        temporary.replace(cache_path)
    return video


def landmark_feature_dimensions(features: str) -> int:
    if features == "position":
        return 2
    if features == "motion":
        return 10
    raise ValueError(f"invalid coordinate features: {features}")


def _temporal_delta(points: torch.Tensor, mask: torch.Tensor, lag: int) -> torch.Tensor:
    delta = torch.zeros_like(points)
    valid = mask[lag:] & mask[:-lag]
    delta[lag:][valid] = points[lag:][valid] - points[:-lag][valid]
    return delta


def transform_landmarks(landmarks: torch.Tensor, mode: str, features: str = "position",
                        augment: bool = False, jitter_std: float = 0.0,
                        point_dropout: float = 0.0, frame_span_dropout: float = 0.0,
                        max_frame_span: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a checkpointed coordinate representation and return its frame mask."""
    if mode not in {"eye-normalized", "pose-frontalized", "clip-centered", "constant"}:
        raise ValueError(f"invalid coordinate mode: {mode}")
    landmark_mask = torch.isfinite(landmarks).all(dim=(1, 2))
    if mode == "clip-centered" and landmark_mask.any():
        center = landmarks[landmark_mask].median(dim=0).values
        landmarks = landmarks - center
    elif mode == "constant":
        landmarks = torch.zeros_like(landmarks)
    landmarks = torch.nan_to_num(landmarks)
    if augment and jitter_std:
        landmarks = landmarks + torch.randn_like(landmarks) * jitter_std
    if augment and point_dropout:
        dropped = torch.rand(landmarks.shape[:2]) < point_dropout
        landmarks[dropped] = 0
    if augment and frame_span_dropout and random.random() < frame_span_dropout:
        span = random.randint(1, min(max_frame_span, len(landmarks)))
        start = random.randint(0, len(landmarks) - span)
        landmarks[start:start + span] = 0
        landmark_mask[start:start + span] = False
    if features == "position":
        return landmarks, landmark_mask
    landmark_feature_dimensions(features)
    velocity = _temporal_delta(landmarks, landmark_mask, 1)
    acceleration = _temporal_delta(velocity, landmark_mask, 1)
    return torch.cat((landmarks, velocity, acceleration,
                      _temporal_delta(landmarks, landmark_mask, 2),
                      _temporal_delta(landmarks, landmark_mask, 4)), dim=-1), landmark_mask


def fit_bigram_log_probs(sequences, classes: int, smoothing: float = 1.0) -> torch.Tensor:
    """Fit a smoothed phone transition model; row zero represents sequence start."""
    if classes < 2 or smoothing <= 0:
        raise ValueError("classes must exceed one and smoothing must be positive")
    counts = torch.full((classes, classes), smoothing, dtype=torch.float64)
    counts[:, BLANK_ID] = 0
    for sequence in sequences:
        previous = BLANK_ID
        for token in sequence:
            counts[previous, int(token)] += 1
            previous = int(token)
    probabilities = counts[:, 1:] / counts[:, 1:].sum(dim=1, keepdim=True)
    result = torch.zeros((classes, classes), dtype=torch.float32)
    result[:, 1:] = probabilities.log().float()
    return result


def fit_trigram_log_probs(sequences, classes: int, smoothing: float = 1.0) -> torch.Tensor:
    """Fit a smoothed trigram LM; zero-valued history slots mark sequence start."""
    if classes < 2 or smoothing <= 0:
        raise ValueError("classes must exceed one and smoothing must be positive")
    counts = torch.full((classes, classes, classes), smoothing, dtype=torch.float64)
    counts[:, :, BLANK_ID] = 0
    for sequence in sequences:
        older = previous = BLANK_ID
        for token in sequence:
            token = int(token)
            counts[older, previous, token] += 1
            older, previous = previous, token
    probabilities = counts[:, :, 1:] / counts[:, :, 1:].sum(dim=2, keepdim=True)
    result = torch.zeros((classes, classes, classes), dtype=torch.float32)
    result[:, :, 1:] = probabilities.log().float()
    return result


class GridClips(Dataset):
    landmark_points = 41

    def __init__(self, root: Path, split: str, size: int = 96, crop: str = "face",
                 limit: int | None = None, augment: bool = False,
                 include_landmarks: bool = False, include_video: bool = True,
                 horizontal_flip: bool = True, coordinate_mode: str = "eye-normalized",
                 frame_cache_dir: Path | None = None, coordinate_features: str = "position",
                 landmark_jitter: float = 0.0, point_dropout: float = 0.0,
                 frame_span_dropout: float = 0.0, max_frame_span: int = 5):
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"invalid split: {split}")
        if crop not in CROPS:
            raise ValueError(f"invalid crop: {crop}")
        if coordinate_mode not in {"eye-normalized", "pose-frontalized",
                                   "clip-centered", "constant"}:
            raise ValueError(f"invalid coordinate mode: {coordinate_mode}")
        landmark_feature_dimensions(coordinate_features)
        if (landmark_jitter < 0 or not 0 <= point_dropout < 1
                or not 0 <= frame_span_dropout <= 1 or max_frame_span < 1):
            raise ValueError("invalid landmark augmentation")
        self.root = root
        self.size = size
        self.crop_name = crop
        self.crop = CROPS[crop]
        self.augment = augment
        self.include_landmarks = include_landmarks
        self.include_video = include_video
        self.horizontal_flip = horizontal_flip
        self.coordinate_mode = coordinate_mode
        self.frame_cache_dir = frame_cache_dir
        self.coordinate_features = coordinate_features
        self.landmark_jitter = landmark_jitter
        self.point_dropout = point_dropout
        self.frame_span_dropout = frame_span_dropout
        self.max_frame_span = max_frame_span
        if not include_video and not include_landmarks:
            raise ValueError("at least one input modality must be enabled")
        rows = [json.loads(line) for line in (root / "clips.jsonl").read_text().splitlines()]
        candidates = [row for row in rows if row["split"] == split]
        self.excluded = []
        self.rows = []
        for row in candidates:
            stem = Path(row["video"]).stem
            alignment = root / "landmark-experiment" / "aligned" / f"s{row['speaker_id']}" / (stem + ".json")
            target = phoneme_target(str(alignment))
            landmark_cache = root / "landmark-experiment" / "landmarks" / f"s{row['speaker_id']}" / (stem + ".npz")
            # The landmark pass provides a trustworthy count of frames OpenCV
            # actually decoded; container metadata can overstate corrupt clips.
            with np.load(landmark_cache) as cached:
                decoded_frames = len(cached["time_ms"])
            ctc_steps = len(target) + sum(left == right for left, right in zip(target, target[1:]))
            if not target or ctc_steps > decoded_frames:
                self.excluded.append({"clip_id": row["clip_id"], "decoded_frames": decoded_frames,
                                      "phones": len(target), "minimum_ctc_steps": ctc_steps})
            else:
                self.rows.append(row)
        if limit is not None:
            self.rows = self.rows[:limit]

    def __len__(self) -> int:
        return len(self.rows)

    def target_sequences(self) -> list[tuple[int, ...]]:
        sequences = []
        for row in self.rows:
            stem = Path(row["video"]).stem
            alignment = (self.root / "landmark-experiment" / "aligned"
                         / f"s{row['speaker_id']}" / f"{stem}.json")
            sequences.append(phoneme_target(str(alignment)))
        return sequences

    def __getitem__(self, index: int):
        row = self.rows[index]
        cache_path = None
        if self.frame_cache_dir is not None:
            cache_path = (self.frame_cache_dir / self.crop_name / str(self.size)
                          / Path(row["video"])).with_suffix(".npy")
        video = (load_video(self.root / row["video"], self.crop, self.size, cache_path)
                 if self.include_video else None)
        if self.augment and video is not None:
            video = torch.clamp(video * random.uniform(0.85, 1.15) + random.uniform(-0.06, 0.06), 0, 1)
            if self.horizontal_flip and random.random() < 0.5:
                video = torch.flip(video, dims=(-1,))
        stem = Path(row["video"]).stem
        alignment = self.root / "landmark-experiment" / "aligned" / f"s{row['speaker_id']}" / (stem + ".json")
        target = torch.tensor(phoneme_target(str(alignment)), dtype=torch.long)
        if self.include_landmarks:
            landmark_path = self.root / "landmark-experiment" / "landmarks" / f"s{row['speaker_id']}" / (stem + ".npz")
            with np.load(landmark_path) as cached:
                landmarks = torch.from_numpy(cached["coordinates"].astype(np.float32))
            frames = min(len(video), len(landmarks)) if video is not None else len(landmarks)
            if video is not None:
                video = video[:frames]
            landmarks = landmarks[:frames]
            landmarks, landmark_mask = transform_landmarks(
                landmarks, self.coordinate_mode, self.coordinate_features, self.augment,
                self.landmark_jitter, self.point_dropout, self.frame_span_dropout,
                self.max_frame_span)
        ctc_steps = len(target) + int((target[1:] == target[:-1]).sum())
        input_steps = video.shape[0] if video is not None else landmarks.shape[0]
        if len(target) == 0 or ctc_steps > input_steps:
            raise ValueError(f"invalid CTC target for {row['clip_id']}")
        if self.include_landmarks:
            if video is None:
                return landmarks, landmark_mask, target, row["clip_id"]
            return video, landmarks, landmark_mask, target, row["clip_id"]
        return video, target, row["clip_id"]


def collate_clips(items):
    videos, targets, clip_ids = zip(*items)
    lengths = torch.tensor([len(video) for video in videos], dtype=torch.long)
    target_lengths = torch.tensor([len(target) for target in targets], dtype=torch.long)
    padded = torch.zeros((len(videos), int(lengths.max()), *videos[0].shape[1:]), dtype=torch.float32)
    for index, video in enumerate(videos):
        padded[index, :len(video)] = video
    return padded, torch.cat(targets), lengths, target_lengths, clip_ids


def collate_fusion_clips(items):
    videos, landmarks, masks, targets, clip_ids = zip(*items)
    lengths = torch.tensor([len(video) for video in videos], dtype=torch.long)
    target_lengths = torch.tensor([len(target) for target in targets], dtype=torch.long)
    steps = int(lengths.max())
    padded_video = torch.zeros((len(videos), steps, *videos[0].shape[1:]), dtype=torch.float32)
    padded_landmarks = torch.zeros((len(videos), steps, *landmarks[0].shape[1:]), dtype=torch.float32)
    padded_masks = torch.zeros((len(videos), steps), dtype=torch.bool)
    for index, (video, points, mask) in enumerate(zip(videos, landmarks, masks)):
        padded_video[index, :len(video)] = video
        padded_landmarks[index, :len(points)] = points
        padded_masks[index, :len(mask)] = mask
    return (padded_video, padded_landmarks, padded_masks, torch.cat(targets),
            lengths, target_lengths, clip_ids)


def collate_landmark_clips(items):
    landmarks, masks, targets, clip_ids = zip(*items)
    lengths = torch.tensor([len(points) for points in landmarks], dtype=torch.long)
    target_lengths = torch.tensor([len(target) for target in targets], dtype=torch.long)
    steps = int(lengths.max())
    padded_landmarks = torch.zeros((len(landmarks), steps, *landmarks[0].shape[1:]), dtype=torch.float32)
    padded_masks = torch.zeros((len(landmarks), steps), dtype=torch.bool)
    for index, (points, mask) in enumerate(zip(landmarks, masks)):
        padded_landmarks[index, :len(points)] = points
        padded_masks[index, :len(mask)] = mask
    return (padded_landmarks, padded_masks, torch.cat(targets), lengths,
            target_lengths, clip_ids)


def _pad_frame_targets(frame_targets, steps: int) -> torch.Tensor:
    padded = torch.full((len(frame_targets), steps), -100, dtype=torch.long)
    for index, target in enumerate(frame_targets):
        padded[index, :len(target)] = target
    return padded


def collate_aligned_clips(items):
    videos, targets, frame_targets, clip_ids = zip(*items)
    base = collate_clips(list(zip(videos, targets, clip_ids)))
    video, packed, lengths, target_lengths, identifiers = base
    return (video, packed, lengths, target_lengths,
            _pad_frame_targets(frame_targets, video.shape[1]), identifiers)


def collate_aligned_fusion_clips(items):
    videos, landmarks, masks, targets, frame_targets, clip_ids = zip(*items)
    base = collate_fusion_clips(list(zip(videos, landmarks, masks, targets, clip_ids)))
    video, points, point_masks, packed, lengths, target_lengths, identifiers = base
    return (video, points, point_masks, packed, lengths, target_lengths,
            _pad_frame_targets(frame_targets, video.shape[1]), identifiers)


def collate_aligned_landmark_clips(items):
    landmarks, masks, targets, frame_targets, clip_ids = zip(*items)
    base = collate_landmark_clips(list(zip(landmarks, masks, targets, clip_ids)))
    points, point_masks, packed, lengths, target_lengths, identifiers = base
    return (points, point_masks, packed, lengths, target_lengths,
            _pad_frame_targets(frame_targets, points.shape[1]), identifiers)


def greedy_decode(logits: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
    paths = logits.argmax(-1).transpose(0, 1).cpu()
    hypotheses = []
    for path, length in zip(paths, lengths):
        collapsed = []
        previous = None
        for token in path[:int(length)]:
            value = int(token)
            if value != BLANK_ID and value != previous:
                collapsed.append(value)
            previous = value
        hypotheses.append(collapsed)
    return hypotheses


def _log_add(*values: float) -> float:
    finite = [value for value in values if value != -math.inf]
    if not finite:
        return -math.inf
    maximum = max(finite)
    return maximum + math.log(sum(math.exp(value - maximum) for value in finite))


def ctc_prefix_beam_search(log_probabilities: torch.Tensor, beam_width: int = 16,
                           top_n: int = 5,
                           token_top_k: int | None = None,
                           transition_log_probs: torch.Tensor | None = None,
                           lm_weight: float = 0.0,
                           token_bonus: float = 0.0) -> list[tuple[list[int], float]]:
    """Return CTC label sequences and log scores using prefix beam search."""
    if log_probabilities.ndim != 2:
        raise ValueError("log_probabilities must have shape [time,classes]")
    if top_n < 1 or beam_width < top_n:
        raise ValueError("beam_width must be at least top_n >= 1")
    if token_top_k is not None and token_top_k < 1:
        raise ValueError("token_top_k must be positive")
    if lm_weight < 0 or not math.isfinite(token_bonus):
        raise ValueError("lm_weight must be nonnegative")
    probabilities = log_probabilities.detach().float().cpu()
    transitions = transition_log_probs.detach().float().cpu() if transition_log_probs is not None else None
    classes = probabilities.shape[1]
    if transitions is not None and transitions.shape not in {
            (classes, classes), (classes, classes, classes)}:
        raise ValueError("transition_log_probs must be a bigram or trigram tensor")
    transition_values = transitions.tolist() if transitions is not None else None

    def language_bonus(prefix: tuple[int, ...], token: int) -> float:
        if transition_values is None:
            return token_bonus
        previous = prefix[-1] if prefix else BLANK_ID
        if transitions.ndim == 2:
            value = transition_values[previous][token]
        else:
            older = prefix[-2] if len(prefix) > 1 else BLANK_ID
            value = transition_values[older][previous][token]
        return lm_weight * value + token_bonus
    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, -math.inf)}
    for frame in probabilities:
        frame_values = frame.tolist()
        if token_top_k is None or token_top_k >= len(frame_values):
            token_ids = range(len(frame_values))
        else:
            token_ids = torch.topk(frame, token_top_k).indices.tolist()
            if BLANK_ID not in token_ids:
                token_ids.append(BLANK_ID)
        next_beams: dict[tuple[int, ...], tuple[float, float]] = {}
        for prefix, (blank_score, token_score) in beams.items():
            for token in token_ids:
                value = frame_values[token]
                next_blank, next_token = next_beams.get(prefix, (-math.inf, -math.inf))
                if token == BLANK_ID:
                    next_blank = _log_add(next_blank, blank_score + value,
                                          token_score + value)
                    next_beams[prefix] = (next_blank, next_token)
                    continue
                if prefix and token == prefix[-1]:
                    next_token = _log_add(next_token, token_score + value)
                    next_beams[prefix] = (next_blank, next_token)
                    extended = prefix + (token,)
                    ext_blank, ext_token = next_beams.get(extended,
                                                           (-math.inf, -math.inf))
                    lm_bonus = language_bonus(prefix, token)
                    ext_token = _log_add(ext_token, blank_score + value + lm_bonus)
                    next_beams[extended] = (ext_blank, ext_token)
                else:
                    extended = prefix + (token,)
                    ext_blank, ext_token = next_beams.get(extended,
                                                           (-math.inf, -math.inf))
                    lm_bonus = language_bonus(prefix, token)
                    ext_token = _log_add(ext_token, blank_score + value + lm_bonus,
                                         token_score + value + lm_bonus)
                    next_beams[extended] = (ext_blank, ext_token)
        ranked = sorted(next_beams.items(),
                        key=lambda item: _log_add(*item[1]), reverse=True)
        beams = dict(ranked[:beam_width])
    ranked = sorted(beams.items(), key=lambda item: _log_add(*item[1]), reverse=True)
    return [(list(prefix), _log_add(*scores)) for prefix, scores in ranked[:top_n]]
