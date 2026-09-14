"""LRS3 loaders for image, 68-point geometry, and hybrid CTC training."""
from __future__ import annotations

import json
from pathlib import Path
import pickle
import re

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from VisualPhoneme.data import (
    CROPS,
    PHONE_TO_ID,
    landmark_feature_dimensions,
    letterbox,
    transform_landmarks,
)


# Put semantically meaningful lip points first so the tongue-gated model's
# aperture calculation remains valid: corners 48/54 and inner lip 62/66.
LRS3_LANDMARK_ORDER = (48, 54, 62, 66) + tuple(
    index for index in range(68) if index not in {48, 54, 62, 66}
)
LRS3_LANDMARK_POINTS = len(LRS3_LANDMARK_ORDER)
MOUTH_POSITIONS = tuple(
    position for position, source in enumerate(LRS3_LANDMARK_ORDER)
    if 48 <= source <= 67
)
LEFT_EYE_POSITIONS = tuple(
    position for position, source in enumerate(LRS3_LANDMARK_ORDER)
    if 36 <= source <= 41
)
RIGHT_EYE_POSITIONS = tuple(
    position for position, source in enumerate(LRS3_LANDMARK_ORDER)
    if 42 <= source <= 47
)


def _words(value: str) -> list[str]:
    return re.findall(r"[a-z']+", value.lower())


def _sequence_agreement(left: list[str], right: list[str]) -> float:
    previous = list(range(len(right) + 1))
    for row, left_value in enumerate(left, 1):
        current = [row]
        for column, right_value in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (left_value != right_value)))
        previous = current
    return 1.0 - previous[-1] / max(len(left), len(right), 1)


def _teacher_chunks(row: dict, document: dict, max_phones: int,
                    transcript_agreement: float) -> list[dict]:
    teacher_transcripts = document.get("provenance", {}).get("teacher_transcripts", [])
    teacher_words = list(teacher_transcripts[0]) if teacher_transcripts else [
        str(entry[2]) for entry in document["tiers"]["words"]["entries"]
    ]
    agreement = _sequence_agreement(_words(row["transcript"]), teacher_words)
    if agreement < transcript_agreement:
        return []
    phone_entries = document["tiers"]["phones"]["entries"]
    groups = []
    for start, end, word in document["tiers"]["words"]["entries"]:
        phones = [re.sub(r"\d", "", str(entry[2])).upper() for entry in phone_entries
                  if float(start) - 1e-4 <= (float(entry[0]) + float(entry[1])) / 2
                  <= float(end) + 1e-4]
        phones = [phone for phone in phones if phone in PHONE_TO_ID]
        if phones:
            groups.append((float(start), float(end), str(word), phones))
    packed = []
    current = []
    current_phones = 0
    for group in groups:
        if current and current_phones + len(group[3]) > max_phones:
            packed.append(current)
            current = []
            current_phones = 0
        current.append(group)
        current_phones += len(group[3])
    if current:
        packed.append(current)
    chunks = []
    fps = float(row["fps"])
    source_frames = int(row["frames"])
    for index, words in enumerate(packed):
        start = max(0.0, words[0][0] - 0.04)
        end = min(source_frames / fps, words[-1][1] + 0.04)
        frame_start = max(0, int(np.floor(start * fps)))
        frame_end = min(source_frames, int(np.ceil(end * fps)))
        phones = [phone for group in words for phone in group[3]]
        chunk = dict(row)
        chunk.update({
            "source_clip_id": row["clip_id"],
            "clip_id": f"{row['clip_id']}@{frame_start}:{frame_end}",
            "frame_start": frame_start,
            "frame_end": frame_end,
            "frames": frame_end - frame_start,
            "phonemes": phones,
            "transcript": " ".join(group[2] for group in words),
            "teacher_transcript_agreement": agreement,
            "teacher_chunk_index": index,
        })
        aligned = []
        relevant_entries = [entry for entry in phone_entries
                            if words[0][0] - 1e-4 <= (float(entry[0]) + float(entry[1])) / 2
                            <= words[-1][1] + 1e-4]
        for frame in range(frame_start, frame_end):
            timestamp = (frame + 0.5) / fps
            phone_id = -100
            for phone_start, phone_end, phone in relevant_entries:
                if float(phone_start) <= timestamp < float(phone_end):
                    normalized = re.sub(r"\d", "", str(phone)).upper()
                    phone_id = PHONE_TO_ID.get(normalized, -100)
                    break
            aligned.append(phone_id)
        chunk["frame_phone_ids"] = aligned
        chunks.append(chunk)
    return chunks


class _NumpyUnpickler(pickle.Unpickler):
    """Load legacy NumPy landmark pickles without permitting arbitrary globals."""

    ALLOWED = {
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
    }

    def find_class(self, module: str, name: str):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"forbidden pickle global: {module}.{name}")
        return super().find_class(module, name)


def load_lrs3_landmarks(path: Path) -> np.ndarray:
    with path.open("rb") as source:
        value = _NumpyUnpickler(source).load()
    if not isinstance(value, (list, np.ndarray)):
        raise ValueError(f"invalid LRS3 landmark container in {path}: {type(value)}")
    points = np.full((len(value), 68, 2), np.nan, dtype=np.float32)
    for index, frame in enumerate(value):
        if frame is None:
            continue
        array = np.asarray(frame, dtype=np.float32)
        if array.shape != (68, 2):
            raise ValueError(f"invalid LRS3 landmark frame in {path}: {array.shape}")
        points[index] = array
    return points[:, LRS3_LANDMARK_ORDER]


def eye_normalize_lrs3(points: torch.Tensor) -> torch.Tensor:
    valid = torch.isfinite(points).all(dim=(1, 2))
    if not valid.any():
        return points
    left = points[:, LEFT_EYE_POSITIONS].mean(dim=1)
    right = points[:, RIGHT_EYE_POSITIONS].mean(dim=1)
    center = (left + right) / 2
    scale = torch.linalg.vector_norm(right - left, dim=1).clamp_min(1e-4)
    return (points - center[:, None, :]) / scale[:, None, None]


def _mouth_crop(frame: np.ndarray, points: np.ndarray | None) -> np.ndarray:
    height, width = frame.shape[:2]
    if points is None or not np.isfinite(points).all():
        x0, y0, x1, y1 = CROPS["mouth"]
        return frame[round(y0 * height):round(y1 * height),
                     round(x0 * width):round(x1 * width)]
    mouth = points[list(MOUTH_POSITIONS)]
    low = mouth.min(axis=0)
    high = mouth.max(axis=0)
    center = (low + high) / 2
    side = max(float(high[0] - low[0]) * 1.55,
               float(high[1] - low[1]) * 2.30, 8.0)
    left = max(0, round(center[0] - side / 2))
    right = min(width, round(center[0] + side / 2))
    top = max(0, round(center[1] - side / 2))
    bottom = min(height, round(center[1] + side / 2))
    return frame[top:bottom, left:right]


def decode_lrs3_video(path: Path, landmarks: np.ndarray | None, crop: str,
                      size: int) -> torch.Tensor:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"cannot open video: {path}")
    frames = []
    try:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if crop == "mouth":
                current = landmarks[index] if landmarks is not None and index < len(landmarks) else None
                gray = _mouth_crop(gray, current)
            frames.append(letterbox(gray, size))
            index += 1
    finally:
        capture.release()
    if not frames:
        raise OSError(f"video contains no decodable frames: {path}")
    return torch.from_numpy(np.stack(frames)).unsqueeze(1).float().div_(255)


def _load_cached_video(cache_path: Path, size: int) -> torch.Tensor:
    frames = np.load(cache_path, allow_pickle=False)
    if frames.ndim != 3 or frames.shape[1:] != (size, size) or frames.dtype != np.uint8:
        raise ValueError(f"invalid frame cache: {cache_path}")
    return torch.from_numpy(np.asarray(frames)).unsqueeze(1).float().div_(255)


class Lrs3Clips(Dataset):
    """Read prepared LRS3 manifests with a GridClips-compatible item contract."""

    landmark_points = LRS3_LANDMARK_POINTS

    def __init__(
        self,
        root: Path,
        split: str,
        size: int = 96,
        crop: str = "mouth",
        limit: int | None = None,
        augment: bool = False,
        include_landmarks: bool = False,
        include_video: bool = True,
        horizontal_flip: bool = True,
        coordinate_mode: str = "eye-normalized",
        frame_cache_dir: Path | None = None,
        coordinate_features: str = "position",
        landmark_jitter: float = 0.0,
        point_dropout: float = 0.0,
        frame_span_dropout: float = 0.0,
        max_frame_span: int = 5,
        teacher_labels_dir: Path | None = None,
        max_chunk_phones: int = 0,
        min_teacher_transcript_agreement: float = 0.8,
        include_frame_targets: bool = False,
    ):
        if split not in {"train", "validation", "test"} or crop not in CROPS:
            raise ValueError("invalid LRS3 split or crop")
        if coordinate_mode not in {"eye-normalized", "clip-centered", "constant"}:
            raise ValueError(f"invalid coordinate mode: {coordinate_mode}")
        if max_chunk_phones < 0 or not 0 <= min_teacher_transcript_agreement <= 1:
            raise ValueError("invalid teacher chunk settings")
        if (teacher_labels_dir is None) != (max_chunk_phones == 0):
            raise ValueError("teacher labels and a positive chunk size must be used together")
        if include_frame_targets and teacher_labels_dir is None:
            raise ValueError("frame targets require teacher chunks")
        landmark_feature_dimensions(coordinate_features)
        self.root = root
        self.split = split
        self.size = size
        self.crop_name = crop
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
        self.include_frame_targets = include_frame_targets
        if not include_video and not include_landmarks:
            raise ValueError("at least one input modality must be enabled")
        manifest = root / "manifests" / f"{split}.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(f"prepare LRS3 first: missing {manifest}")
        candidates = [json.loads(line) for line in manifest.read_text().splitlines()]
        self.excluded = []
        self.rows = []
        if teacher_labels_dir is not None:
            if split != "train":
                raise ValueError("teacher chunks are training-only")
            if limit is not None:
                candidates = candidates[:limit]
            label_manifest = teacher_labels_dir / "labels.jsonl"
            if not label_manifest.is_file():
                raise FileNotFoundError(f"missing teacher manifest: {label_manifest}")
            labels = {record["source"]: record for record in (
                json.loads(line) for line in label_manifest.read_text().splitlines()
                if line.strip())}
            expanded = []
            for row in candidates:
                source = str((root / row["video"]).resolve())
                record = labels.get(source)
                if not record or not record.get("alignment"):
                    self.excluded.append({"clip_id": row["clip_id"],
                                          "reason": "no_teacher_alignment"})
                    continue
                document = json.loads(Path(record["alignment"]).read_text())
                if document.get("label_status") != "accepted":
                    self.excluded.append({"clip_id": row["clip_id"],
                                          "reason": "teacher_rejected"})
                    continue
                chunks = _teacher_chunks(row, document, max_chunk_phones,
                                          min_teacher_transcript_agreement)
                if not chunks:
                    self.excluded.append({"clip_id": row["clip_id"],
                                          "reason": "teacher_transcript_mismatch"})
                    continue
                expanded.extend(chunks)
            candidates = expanded
        for row in candidates:
            target = tuple(PHONE_TO_ID[phone] for phone in row["phonemes"])
            repeated = sum(left == right for left, right in zip(target, target[1:]))
            if float(row.get("pronunciation_coverage", 1.0)) < 0.95:
                self.excluded.append({"clip_id": row["clip_id"],
                                      "reason": "low_pronunciation_coverage"})
            elif not target or len(target) + repeated > int(row["frames"]):
                self.excluded.append({"clip_id": row["clip_id"], "reason": "invalid_ctc"})
            elif include_landmarks and not row.get("landmarks"):
                self.excluded.append({"clip_id": row["clip_id"], "reason": "no_landmark_join"})
            else:
                self.rows.append(row)
        if limit is not None and teacher_labels_dir is None:
            self.rows = self.rows[:limit]

    def __len__(self) -> int:
        return len(self.rows)

    def target_sequences(self) -> list[tuple[int, ...]]:
        return [tuple(PHONE_TO_ID[phone] for phone in row["phonemes"])
                for row in self.rows]

    def _landmarks(self, row: dict) -> tuple[np.ndarray | None, torch.Tensor | None]:
        if not row.get("landmarks"):
            return None, None
        raw = load_lrs3_landmarks(self.root / row["landmarks"])
        normalized = eye_normalize_lrs3(torch.from_numpy(raw.copy()))
        return raw, normalized

    def _video(self, row: dict, raw_landmarks: np.ndarray | None) -> torch.Tensor:
        source = self.root / row["video"]
        cache_path = None
        if self.frame_cache_dir is not None:
            cache_id = row.get("source_clip_id", row["clip_id"])
            safe_id = cache_id.replace("/", "_").replace(":", "_")
            cache_path = self.frame_cache_dir / self.crop_name / str(self.size) / f"{safe_id}.npy"
            if cache_path.is_file():
                return _load_cached_video(cache_path, self.size)
        if row["source_type"] == "npy-mouth":
            frames = np.load(source, allow_pickle=False)
            if frames.ndim != 3:
                raise ValueError(f"invalid LRS3 test video array: {source}")
            if self.crop_name != "mouth":
                raise ValueError("LRS3 parquet test videos support only mouth crop")
            video = torch.from_numpy(np.stack([letterbox(frame, self.size) for frame in frames]))
            video = video.unsqueeze(1).float().div_(255)
        else:
            video = decode_lrs3_video(source, raw_landmarks, self.crop_name, self.size)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_name(f"{cache_path.name}.{id(self)}.tmp")
            with temporary.open("wb") as output:
                np.save(output, video.squeeze(1).mul(255).round().byte().numpy(), allow_pickle=False)
            temporary.replace(cache_path)
        return video

    def __getitem__(self, index: int):
        row = self.rows[index]
        raw_landmarks, normalized_landmarks = self._landmarks(row)
        video = self._video(row, raw_landmarks) if self.include_video else None
        landmarks = None
        landmark_mask = None
        if self.include_landmarks:
            if normalized_landmarks is None:
                raise ValueError(f"missing landmarks for {row['clip_id']}")
            landmarks, landmark_mask = transform_landmarks(
                normalized_landmarks, self.coordinate_mode, self.coordinate_features,
                self.augment, self.landmark_jitter, self.point_dropout,
                self.frame_span_dropout, self.max_frame_span,
            )
        frame_start = int(row.get("frame_start", 0))
        frame_end = int(row.get("frame_end", min(
            len(value) for value in (video, landmarks) if value is not None
        )))
        if video is not None:
            video = video[frame_start:frame_end]
        if landmarks is not None:
            landmarks = landmarks[frame_start:frame_end]
            landmark_mask = landmark_mask[frame_start:frame_end]
        available = [len(value) for value in (video, landmarks) if value is not None]
        frames = min(available)
        if video is not None:
            video = video[:frames]
            if self.augment:
                video = torch.clamp(
                    video * np.random.uniform(0.85, 1.15) + np.random.uniform(-0.06, 0.06),
                    0, 1,
                )
                if self.horizontal_flip and np.random.random() < 0.5:
                    video = torch.flip(video, dims=(-1,))
        if landmarks is not None:
            landmarks = landmarks[:frames]
            landmark_mask = landmark_mask[:frames]
        target = torch.tensor([PHONE_TO_ID[phone] for phone in row["phonemes"]],
                              dtype=torch.long)
        frame_targets = None
        if self.include_frame_targets:
            frame_targets = torch.tensor(row["frame_phone_ids"], dtype=torch.long)
            frame_targets = frame_targets[:frames]
        ctc_steps = len(target) + int((target[1:] == target[:-1]).sum())
        if ctc_steps > frames:
            raise ValueError(
                f"decoded frames cannot align CTC target for {row['clip_id']}: "
                f"frames={frames} minimum_ctc_steps={ctc_steps}"
            )
        if self.include_landmarks:
            if video is None:
                if frame_targets is not None:
                    return landmarks, landmark_mask, target, frame_targets, row["clip_id"]
                return landmarks, landmark_mask, target, row["clip_id"]
            if frame_targets is not None:
                return video, landmarks, landmark_mask, target, frame_targets, row["clip_id"]
            return video, landmarks, landmark_mask, target, row["clip_id"]
        if frame_targets is not None:
            return video, target, frame_targets, row["clip_id"]
        return video, target, row["clip_id"]
