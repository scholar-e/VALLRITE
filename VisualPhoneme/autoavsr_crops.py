"""Auto-AVSR mouth crops from LRS3 video and supplied 68-point landmarks.

Reproduces `preparation/detectors/retinaface/video_process.py` from
https://github.com/mpc001/auto_avsr (Apache 2.0): fill missing landmarks by
linear interpolation, smooth them over a 12-frame window while keeping each
frame's own centre, warp the frame onto the 20-words mean face with a partial
affine fit on eight stable points, and cut a 96x96 patch around the mouth
(points 48-67). Output is uint8 [time,96,96] grayscale, the input expected by
`autoavsr_encoder.autoavsr_video_transform`.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import logging
from pathlib import Path
import time

import cv2
import numpy as np

from VisualPhoneme.lrs3_data import _NumpyUnpickler

LOGGER = logging.getLogger(__name__)
MEAN_FACE = Path(__file__).resolve().parent / "assets" / "autoavsr_20words_mean_face.npy"
STABLE_POINTS = (28, 33, 36, 39, 42, 45, 48, 54)
MOUTH = slice(48, 68)
WINDOW_MARGIN = 12
CROP = 96
CANVAS = 256


def load_raw_landmarks(path: Path) -> list[np.ndarray | None]:
    """Per-frame 68x2 points in the original iBUG order; None where undetected.

    `lrs3_data.load_lrs3_landmarks` reorders points for the landmark models, which
    would break the reference indices used here.
    """
    with path.open("rb") as source:
        value = _NumpyUnpickler(source).load()
    frames = []
    for frame in value:
        array = None if frame is None else np.asarray(frame, dtype=np.float64)
        if array is not None and array.shape != (68, 2):
            raise ValueError(f"invalid landmark frame shape {array.shape} in {path}")
        frames.append(array if array is not None and np.isfinite(array).all() else None)
    return frames


def interpolate_landmarks(landmarks: list[np.ndarray | None]) -> np.ndarray | None:
    """Linearly fill gaps; hold the first/last detection at the edges."""
    valid = [index for index, points in enumerate(landmarks) if points is not None]
    if not valid:
        return None
    filled = [None if points is None else np.asarray(points, dtype=np.float64)
              for points in landmarks]
    for start, stop in zip(valid, valid[1:]):
        for offset in range(1, stop - start):
            filled[start + offset] = (filled[start]
                                      + offset / (stop - start) * (filled[stop] - filled[start]))
    filled[:valid[0]] = [filled[valid[0]]] * valid[0]
    filled[valid[-1]:] = [filled[valid[-1]]] * (len(filled) - valid[-1])
    return np.stack(filled)


def crop_mouths(frames: list[np.ndarray], landmarks: np.ndarray,
                reference: np.ndarray) -> np.ndarray:
    """frames: grayscale uint8 images; landmarks [time,68,2] in frame pixels."""
    if len(frames) != len(landmarks):
        raise ValueError(f"{len(frames)} frames but {len(landmarks)} landmark frames")
    if len(landmarks) < WINDOW_MARGIN:
        raise ValueError(f"fewer than {WINDOW_MARGIN} frames")
    stable_reference = reference[list(STABLE_POINTS)]
    patches = []
    for index, frame in enumerate(frames):
        margin = min(WINDOW_MARGIN // 2, index, len(landmarks) - 1 - index)
        smoothed = landmarks[index - margin:index + margin + 1].mean(axis=0)
        smoothed = smoothed + landmarks[index].mean(axis=0) - smoothed.mean(axis=0)
        transform = cv2.estimateAffinePartial2D(smoothed[list(STABLE_POINTS)],
                                                stable_reference, method=cv2.LMEDS)[0]
        if transform is None:
            raise ValueError(f"affine estimation failed at frame {index}")
        warped = cv2.warpAffine(frame, transform, (CANVAS, CANVAS), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mouth = smoothed[MOUTH] @ transform[:, :2].T + transform[:, 2]
        center_x, center_y = mouth.mean(axis=0)
        half = CROP // 2
        if abs(center_y - CANVAS / 2) > half + 5 or abs(center_x - CANVAS / 2) > half + 5:
            raise ValueError(f"mouth too far from the aligned centre at frame {index}")
        top = int(round(np.clip(center_y - half, 0, CANVAS)))
        left = int(round(np.clip(center_x - half, 0, CANVAS)))
        patch = warped[top:int(round(np.clip(center_y + half, 0, CANVAS))),
                       left:int(round(np.clip(center_x + half, 0, CANVAS)))]
        if patch.shape != (CROP, CROP):
            raise ValueError(f"patch {patch.shape} at frame {index}")
        patches.append(patch)
    return np.stack(patches)


def read_gray_frames(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"cannot open video: {path}")
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    finally:
        capture.release()
    return frames


def _process(job: tuple[dict, str, str]) -> dict:
    row, data_root, output_root = job
    root, output = Path(data_root), Path(output_root)
    relative = Path(row["video"]).relative_to("raw").with_suffix(".npy")
    destination = output / relative
    record = {"clip_id": row["clip_id"], "crops": str(relative)}
    if destination.exists():
        return {**record, "status": "exists",
                "frames": int(np.load(destination, mmap_mode="r").shape[0])}
    try:
        frames = read_gray_frames(root / row["video"])
        landmarks = interpolate_landmarks(load_raw_landmarks(root / row["landmarks"]))
        if landmarks is None:
            raise ValueError("no landmarks detected in any frame")
        # Landmark and decoded frame counts occasionally differ by a frame.
        count = min(len(frames), len(landmarks))
        if max(len(frames), len(landmarks)) - count > 2:
            raise ValueError(f"{len(frames)} frames but {len(landmarks)} landmark frames")
        patches = crop_mouths(frames[:count], landmarks[:count],
                              np.load(MEAN_FACE, allow_pickle=False))
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp.npy")
        np.save(temporary, patches, allow_pickle=False)
        temporary.rename(destination)
        return {**record, "status": "ok", "frames": len(patches),
                "frame_count_mismatch": len(frames) - len(landmarks)}
    except (OSError, ValueError) as error:
        return {**record, "status": "failed", "error": str(error)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True,
                        help="JSONL manifests; rows without landmarks are skipped and counted")
    parser.add_argument("--data-root", type=Path, default=Path("datasets/lrs3"))
    parser.add_argument("--output", type=Path, default=Path("datasets/lrs3/autoavsr-crops"))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(filename)s:%(lineno)d %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(args.output / "crop.log"),
                                  logging.StreamHandler()], force=True)
    rows, skipped = [], 0
    for manifest in args.manifests:
        for line in manifest.read_text().splitlines():
            row = json.loads(line)
            if row.get("landmarks") and row.get("source_type") == "video":
                rows.append(row)
            else:
                skipped += 1
    rows = rows[:args.limit] if args.limit else rows
    LOGGER.info("Cropping %d clips with %d workers (%d rows without landmarks skipped); "
                "mean face sha256 %s", len(rows), args.workers, skipped,
                hashlib.sha256(MEAN_FACE.read_bytes()).hexdigest())
    started, counts = time.monotonic(), {"ok": 0, "exists": 0, "failed": 0}
    with ProcessPoolExecutor(args.workers) as pool, \
            (args.output / "index.jsonl").open("a") as index:
        jobs = ((row, str(args.data_root), str(args.output)) for row in rows)
        for done, record in enumerate(pool.map(_process, jobs, chunksize=8), 1):
            counts[record["status"]] += 1
            index.write(json.dumps(record) + "\n")
            if record["status"] == "failed":
                LOGGER.warning("%s: %s", record["clip_id"], record["error"])
            if done % 2000 == 0:
                LOGGER.info("%d/%d clips, %s, %.0fs", done, len(rows), counts,
                            time.monotonic() - started)
    LOGGER.info("Finished %d clips: %s in %.0fs", len(rows), counts, time.monotonic() - started)


if __name__ == "__main__":
    main()
