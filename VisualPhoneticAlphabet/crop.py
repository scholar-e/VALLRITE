"""Isolate a tracked face and approximate neck region without extra models.

Uses OpenCV's bundled frontal-face cascade. Crops are rectangular context
regions, not segmentation masks or anatomical neck measurements.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def overlap(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    intersection = max(0, min(a[0]+a[2], b[0]+b[2])-x) * max(0, min(a[1]+a[3], b[1]+b[3])-y)
    return intersection / (a[2]*a[3]+b[2]*b[3]-intersection)


class FaceNeckCropper:
    """Conservative single-face association with no stale-frame interpolation.

    Ambiguous initialization is rejected. A track may bridge up to five missed
    detections for association only; those frames still return no crop.
    """
    def __init__(self, smoothing=.65, max_gap=5):
        import cv2
        if not 0 < smoothing <= 1 or max_gap < 0:
            raise ValueError('invalid smoothing or max_gap')
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if self.detector.empty():
            raise RuntimeError('OpenCV bundled face cascade is unavailable')
        self.smoothing = smoothing
        self.max_gap = max_gap
        self.previous = None
        self.misses = 0
        self.segment = 0

    def select(self, boxes):
        boxes = [tuple(map(float, b)) for b in boxes]
        selected = None
        if self.previous is None:
            if len(boxes) == 1:
                selected = boxes[0]
                self.segment += 1
        else:
            matches = [b for b in boxes if overlap(b, self.previous) >= .2]
            if len(matches) == 1:
                selected = matches[0]
        if selected is None:
            self.misses += 1
            if self.misses > self.max_gap:
                self.previous = None
            return None
        # Never smooth across a missing observation.
        if self.previous is not None and self.misses == 0:
            selected = tuple(self.smoothing*b + (1-self.smoothing)*a for a, b in zip(self.previous, selected))
        self.previous = selected
        self.misses = 0
        return selected

    def process(self, pixels):
        import cv2
        gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
        scale = min(1., 960 / max(gray.shape))
        small = cv2.resize(gray, None, fx=scale, fy=scale) if scale < 1 else gray
        detections = self.detector.detectMultiScale(small, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
        boxes = [tuple(float(v)/scale for v in box) for box in detections]
        face = self.select(boxes)
        metadata = {'method': 'opencv-haar-face-neck-0.1', 'status': 'missing',
                    'detected_faces': len(boxes), 'face_xywh': None, 'crop_xywh': None,
                    'tracking_confidence': None, 'track_segment': self.segment}
        if face is None:
            return None, metadata
        x, y, w, h = face
        # Include forehead, ears, jaw, and approximate neck/upper shoulder context.
        left, top = math.floor(x-.3*w), math.floor(y-.25*h)
        right, bottom = math.ceil(x+1.3*w), math.ceil(y+1.85*h)
        height, width = pixels.shape[:2]
        x0, y0 = max(0, left), max(0, top)
        x1, y1 = min(width, right), min(height, bottom)
        metadata.update(status='observed', face_xywh=list(face),
                        crop_xywh=[x0, y0, x1-x0, y1-y0],
                        clipped=left < 0 or top < 0 or right > width or bottom > height)
        return pixels[y0:y1, x0:x1].copy(), metadata


def isolate(video, output, size=256, max_frames=None):
    """Write fixed-size silent crops plus original-timestamp JSONL sidecar."""
    import cv2
    import numpy as np
    from .__main__ import sha256
    if size < 32 or (max_frames is not None and max_frames < 1):
        raise ValueError('size must be >= 32 and max_frames positive')
    if size % 2:
        raise ValueError('size must be even for MP4 encoding')
    sidecar = output.with_suffix('.jsonl')
    if output.resolve() == video.resolve() or sidecar.resolve() == video.resolve():
        raise ValueError('output must not overwrite source')
    if output.exists() or sidecar.exists():
        raise ValueError('output or sidecar already exists; choose a new output path')
    cap = cv2.VideoCapture(str(video))
    writer = None
    try:
        if not cap.isOpened():
            raise ValueError(f'cannot open video: {video}')
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or not 0 < fps <= 1000:
            raise ValueError('invalid source frame rate')
        cropper = FaceNeckCropper()
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (size, size))
        if not writer.isOpened():
            raise RuntimeError('cannot create output video')
        provenance = {'source': str(video.resolve()), 'source_sha256': sha256(video),
                      'opencv_version': cv2.__version__, 'source_fps': fps,
                      'output_size': size, 'max_frames': max_frames,
                      'timing': 'CFR output; original PTS retained per frame', 'audio': 'omitted'}
        count, previous = 0, None
        with sidecar.open('w', encoding='utf-8') as stream:
            while max_frames is None or count < max_frames:
                ok, pixels = cap.read()
                if not ok:
                    break
                pts = cap.get(cv2.CAP_PROP_POS_MSEC)
                fallback = not math.isfinite(pts) or pts < 0 or (previous is not None and pts <= previous)
                if fallback:
                    pts = 0. if previous is None else previous+1000/fps
                previous = pts
                crop, metadata = cropper.process(pixels)
                canvas = np.zeros((size, size, 3), dtype=np.uint8)
                if crop is not None:
                    h, w = crop.shape[:2]
                    ratio = min(size/w, size/h)
                    rw, rh = max(1, round(w*ratio)), max(1, round(h*ratio))
                    resized = cv2.resize(crop, (rw, rh), interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR)
                    canvas[(size-rh)//2:(size-rh)//2+rh, (size-rw)//2:(size-rw)//2+rw] = resized
                writer.write(canvas)
                stream.write(json.dumps({'schema': 'face-neck-0.1', 'frame_index': count,
                    'time_ms': pts, 'timestamp_source': 'fps_fallback' if fallback else 'opencv_pts',
                    'crop': metadata, 'provenance': provenance if count == 0 else None}, allow_nan=False)+'\n')
                count += 1
        if count == 0:
            raise ValueError('source contains no decodable frames')
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--output', type=Path, required=True, help='new .mp4 crop video; also creates .jsonl sidecar')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--max-frames', type=int)
    args = parser.parse_args()
    if not args.video.is_file() or args.output.suffix.lower() != '.mp4':
        parser.error('input must exist and output must end in .mp4')
    try:
        count = isolate(args.video, args.output, args.size, args.max_frames)
        print(f'Wrote {count} frames to {args.output} and {args.output.with_suffix(".jsonl")}')
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        parser.exit(1, f'Face/neck isolation failed: {error}\n')


if __name__ == '__main__':
    main()
