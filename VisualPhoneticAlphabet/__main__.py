"""Extract VPA JSONL with an optional lightweight MediaPipe video backend."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from .core import build_record, observe, POINT_IDS


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def extract(video, model, max_frames=None, roi=None, face_neck=False):
    import cv2
    import mediapipe as mp
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f'cannot open video: {video}')
    from .crop import FaceNeckCropper
    cropper = FaceNeckCropper() if face_neck else None
    frames = []
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not 0 < fps <= 1000:
            raise ValueError('video must have a valid frame rate <= 1000')
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=2)
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
            while max_frames is None or len(frames) < max_frames:
                ok, pixels = cap.read()
                if not ok:
                    break
                source_height, source_width = pixels.shape[:2]
                if roi is not None:
                    x, y, width, height = roi
                    if min(x, y) < 0 or min(width, height) <= 0 or x+width > pixels.shape[1] or y+height > pixels.shape[0]:
                        raise ValueError('ROI must be a positive rectangle inside every frame')
                    pixels = pixels[y:y+height, x:x+width].copy()
                # OpenCV PTS is preferred; fallback is explicit in provenance.
                pts = cap.get(cv2.CAP_PROP_POS_MSEC)
                fallback = not math.isfinite(pts) or pts < 0 or bool(frames and pts <= frames[-1]['time_ms'])
                if fallback:
                    pts = len(frames)*1000/fps
                    if frames:
                        pts = max(pts, frames[-1]['time_ms']+1000/fps)
                timestamp = round(pts)
                if frames:
                    timestamp = max(timestamp, round(frames[-1]['time_ms'])+1)
                crop_metadata = None
                if cropper is not None:
                    pixels, crop_metadata = cropper.process(pixels)
                points, face_count, blur = None, 0, None
                if pixels is not None:
                    result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)), timestamp)
                    h, w = pixels.shape[:2]
                    face_count = len(result.face_landmarks)
                    points = [(p.x*w, p.y*h) for p in result.face_landmarks[0]] if face_count == 1 else None
                    blur = float(cv2.Laplacian(cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
                previous = frames[-1] if frames else None
                if crop_metadata and previous and previous['quality'].get('face_neck', {}).get('track_segment') != crop_metadata['track_segment']:
                    previous = None
                frame = observe(pts, points, previous)
                offset_x, offset_y = roi[:2] if roi is not None else (0, 0)
                if crop_metadata and crop_metadata.get('crop_xywh'):
                    offset_x += crop_metadata['crop_xywh'][0]
                    offset_y += crop_metadata['crop_xywh'][1]
                frame['quality']['source_size'] = [source_width, source_height]
                frame['quality']['source_landmarks'] = ({str(i): [points[i][0]+offset_x, points[i][1]+offset_y]
                    for i in POINT_IDS} if points is not None else None)
                frame['quality']['timestamp_source'] = 'fps_fallback' if fallback else 'opencv_pts'
                frame['quality']['face_count'] = face_count
                frame['quality']['blur_laplacian_variance'] = blur
                if crop_metadata is not None:
                    frame['quality']['face_neck'] = crop_metadata
                frames.append(frame)
    finally:
        cap.release()
    return build_record(video.stem, fps, frames, {'source': str(video.resolve()),
        'source_sha256': sha256(video), 'model_sha256': sha256(model),
        'extractor': 'mediapipe-vpa-0.1', 'mediapipe_version': mp.__version__,
        'opencv_version': cv2.__version__, 'max_frames': max_frames, 'roi_xywh': roi, 'face_neck': face_neck,
        'speaker_id': None, 'split': None, 'license': None})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--model', type=Path, required=True, help='local face_landmarker.task')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--roi', type=int, nargs=4, metavar=('X', 'Y', 'WIDTH', 'HEIGHT'), help='explicit speaker region in source pixels')
    parser.add_argument('--face-neck', action='store_true', help='automatically isolate tracked face and neck before landmark extraction')
    args = parser.parse_args()
    if not args.video.is_file() or not args.model.is_file():
        parser.error('video and model must be existing files')
    if args.max_frames is not None and args.max_frames < 1:
        parser.error('--max-frames must be positive')
    try:
        record = extract(args.video, args.model, args.max_frames, args.roi, args.face_neck)
        import jsonschema
        schema = json.loads(Path(__file__).with_name('schema.json').read_text())
        jsonschema.validate(record, schema)
        payload = json.dumps(record, allow_nan=False)+'\n'
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding='utf-8')
    except (ValueError, ImportError, RuntimeError, OSError) as error:
        parser.exit(1, f'VPA extraction failed: {error}\n')


if __name__ == '__main__':
    main()
