"""Dependency-free geometry and provisional temporal gestures."""
from __future__ import annotations

import math

ARPABET_IPA = dict(zip(
    'AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH'.split(),
    'ɑ æ ʌ ɔ aʊ aɪ b tʃ d ð ɛ ɝ eɪ f ɡ h ɪ i dʒ k l m n ŋ oʊ ɔɪ p ɹ s ʃ t θ ʊ u v w j z ʒ'.split(), strict=True))
FEATURES = ('inner_aperture', 'outer_aperture', 'mouth_width', 'width_to_height',
            'rounding', 'bilabial_contact', 'opening_velocity', 'landmark_speed')
# Ordered contours follow MediaPipe FACEMESH_LIPS (Apache-2.0):
# https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py
BASELINE_POINT_IDS = (61, 291, 13, 14, 0, 17, 152)
OUTER_LIP_IDS = (61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
                 291, 375, 321, 405, 314, 17, 84, 181, 91, 146)
INNER_LIP_IDS = (78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
                 308, 324, 318, 402, 317, 14, 87, 178, 88, 95)
POINT_IDS = tuple(dict.fromkeys((*BASELINE_POINT_IDS, *OUTER_LIP_IDS, *INNER_LIP_IDS)))


def observe(time_ms, landmarks, previous=None):
    """Landmarks are MediaPipe-indexed pixel (x,y) pairs, never normalized x/y.

    Eye corners remove translation, scale and roll only. Missing evidence resets
    derivatives. Rounding/contact are geometric proxies, not probabilities.
    """
    if not math.isfinite(time_ms) or time_ms < 0:
        raise ValueError('time_ms must be finite and nonnegative')
    frame = {'time_ms': time_ms, 'features': dict.fromkeys(FEATURES),
             'quality': {'tracking_confidence': None, 'occluded': None,
                         'head_yaw_degrees': None, 'head_pitch_degrees': None,
                         'head_roll_degrees': None, 'status': 'missing'},
             'landmarks': None}
    if landmarks is None:
        return frame
    if len(landmarks) < 468 or any(len(p) != 2 or not all(math.isfinite(v) for v in p) for p in landmarks):
        raise ValueError('expected at least 468 finite pixel x/y landmarks')
    left, right = landmarks[33], landmarks[263]
    dx, dy = right[0] - left[0], right[1] - left[1]
    scale = math.hypot(dx, dy)
    if scale < 1e-6:
        return frame
    origin = ((left[0] + right[0])/2, (left[1] + right[1])/2)
    def normalized(p):
        x, y = p[0]-origin[0], p[1]-origin[1]
        return [(x*dx+y*dy)/scale**2, (-x*dy+y*dx)/scale**2]
    points = {str(i): normalized(landmarks[i]) for i in POINT_IDS}
    def distance(a, b):
        return math.dist(points[str(a)], points[str(b)])
    aperture, width = distance(13, 14), distance(61, 291)
    ratio = aperture / width if width > 1e-6 else 0
    frame['features'].update(inner_aperture=aperture, outer_aperture=distance(0, 17),
                             mouth_width=width, width_to_height=width/aperture if aperture > 1e-6 else None,
                             rounding=min(1., ratio), bilabial_contact=max(0., 1-ratio/.08))
    frame['landmarks'] = points
    frame['quality'].update(status='observed', head_roll_degrees=math.degrees(math.atan2(dy, dx)))
    if previous and previous['landmarks'] is not None:
        dt = (time_ms-previous['time_ms'])/1000
        if dt <= 0:
            raise ValueError('timestamps must increase')
        frame['features']['opening_velocity'] = (aperture-previous['features']['inner_aperture'])/dt
        frame['features']['landmark_speed'] = sum(math.dist(points[str(i)], previous['landmarks'][str(i)])
            for i in BASELINE_POINT_IDS)/len(BASELINE_POINT_IDS)/dt
    return frame


def build_record(clip_id, fps, frames, provenance=None):
    if not math.isfinite(fps) or fps <= 0 or not frames:
        raise ValueError('positive source_fps and nonempty frames required')
    if any(b['time_ms'] <= a['time_ms'] for a, b in zip(frames, frames[1:])):
        raise ValueError('timestamps must increase')
    gestures, active = [], {}
    for index, frame in enumerate(frames):
        f = frame['features']
        tokens = set()
        if frame['quality']['status'] != 'observed':
            tokens.add('UNK-VIS')
        else:
            if f['bilabial_contact'] >= .75:
                tokens.add('BCL')
            if f['rounding'] >= .45:
                tokens.add('RND')
            if f['opening_velocity'] is not None and f['opening_velocity'] > .15:
                tokens.add('OPEN')
        for token in list(active):
            if token not in tokens:
                gestures.append(_gesture(token, active.pop(token), frame['time_ms']))
                if token == 'BCL' and frame['quality']['status'] == 'observed':
                    end = frames[index+1]['time_ms'] if index+1 < len(frames) else frame['time_ms']+1000/fps
                    gestures.append(_gesture('BCL-REL', frame['time_ms'], end))
        for token in tokens:
            active.setdefault(token, frame['time_ms'])
    end = frames[-1]['time_ms']+1000/fps
    gestures.extend(_gesture(t, s, end) for t, s in active.items())
    return {'schema': 'vpa-0.1', 'vocabulary_version': 'arpabet-ipa-en-0.1',
            'clip_id': clip_id, 'source_fps': fps, 'provenance': provenance or {},
            'frames': frames, 'gestures': sorted(gestures, key=lambda g: (g['start_ms'], g['token'])),
            'teacher_labels': [], 'phoneme_hypotheses': [],
            'summary': {'frame_count': len(frames), 'observed_fraction': sum(f['quality']['status']=='observed' for f in frames)/len(frames)}}


def _gesture(token, start, end):
    return {'token': token, 'start_ms': start, 'end_ms': end, 'confidence': None,
            'method': 'geometry-thresholds-0.1', 'phoneme_candidates': [],
            'compatible_arpabet': ['P', 'B', 'M'] if token in ('BCL', 'BCL-REL') else [],
            'unobservable': ['voicing', 'nasality', 'hidden_tongue_position']}
