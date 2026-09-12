# Visual Phonetic Alphabet

This directory defines the plan for a Visual Phonetic Alphabet (VPA): a
machine-readable description of visible speech articulation. It complements
IPA/ARPAbet rather than replacing them.

The key distinction is:

- IPA describes speech sounds and how they are articulated.
- VPA describes evidence that a camera can actually observe.
- A probabilistic mapping connects VPA gestures to possible phonemes.

VPA must not claim to observe voicing, nasality, or hidden tongue position
when those properties are not visible. For example, a bilabial closure can
support `/p/`, `/b/`, and `/m/` with different probabilities without forcing
an arbitrary early decision.

## Intended pipeline

```text
face video
  -> tracked and stabilized mouth/lower-face regions
  -> learned video, landmark, and motion encoders
  -> continuous visible measurements + gesture events
  -> calibrated IPA/ARPAbet lattice
  -> pronunciation dictionary/WFST
  -> compact neural or Qwen reranker
  -> transcript
```

The learned video representation remains the primary signal. Explicit VPA
measurements provide auxiliary supervision, interpretability, debugging, and
a stable interchange format; they are not intended to compress the entire
video into a handful of handcrafted scalars.

## Design principles

1. **Observation before inference.** Store visible evidence separately from
   inferred phonemes.
2. **Continuous before categorical.** Retain geometry and motion values before
   converting them into gesture tokens.
3. **Temporal by construction.** Model transitions, closures, holds, and
   releases rather than isolated mouth shapes.
4. **Uncertainty is data.** Preserve ranked candidates, confidence, and
   unobservable properties.
5. **Speaker independence.** Normalize pose and scale, and test only on
   speaker-disjoint splits.
6. **Evidence-driven inventory.** A proposed feature or gesture stays in VPA
   only if held-out ablations show useful information.
7. **Backward compatibility.** Maintain mappings to the 39-class ARPAbet
   inventory used by the existing VALLR checkpoint.

## Version 0 feature inventory

The initial inventory is deliberately broader than five simple lip measures,
but small enough for controlled ablations.

### Geometry

- Inner- and outer-lip aperture
- Mouth width and width-to-height ratio
- Upper- and lower-lip curvature
- Upper/lower-lip relative displacement
- Jaw displacement

### Contact and appearance

- Complete and partial bilabial closure
- Lip compression
- Lower-lip/upper-teeth contact
- Teeth visibility
- Oral-cavity visibility/darkness
- Tongue visibility, marked unknown when occluded

### Motion and duration

- Horizontal and vertical optical flow
- Lip-opening and lip-closing velocity
- Lip-corner and jaw velocity
- Closure, hold, and release duration
- Release velocity
- Anticipatory and carry-over motion

### Reliability and nuisance variables

- Landmark/tracking confidence
- Head yaw, pitch, and roll
- Motion blur and image quality
- Occlusion
- Speaking/non-speaking probability

Reliability and pose fields are not speech symbols. They allow downstream
models to down-weight uncertain evidence and prevent nuisance variation from
being mistaken for articulation.

## Provisional gesture tokens

Gesture tokens summarize intervals of continuous observations. Their names and
thresholds remain provisional until validated.

| Token | Meaning |
|---|---|
| `BCL` | complete bilabial closure |
| `BCL-PART` | partial bilabial closure |
| `BCL-REL` | release from bilabial closure |
| `LFD` | lower-lip/upper-teeth contact |
| `RND` | lip rounding |
| `PRO` | forward lip protrusion |
| `SPR` | lateral lip spreading |
| `OPEN` | mouth/jaw opening |
| `TNG` | visible tongue gesture |
| `HOLD` | sustained articulation |
| `TRANS` | transition without a stable target |
| `UNK-VIS` | hidden, occluded, or low-confidence evidence |

Tokens may carry intensity, phase, and duration, such as `RND:0.82`,
`BCL:HOLD:74ms`, or `BCL-REL:FAST`. VPA tokens are multi-label: rounding and
opening can occur simultaneously.

## Record schema

The first schema should be JSONL so records are inspectable and streamable.
Every clip record will contain provenance, timestamps, continuous features,
gesture intervals, aligned teacher labels, and phoneme hypotheses.

```json
{
  "schema": "vpa-0.1",
  "clip_id": "avspeech:1:H1ulMfj5wRY:112.320-116.940",
  "source_fps": 50.0,
  "frames": [
    {
      "time_ms": 640,
      "features": {
        "inner_aperture": 0.31,
        "mouth_width": 0.68,
        "rounding": 0.82,
        "bilabial_contact": 0.04
      },
      "quality": {
        "tracking_confidence": 0.94,
        "occluded": false,
        "head_yaw_degrees": 4.2
      }
    }
  ],
  "gestures": [
    {
      "token": "BCL",
      "start_ms": 420,
      "end_ms": 510,
      "confidence": 0.93,
      "phoneme_candidates": [
        {"ipa": "p", "arpabet": "P", "probability": 0.40},
        {"ipa": "b", "arpabet": "B", "probability": 0.35},
        {"ipa": "m", "arpabet": "M", "probability": 0.25}
      ],
      "unobservable": ["voicing", "nasality"]
    }
  ]
}
```

Probability fields must identify whether they are raw, temperature-calibrated,
or normalized within a pruned candidate set. Schema versions must never change
the meaning of an existing field.

## Data and supervision plan

### Video preparation

1. Track the face across the complete clip.
2. Normalize rigid head motion without removing non-rigid speech motion.
3. Produce synchronized mouth-only and lower/full-face crops.
4. Preserve original timestamps and frame rate.
5. Extract dense 2D/3D landmarks and local motion fields.
6. Mark—not silently discard—blurred, occluded, and failed frames.

### Training-only audio teacher

1. Obtain the reference transcript where licensing permits.
2. Convert it to stress-preserving ARPAbet and IPA alternatives.
3. Run an audio phoneme recognizer and forced aligner.
4. Retain teacher posterior probabilities and timestamps.
5. Compare audio labels with transcript-derived pronunciations.
6. Down-weight or reject clips with poor synchronization or disagreement.

Audio is supervision, not an inference requirement. Train/validation/test
splits must be fixed before teacher generation to prevent leakage.

### Gesture discovery

Use two complementary paths:

- Train interpretable auxiliary heads for proposed measurements and contacts.
- Cluster self-supervised motion embeddings to discover recurring visual units
  not represented by the provisional inventory.

A discovered cluster is promoted to a named VPA gesture only when it is stable
across speakers and improves phoneme recall or transcript WER. Clusters that
primarily identify faces, poses, or recording conditions are rejected.

## Model plan

The visual model will use a shared spatiotemporal encoder with multiple heads:

```text
shared video encoder
  + landmark/motion fusion
      |- CTC phoneme posterior head
      |- continuous VPA feature head
      |- gesture event head
      |- boundary/duration head
      `- audio-teacher distillation head (training only)
```

A starting loss is:

```text
L = lambda_ctc * L_ctc
  + lambda_feature * L_feature
  + lambda_gesture * L_gesture
  + lambda_boundary * L_boundary
  + lambda_teacher * L_teacher
  + lambda_temporal * L_temporal_consistency
```

Weights will be selected on validation WER and calibration, not merely on the
individual auxiliary losses.

## Decoder integration

VALLRITE currently emits a greedy phoneme sequence and per-step top-k values.
VPA integration should replace that intermediate format with a proper CTC
prefix beam or lattice containing timing and cumulative log probabilities.

The first-pass decoder should combine:

```text
visual/gesture lattice x pronunciation lexicon x word language model
```

Candidate scores should remain explicit:

```text
score = visual_logp
      + alpha * pronunciation_logp
      + beta * word_language_logp
      + gamma * insertion_penalty
```

A compact lattice-aware Transformer and the Qwen adapter will then be compared
as optional second-pass rerankers. Qwen must not receive probabilities only as
unstructured prose, and it must not be allowed to overwhelm visual evidence
without a measured WER improvement.

## Experimental program

### Baselines

- Existing VALLR greedy phoneme decoding
- CTC prefix beam without VPA features
- CTC beam plus pronunciation dictionary
- Dictionary/WFST plus n-gram language model

### Feature ablations

Add each family independently and cumulatively:

1. Learned RGB video features only
2. Landmark trajectories
3. Geometry
4. Optical flow and motion derivatives
5. Contact/appearance features
6. Jaw and cheek context
7. Protrusion and sparse tongue visibility
8. Hand-designed plus discovered gesture tokens

### Required metrics

| Layer | Metrics |
|---|---|
| Tracking | failure rate, landmark error, usable-frame rate |
| VPA features | regression/classification error by speaker and pose |
| Gestures | stability, boundary error, phonetic mutual information |
| Phonemes | PER, top-k PER, oracle lattice recall |
| Probabilities | expected calibration error, Brier score |
| Lexical decoder | oracle lattice WER, search error |
| End to end | WER/CER by speaker, pose, accent, and quality |
| Deployment | latency, peak RAM, model size, energy |

Every result must use speaker-disjoint evaluation. A feature is removed when it
only improves training performance, primarily encodes speaker identity, or
fails to improve held-out phoneme recall or WER at a reasonable compute cost.

## Delivery milestones

### M0: Specification

- Freeze `vpa-0.1` terminology and JSON schema.
- Document normalization, units, coordinate systems, and missing values.
- Define ARPAbet/IPA/viseme compatibility tables.

### M1: Observable-feature extractor

- Stabilized mouth and lower-face crops.
- Landmark, geometry, quality, and temporal-motion extraction.
- Visual inspection and distribution reports on AVSpeech samples.

### M2: Supervised alignment

- Audio teacher and forced-alignment pipeline.
- Timestamped transcript, audio, and visual training records.
- Synchronization and label-quality filters.

### M3: Multi-task visual model

- Auxiliary feature, gesture, and boundary heads.
- Speaker-disjoint training and ablations.
- Calibrated phoneme probabilities.

### M4: VPA lattice decoder

- CTC prefix beam/lattice implementation.
- Pronunciation alternatives and dictionary/WFST integration.
- Oracle-recall and search-error evaluation.

### M5: Contextual reranking

- Compact lattice-aware Transformer baseline.
- Uncertainty-aware Qwen comparison.
- Conservative decoding that preserves source fidelity.

### M6: Mobile prototype

- Quantized feature extractor and selected reranker.
- End-to-end benchmarks on a named reference phone.
- Documented accuracy, memory, latency, thermal, and battery tradeoffs.

## MVP acceptance criteria

The first useful release does not require the entire alphabet. It must:

1. Reproducibly extract aperture, width, rounding, closure, landmark motion,
   and confidence from unseen videos.
2. Emit timestamped VPA JSONL conforming to the versioned schema.
3. Preserve top-k phoneme candidates and calibrated probabilities.
4. Improve either oracle top-k phoneme recall or end-to-end WER over the
   existing CTC beam baseline on held-out speakers.
5. Demonstrate through ablations that the gain comes from visible-articulation
   information rather than speaker identity or additional parameter count.

If criterion 4 is not met, the feature inventory must be revised before the
project advances to a larger alphabet or mobile optimization.


## Running the lightweight observable-feature baseline

Implemented in `core.py` and `__main__.py`. This is an M0/partial-M1 baseline,
not completion of the research MVP acceptance criteria above. It runs separately
from the learned VALLR encoder and never substitutes geometry for its predictions.
No audio, LLM, training corpus download, or VALLR checkpoint is needed.

From the repository root (Python 3.11+):

```bash
python -m venv .venv-vpa
.venv-vpa-gpu/bin/pip install -r VisualPhoneticAlphabet/requirements.txt
curl -L --fail -o /tmp/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
.venv-vpa-gpu/bin/python -m VisualPhoneticAlphabet \
  datasets/avspeech/clips/0000001_H1ulMfj5wRY_112.320_116.940.mp4 \
  --model /tmp/face_landmarker.task --output /tmp/vpa.jsonl --max-frames 100 \
  --roi 0 100 350 400
python -m unittest discover -s VisualPhoneticAlphabet/tests -v
```

The local model is approximately 3.7 MB. See Google's
[Face Landmarker documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python)
for its API and model terms. Downloads are explicit; inference is offline.
The CLI overwrites its output with one schema-validated clip record. Omit
`--max-frames` for the full clip. Memory scales with clip duration; use short clips
for this initial build. Original media is never modified.

### Frozen baseline conventions

- `schema.json` is the executable `vpa-0.1` baseline contract. Empty teacher and
  phoneme arrays deliberately reserve future interfaces without pretending that
  this extractor supplies trained or calibrated predictions. Extending them
  requires a versioned schema update. The illustrative probabilities earlier in
  this specification are not implemented predictions.
- Pixel landmarks follow MediaPipe indices. Stored selected landmarks are 2D,
  centered on the midpoint of outer eye corners 33/263, rotated to align those
  corners, and divided by their distance. Positive y points down. This removes
  translation, scale, and roll; it does not correct yaw/pitch or export crops.
- Apertures and width are Euclidean distances in eye-distance units. Inner
  aperture uses 13/14, outer aperture 0/17, width 61/291. Width-to-height is null
  at zero aperture. Rounding is `min(1, aperture/width)`, a shape proxy, not
  evidence of protrusion. Contact is `max(0, 1 - (aperture/width)/0.08)`, a
  provisional closure proxy, not a measured physical contact probability.
- Opening velocity and mean selected-landmark speed use units per second;
  derivatives are null on the first frame and after every missing frame.
- Times are source milliseconds from OpenCV PTS with an explicitly flagged FPS
  fallback when timestamps fail to advance. Detector timestamps are rounded to
  strictly increasing integer milliseconds. Intervals are half-open; the final
  interval ends one nominal frame after the last observation.
- `BCL` uses contact >= 0.75; `RND` uses rounding >= 0.45; `OPEN` uses opening
  velocity > 0.15. `BCL-REL` follows a closure only into observed evidence.
  `UNK-VIS` covers missing or multiple-face frames. Tokens can overlap. These
  thresholds have not been tuned or validated; short detection jitter can cause
  short events. `compatible_arpabet` lists P/B/M without assigning probabilities.
- Confidence, occlusion, yaw, and pitch are null because this backend does not
  establish them. Face presence is not calibrated tracking confidence.
  Full-frame Laplacian variance is a raw blur diagnostic, not mouth visibility.
  Speaking activity, tongue/teeth contact, optical flow, and crop stabilization
  are not yet implemented. Multiple faces are conservatively marked missing;
  active-speaker selection and identity continuity remain future work.
- `ARPABET_IPA` in `core.py` contains all 39 checkpoint phonemes. It is a broad
  English stress-free compatibility mapping, not an accent/pronunciation model.
  CTC blank is not silence. No synthetic silence, unknown phoneme, or occlusion
  phoneme is introduced; missing visual evidence is explicitly `UNK-VIS`.
- Source/model SHA-256 hashes, dependency versions, timestamp fallback, and
  frame limits are recorded. Speaker/split/license remain null until supplied by
  a corpus manifest; do not use these records for evaluation without those data.

Still required before claiming the research MVP: stable crops and tracking,
trained auxiliary heads, retained/calibrated learned phoneme distributions,
CTC beam/lattice integration, and speaker-disjoint recall/WER ablations.

### First smoke test

Tested with MediaPipe 0.10.35 and OpenCV 4.14.0 on the first 100 frames
(50 FPS, 0–1980 ms) of the existing AVSpeech sample above. Full-frame detection
returned no faces; the explicit `--roi 0 100 350 400` speaker region returned
observations for all 100 frames and 31 provisional gesture intervals. Both
outputs passed schema validation. The ROI is a manually supplied test condition,
not automatic speaker tracking; it is recorded in provenance. Coordinates are
relative to that region before eye normalization. This checks execution and
missing-data behavior, not articulation accuracy, calibration, or WER.

The four deterministic unit tests cover rigid-transform invariance, missing-frame
motion resets, closure/release behavior, schema conformance, and malformed input.
Model SHA-256 used:
`64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.

## Automatic face and neck isolation

`crop.py` provides a reusable `FaceNeckCropper` and standalone video tool.
It uses the frontal-face cascade bundled with OpenCV, with no extra model or
package download. The rectangular region includes ears/forehead and extends
below the jaw toward the neck and upper shoulders. It is not segmentation:
background inside the rectangle remains, and neck coverage is approximate.

Use it directly before VPA landmark extraction, retaining original timestamps:

```bash
.venv-vpa-gpu/bin/python -m VisualPhoneticAlphabet \
  datasets/avspeech/clips/0000001_H1ulMfj5wRY_112.320_116.940.mp4 \
  --model /tmp/face_landmarker.task --output /tmp/vpa-auto.jsonl \
  --face-neck --max-frames 100
```

Or export an isolated silent video for other processing:

```bash
.venv-vpa-gpu/bin/python -m VisualPhoneticAlphabet.crop \
  datasets/avspeech/clips/0000001_H1ulMfj5wRY_112.320_116.940.mp4 \
  --output /tmp/face-neck.mp4 --size 256 --max-frames 100
```

The output path must be new. The tool also writes `/tmp/face-neck.jsonl`, with
source timestamps, face/crop rectangles, clipping, detection count, track segment,
missing status, and provenance. Crops are aspect-preserving and letterboxed.
Missing detections produce black frames rather than repeating a previous face.
The MP4 uses the source nominal FPS; for variable-frame-rate inputs, the sidecar
is authoritative for original timing. Use direct `--face-neck` integration when
VPA must retain source PTS. No audio is copied.

A new track requires exactly one detected face. Subsequent detections must have
an unambiguous overlap with the preceding face. Crop boxes are smoothed only
across consecutive observations; missed detections never generate stale crops.
Association resets after six misses. A new segment is not a verified speaker
identity: this lightweight tracker cannot resolve crossing faces, shot changes,
or active speakers. Profile faces, occlusion, and very small faces may be missed.
Optional `--roi X Y WIDTH HEIGHT` limits the search before `--face-neck`; recorded
crop coordinates are relative to that ROI when used together. Crop failures flow
through VPA as missing observations and reset motion derivatives.

Validation: the existing 100-frame AVSpeech sample produced 100 detected crops
and 100 VPA landmark observations automatically (versus zero landmark observations
on the original wide frames). The exported video was verified at 256×256,
100 frames, 50 FPS; its first frame was visually inspected. This is a processing
smoke test, not recognition accuracy evidence. Six additional tests cover box
association, ambiguity, gaps, unrelated faces, boundary clipping, and missing
crops. Run the complete suite with the optional environment:

```bash
.venv-vpa-gpu/bin/python -m unittest discover -s VisualPhoneticAlphabet/tests -v
```

## Phoneme accuracy diagnostic

The subsequent two-clip GRID smoke test found **90.625% transcript-derived
phoneme error rate** for both original and face/neck inputs with the current
VALLR CLI. Cropping retained only 37.33% of frames on one example. The current
8-step output is insufficient for the 16-phone reference sentences; VPA itself
still emits no learned phoneme predictions. See the
[evaluation report](evaluation/README.md) for predictions, caveats, source
transcripts, and reproduction commands. The earlier detection-only smoke tests
must not be interpreted as phoneme accuracy results.


## Basic browser visualization

Open `web/index.html` directly in a browser. No server, package installation,
network access, or build step is required. Alternatively, from the repository
root run `python -m http.server 8000` and visit
`http://localhost:8000/VisualPhoneticAlphabet/web/`.

The inspector includes a real 100-frame AVSpeech ROI smoke-test export (the
manual ROI sample described above), stored in `web/sample.js`. It displays
normalized selected mouth/chin landmarks, frame measurements, feature traces,
and the five currently implemented gesture types. The original source video appears
behind a translucent frame-data overlay, with playback and seeking synchronized
to frame timestamps. Toggle “Show frame overlay” for an unobstructed view, or
expand “Normalized mouth landmarks” inside the overlay to inspect the geometry. For
imported clips, choose the matching local video; switching clips clears that selection.
Outside the exported data range, the nearest recorded frame remains visible.
Play or scrub the recording,
select a gesture interval to jump to it, or load your own CLI `.jsonl` export.
Multiple clip records can be selected with the Clip menu. Files are read locally
in the browser, with a 20 MB limit. The viewer checks basic record structure;
the CLI remains responsible for full schema validation.

The landmark view uses fixed bounds across the loaded clip and connects the
sparse measured points, not a dense anatomical contour. Missing frames and
unknown values remain explicit. This page visualizes exported results; it does
not run video extraction, assign phoneme probabilities, or generate transcripts.


### Video-position mouth overlay

New CLI exports retain `quality.source_size` as `[width, height]` and
`quality.source_landmarks` as MediaPipe-indexed `[x, y]` coordinates in original
video pixels. ROI and face/neck crop offsets are included. These optional quality
fields supplement the existing eye-normalized landmarks without changing them.
Missing detections store null source landmarks.

The browser draws the six selected lip points directly over the original video,
accounting for resize and letterboxing. “Show mouth points” toggles them. The
bundled 100-frame sample has been regenerated with these coordinates. Older
exports must be re-extracted to display video-position points; normalized points
alone cannot recover their original positions. Points are hidden beyond the
exported time range and on missing frames. Use the original matching video,
not a separately cropped or time-shifted version.


The viewer also provides single-frame back/forward controls, 0.25× and 0.5×
playback, and adjustable point size. Supported browsers synchronize overlays to
displayed video-frame timestamps; other browsers use the video playback clock.
These controls change visualization only; extracted coordinates and gesture
thresholds remain unchanged.

### Expanded lip contours and exploratory shape measurements

The extractor now retains 40 lip points (20 outer and 20 inner), plus the
existing chin point, in both normalized and source coordinates. Contour ordering
follows [MediaPipe FACEMESH_LIPS](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py).
The original seven-point `landmark_speed`, all eight baseline feature definitions,
and gesture thresholds remain unchanged. The existing schema already permits
additional indexed landmarks; legacy records continue to load.

The browser calculates seven additional measurements from the normalized points
in `web/shape.js`. These are derived for inspection, not added to the JSONL
`features` contract:

| Measurement | Definition | Potential distinction to investigate |
|---|---|---|
| Opening area | Absolute shoelace area of the inner contour, eye-distance² | Overall opening beyond a single center gap |
| Opening area / width² | Area divided by squared 61–291 distance | Opening shape relative to mouth width |
| Side gap toward 61 | Euclidean distance 81–178 | Opening on one side of the mouth |
| Side gap toward 291 | Euclidean distance 311–402 | Opening on the other side |
| Side-gap asymmetry / width | Absolute difference of side gaps / mouth width | Uneven opening, also sensitive to pose |
| Upper lip thickness (2D) | Distance 0–13, eye-distance units | Projected upper-lip shape |
| Lower lip thickness (2D) | Distance 14–17, eye-distance units | Projected lower-lip shape |

These are exploratory 2D geometry measurements, not established phoneme
discriminators or evidence of physical contact, teeth, tongue, or protrusion.
Pose, occlusion, resolution, and landmark error can affect them. Missing required
points produce unknown values, and width-normalized ratios are unknown at zero
width. Discrimination gains need speaker-disjoint ablation/evaluation.

The regenerated sample includes all 40 lip points for its 100 observed frames.
Green denotes the outer contour; amber denotes the inner contour. Old exports
need re-extraction for the full contours and area/side-gap measurements.

Check derived geometry with:

```bash
node VisualPhoneticAlphabet/tests/test_shape.cjs
```

### GRID pilot download

`python -m VisualPhoneticAlphabet.download_grid` downloads speakers 1–12
from the [official GRID corpus](https://spandh.dcs.shef.ac.uk/gridcorpus/), for
research use. It stores normal-quality 360×288 video, original 50 kHz audio,
and word alignments under `datasets/grid-pilot/`. This pilot prioritizes a
manageable download; the subtle lip contours may warrant high-resolution video
in a follow-up experiment.

The downloader resumes partial transfers, checks ZIP CRCs, safely extracts
archives, and records SHA-256 hashes and source URLs in
`download-manifest.json`. The fixed speaker split is 1–8 train, 9–10 validation,
and 11–12 test. It is a pilot split, not an official benchmark protocol or a
claim of demographic balance. Phoneme timings have not yet been generated;
word labels must not be treated as frame-level phoneme supervision. Retain the
original audio/video timing when aligning. Downloaded media stay git-ignored.
