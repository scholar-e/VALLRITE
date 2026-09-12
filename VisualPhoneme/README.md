# Compact visual phoneme recognizer

This branch trains a small visual-only network directly on face video. A shared
2D CNN recognizes each grayscale frame, a dilated temporal convolution stack
models motion and coarticulation, and a CTC head predicts an ARPAbet phoneme
sequence. Audio is used only to create training targets; inference uses video.

An optional fusion path embeds the 41 eye-normalized VPA coordinates, including
40 lip-contour points and the chin, with an explicit per-frame visibility bit.
The coordinate vector joins the image embedding before temporal modeling.
Missing coordinates are zero-filled only after the visibility mask is computed,
and their embedding is suppressed. A second, gated fusion design starts from a
trained coordinate model and learns a small image correction rather than forcing
the two modalities to contribute equally.

The default model has 123,096 parameters and retains one output step per
video frame. It therefore avoids the eight-step capacity limit in the current
VALLR checkpoint. This is an independent baseline, not a replacement for VALLR
until speaker-disjoint phoneme error rate supports that decision.

## Data protocol

- GRID speakers 1–8: training
- GRID speakers 9–10: validation and early stopping
- GRID speakers 11–12: locked test set
- Targets: stress-stripped phoneme sequences from the existing MFA alignment
- Default input: centered GRID mouth region from the face cutout, grayscale,
  letterboxed to 96×96
- Loss: CTC, so exact frame boundaries are not used during training
- Primary metric: sequence phoneme error rate (PER)

Before training, the loader compares each target's minimum CTC length with the
frame count observed during landmark extraction. It records and excludes clips
that cannot be decoded completely. The current corpus has 105 such training
clips, all from speaker 8; none occur in validation.

The fixed fractional crop is valid only for GRID's centered camera. Other data
must use a tracked face crop with explicit missing-frame metadata. Neither model
sees audio, transcripts, or speaker IDs at inference. The fusion model does use
coordinates extracted from the silent video.

## Methodology

### Research question

Test whether a lightweight image-and-time model can recover phoneme sequences
from silent video, and whether explicit facial geometry adds useful information
beyond mouth pixels. Change one input or model component at a time and select it
using validation speakers only.

### Target construction

MFA aligns each GRID recording's audio and transcript. Training takes the
ordered ARPAbet intervals, removes stress digits and silence labels, and uses
the resulting phoneme sequence as its target. CTC learns the video-to-phoneme
timing; forced frame boundaries are not supplied to the loss. Audio and MFA are
training-time supervision only.

Reject a clip when its decoded video is empty or shorter than the minimum CTC
path, including blank steps required between repeated adjacent phonemes. Record
every exclusion. Do not silently repair, repeat, or pad corrupt source frames.

### Visual inputs

Decode video at its native 25 FPS. For the GRID experiment, take the fixed
centered mouth region, convert it to grayscale, preserve its aspect ratio by
letterboxing, and resize it to 96×96. Training applies clip-consistent brightness
and contrast variation. A tracked crop must replace the fixed rectangle on
uncontrolled video.

The fusion condition also loads 41 VPA points at each frame: 40 inner/outer lip
points plus the chin. Coordinates are centered, scaled, and roll-normalized by
the eye corners during VPA extraction. Preserve the point order. Mark a frame
observed only when all coordinates are finite; replace missing numeric values
with zero only after creating that mask.

The `motion` representation concatenates each centered position with first- and
second-order differences at lag 1 and displacement differences at lags 2 and 4.
This gives ten values per point and makes short and longer articulation dynamics
explicit while preserving the same 25 FPS temporal grid.

### Model and fusion

The image branch applies four stride-two 2D convolution blocks independently to
each frame and produces a 96-value image embedding. The coordinate branch
flattens 41 two-dimensional points, appends visibility, and maps the result to a
32-value embedding. Its output is suppressed on missing frames. Concatenate the
image and coordinate embeddings and project them back to 96 values.

Four residual temporal convolutions with dilations 1, 2, 4, and 8 then cover
about 61 frames, or 2.44 seconds at 25 FPS. A linear CTC head emits blank plus
39 stress-free ARPAbet classes at every frame. Greedy inference collapses
repeated classes and removes blank. The image-only model has 123,096 parameters;
the original fusion model has 139,480. The regularized motion-coordinate model
has 60,808 parameters, and its gated image-residual extension has 149,209.

Gated fusion copies the coordinate encoder, temporal stack, and classifier from
the selected coordinate checkpoint. A learned scalar gate, initialized near
zero, scales a 96-value image residual before the shared temporal stack. Image
and coordinate modality dropout prevent either path from assuming the other is
always present. This preserves the coordinate baseline at initialization and
lets validation determine whether pixels provide an additive correction.

### Training and selection

Train with AdamW, gradient clipping, and CTC loss. Use the same random seed,
crop, image size, optimizer settings, and speaker split when comparing image
and fusion conditions. The current regularized runs use coordinate jitter,
point dropout, contiguous frame-span dropout, a 32-value landmark bottleneck,
and stronger weight decay. Stop after the declared patience or wall-clock
budget. Inspect training PER for overfitting, but never choose a model from
training PER.

The present image baseline used horizontal reflection while fusion disables it
to avoid mismatching left/right landmark identities. That makes the first fusion
comparison exploratory. A confirmatory comparison must either disable reflection
for both models or implement the correct symmetric landmark permutation.

The controlled ablation CLI uses `--architecture image`, `coordinates`, or
`fusion`. Use `--no-horizontal-flip` for comparable image runs.
`--coordinate-mode clip-centered` subtracts the per-clip median geometry to
remove static face shape; `constant` is a duration/grammar control. These modes
are stored in checkpoints and automatically reproduced during inference.

For repeated image experiments, `--frame-cache-dir` stores decoded, cropped
uint8 frames. The first pass populates the cache and later epochs avoid MPEG
decoding. The current GRID train/validation cache uses about 6.4 GiB.

### Probabilistic Top-N selection

CTC training still minimizes sequence negative log likelihood; exact Top-N
membership is discrete and cannot be used directly as a gradient loss. During
validation, `--top-n 5 --beam-width 16 --beam-token-top-k 8` runs a pruned CTC
prefix beam and measures both complete-reference recall in the five candidates
and oracle PER@5, the lowest edit rate among them. New runs select the lowest
oracle PER@N, with exact Top-N recall as the tie-breaker, because complete-reference
hits are sparse. The learning-rate scheduler also uses oracle PER@N.

A count-based phoneme bigram supplies a tiny autoregressive prior during beam
search. It is fit only from training targets with additive smoothing and is
stored in the checkpoint; no transcript, audio, validation reference, or test
reference is consulted at inference. Its weighted log probability is added
when a candidate prefix is extended. This is shallow fusion, not a replacement
for the visual CTC model and not a large external language model.

Prediction JSON includes the ranked phoneme candidates, CTC log scores, and
probabilities normalized within the returned candidate set. These relative
probabilities are useful to the next probabilistic stage but are not calibrated
probabilities over every possible phoneme sequence.

Use `evaluate_nbest.py` to encode validation clips once and report a consistent
oracle curve at N=1, 3, 5, 10, and 20. The evaluator supports multiple language
model weights without rerunning the neural network. Keep the beam width at least
as large as the maximum requested N.

### Evaluation

PER is the total Levenshtein substitutions, deletions, and insertions divided by
the total reference phonemes. Report pooled PER, deletions/substitutions/
insertions, per-speaker PER, prediction length, and runtime. Validation speakers
9–10 support development. Keep speakers 11–12 locked until the architecture and
decoder are frozen, then evaluate them once. Report the image-only condition,
coordinate-only condition, and fusion condition with identical preprocessing.

### Tongue-specific extension

Use inner-lip coordinates to rectify a small inner-mouth crop, then apply a
separate CNN that predicts a tongue-visibility score and a compact tongue
embedding. Gate the embedding by visibility before fusion so hidden tongue
position is not invented. Supervise this branch with manually reviewed tongue
masks or keypoints; phoneme identity alone is not a valid tongue-location label.
Evaluate it first on visibly relevant events such as `TH`, `DH`, and some `L`
frames, while allowing an explicit unobservable state.

### Data expansion decision

The current bottleneck is speaker generalization. Eight training speakers
provide many utterances but little variation in facial geometry, articulation,
camera, and lighting. Add data now when it contributes new speakers and retains
reliable synchronization, licenses, provenance, and speaker-disjoint splits.
More near-duplicate utterances from the same GRID speakers have lower value.

The initial modality ablation and stronger coordinate regularization are now
complete. The next data experiment should add speakers in fixed increments while
holding this architecture unchanged. Retain an increment only when it improves
held-out-speaker PER or calibration rather than training loss.

[`AudioPhonemeLabeler`](../AudioPhonemeLabeler/README.md) produces audibly
supervised draft ARPAbet labels for candidate external media. Admit only its
accepted records, preserve source-level speaker grouping, and manually audit a
sample before retraining. Audio remains a dataset-construction teacher and is
not an input to the deployed visual model.

## Run

From the VALLRITE repository root:

```bash
.venv-vpa-gpu/bin/python -m VisualPhoneme.train \
  --data-root datasets/grid-pilot \
  --output-dir checkpoints/visual-phoneme-mouth \
  --max-minutes 30
```

Train and select the centered-coordinate model using Top-5 sequence scoring:

```bash
.venv-vpa-gpu/bin/python -m VisualPhoneme.train \
  --architecture coordinates \
  --coordinate-mode clip-centered \
  --top-n 5 --beam-width 16 --beam-token-top-k 8 \
  --selection-metric top-n-exact \
  --output-dir checkpoints/ablations/coordinates-clip-centered-top5 \
  --max-minutes 30
```

Train the regularized motion-coordinate model and its gated image extension:

```bash
.venv-vpa-gpu/bin/python -m VisualPhoneme.train \
  --architecture coordinates --coordinate-mode clip-centered \
  --coordinate-features motion --landmark-bottleneck 32 \
  --landmark-jitter 0.003 --point-dropout 0.08 \
  --frame-span-dropout 0.15 --max-frame-span 4 --weight-decay 0.003 \
  --top-n 5 --beam-width 16 --beam-token-top-k 8 --lm-weight 0.25 \
  --selection-metric oracle-per-at-n \
  --output-dir checkpoints/ablations/coordinates-motion-regularized \
  --max-minutes 30

.venv-vpa-gpu/bin/python -m VisualPhoneme.train \
  --architecture gated-fusion --coordinate-mode clip-centered \
  --coordinate-features motion --landmark-bottleneck 32 \
  --initialize-coordinate-checkpoint \
    checkpoints/ablations/coordinates-motion-regularized/best.pt \
  --image-modality-dropout 0.20 --coordinate-modality-dropout 0.02 \
  --landmark-jitter 0.003 --point-dropout 0.08 \
  --frame-span-dropout 0.15 --max-frame-span 4 --weight-decay 0.003 \
  --top-n 5 --beam-width 16 --beam-token-top-k 8 --lm-weight 0.25 \
  --selection-metric oracle-per-at-n \
  --output-dir checkpoints/ablations/gated-fusion-motion-regularized \
  --max-minutes 30
```

Recompute a wider oracle curve from a saved checkpoint:

```bash
.venv-vpa-gpu/bin/python -m VisualPhoneme.evaluate_nbest \
  --checkpoint path/to/best.pt --data-root datasets/grid-pilot \
  --n-values 1,3,5,10,20 --lm-weights 0,0.25,0.5 \
  --beam-width 64 --beam-token-top-k 8 --device cuda
```

Quick pipeline check:

```bash
.venv-vpa-gpu/bin/python -m VisualPhoneme.train \
  --data-root datasets/grid-pilot \
  --output-dir checkpoints/visual-phoneme-smoke \
  --train-limit 32 --validation-limit 16 --epochs 2 --workers 0
```

Training writes `train.log`, `metrics.json`, and `best.pt`. Logs include source
filename and line number and are flushed at epoch boundaries. The test speakers
are intentionally absent from the training command; a separate frozen test
evaluator should be added only after model choices are complete.

Predict from a video using the best saved checkpoint:

```bash
.venv-vpa-gpu/bin/python -m VisualPhoneme.predict path/to/video.mpg \
  --checkpoint checkpoints/visual-phoneme-mouth/best.pt
```

This writes a JSON prediction and a sidecar log next to the input unless
`--output` specifies another JSON path. The fixed crop assumes GRID framing.

Train the combined coordinate and image model:

```bash
.venv-vpa-gpu/bin/python -m VisualPhoneme.train \
  --landmark-fusion \
  --output-dir checkpoints/visual-phoneme-fusion \
  --max-minutes 30
```

Fusion inference currently accepts the coordinate cache produced by the VPA
extractor. General video inference will need to run that extractor first:

```bash
.venv-vpa-gpu/bin/python -m VisualPhoneme.predict path/to/video.mpg \
  --checkpoint checkpoints/visual-phoneme-fusion/best.pt \
  --landmarks-npz path/to/video.npz
```
