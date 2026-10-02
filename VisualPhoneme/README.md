# Compact visual phoneme recognizer

This branch trains a small visual-only network directly on face video. A shared
2D CNN recognizes each grayscale frame, a dilated temporal convolution stack
models motion and coarticulation, and a CTC head predicts an ARPAbet phoneme
sequence. Audio is used only to create training targets; inference uses video.

## LRS3 integration

The 30-hour LRS3 trainval split is supported alongside GRID. Source archives
are kept under `datasets/lrs3/downloads/`; extracted face-track videos live in
`datasets/lrs3/raw/trainval/`. The preparation pass creates transcript-derived,
stress-free CMUdict targets and uses AV-HuBERT's published 1,200-utterance
validation list rather than inventing a random clip split:

```bash
.venv-vpa-gpu/bin/python -m VisualPhoneme.prepare_lrs3 \
  --root datasets/lrs3
```

The supplied 68-point tracks are eye-normalized for the coordinate branch. Lip
corners and inner-lip aperture points are given stable leading positions so the
hybrid and observability-gated branches retain their intended semantics.
Missing detections remain time-aligned as masked `NaN` frames.

The 403-hour LRS3 pretrain partition can be added to the existing 30-hour
trainval manifest with:

```bash
.venv-vpa-gpu/bin/python -m VisualPhoneme.prepare_lrs3 \
  --root datasets/lrs3 --include-pretrain --pretrain-max-phones 16
```

This uses LRS3's supplied word boundaries to divide long recordings into short
CTC-safe chunks. The pretrain release does not include matching 68-point tracks,
so `--architecture large-image` provides a 9.26M-parameter image/temporal model
whose `frame_encoder` and `image_projection` tensor shapes exactly match
`large-gated-fusion`. Its visual path can therefore be transferred with
`--initialize-image-checkpoint` before hybrid fine-tuning. Validation remains
the original AV-HuBERT 1,200-ID split and pretrain material is training-only.

An LRS3 hybrid training run can then be launched with:

```bash
.venv-vpa-gpu/bin/python -m VisualPhoneme.train \
  --dataset lrs3 \
  --data-root datasets/lrs3 \
  --architecture gated-fusion \
  --crop mouth \
  --image-size 96 \
  --coordinate-mode eye-normalized \
  --coordinate-features motion \
  --output-dir checkpoints/lrs3-gated-fusion
```

For weak time supervision, audio-teacher `labels.jsonl` output can be joined to
the original LRS3 files with `--lrs3-teacher-labels`. The loader verifies the
teacher transcript against the official transcript, turns consecutive timed
words into short training chunks, and creates auxiliary nonblank frame labels.
For example, the quarter-data coordinate curriculum uses:

```bash
.venv-vpa-gpu/bin/python -m VisualPhoneme.train \
  --dataset lrs3 --data-root datasets/lrs3 \
  --architecture coordinates --coordinate-mode clip-centered \
  --coordinate-features motion --landmark-bottleneck 32 \
  --train-limit 7895 \
  --lrs3-teacher-labels datasets/lrs3/audio-teacher-quarter \
  --max-chunk-phones 16 --min-teacher-transcript-agreement 0.8 \
  --aligned-frame-loss-weight 1 --frame-only-epochs 5 \
  --ctc-loss-weight 0.1 --ctc-upsample-factor 2
```

Teacher timing is word-level. The phones inside each word are uniformly divided
over its interval, so the auxiliary labels are deliberately treated as weak
supervision. Validation is never teacher-chunked.

The mirrored test Parquet files contain 1,321 precomputed 96×96 grayscale mouth
videos and transcripts. They are materialized as individual NumPy arrays for
efficient evaluation. They do not retain the original LRS3 clip identifiers,
so there is currently no verified join to the separately named test landmark
files. Test evaluation is therefore image-only until that mapping is recovered;
train and validation support image, coordinates, and hybrid inputs.

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

`--architecture large-gated-fusion` is a separate 9,405,321-parameter LRS3
family. It widens the image/geometry representation to 384 channels and uses
ten full residual temporal-convolution blocks with repeated dilations 1, 2, 4,
8, and 16. It keeps the same aligned-data loader, per-frame CTC output, decoding,
and checkpoint metadata. Existing compact GRID architecture names and tensor
shapes are unchanged, so their checkpoints remain reproducible. The large model
is about 60 times the compact LRS3 hybrid but about 19 times smaller than the
182.4M-parameter original VALLR visual model.

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

Prediction also preserves visual ambiguity through a fixed, exhaustive viseme
partition. Each decoded position contains its observed phone, group name, and
every allowed member—for example `{B,M,P}`, `{F,V}`, `{DH,TH}`, and
`{CH,JH,SH,ZH}`. Selecting a group therefore passes all of its phones to the
next decoder instead of prematurely choosing among visually indistinguishable
sounds. This guarantees inclusion only within the selected group; it cannot
recover the true phone when the visual model selects the wrong group. The raw
39-phone probabilities and original phoneme hypotheses remain available.

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

### Observable inner-mouth extension

`--architecture tongue-gated-fusion` adds a separate CNN over the central
inner-mouth crop. Its residual is multiplied by an observability value derived
from tracked lip aperture and is zero for closed or untracked mouths. This lets
the model use visible tongue and teeth pixels without claiming to infer hidden
tongue position. The fixed crop is an initial GRID-camera implementation; a
landmark-rectified crop is still preferable for unconstrained video.

`--initialize-image-checkpoint` can initialize the full-mouth and inner-mouth
encoders from an image-only model. `--freeze-coordinate-epochs 3` first trains
the visual residuals against the fixed coordinate recognizer, then jointly
fine-tunes all paths. Gate initialization is explicit and checkpointed.

The initial run did not improve validation: the image-pretrained full-mouth
model reached 36.70% oracle PER@5, and the inner-mouth model reached 37.41%,
versus 36.35–36.38% for the coordinate-dominant gated baseline. Both visual
gates stayed near 4.9%, proving that the branches were active rather than
collapsed. This is evidence that phoneme CTC supervision alone is insufficient
for a tongue specialization, not evidence that tongue pixels lack value.

`AudioPhonemeLabeler.tongue_review` exports candidate `TH`, `DH`, and `L`
intervals with `tongue_visibility: null` and `admit_to_training: false` for
manual review. Only reviewed visibility/mask or keypoint labels may supervise a
tongue-specific auxiliary objective; phoneme identity is not a tongue-location
label.

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

Scale transcript-constrained acoustic boundaries across an LRS3 manifest
without rerunning ASR:

```bash
tools/with-vpa-gpu .venv-vpa-gpu/bin/python \
  -m AudioPhonemeLabeler.align_lrs3_manifest \
  --manifest datasets/lrs3/manifests/train.jsonl \
  --data-root datasets/lrs3 \
  --output-dir datasets/lrs3/audio-teacher-full-forced \
  --seed-labels datasets/lrs3/audio-teacher-quarter-forced/labels.jsonl \
  --device cuda --local-files-only
```

The command is resumable through `labels.jsonl`, writes dual-output progress to
`align.log`, and keeps pronunciation-coverage rejections auditable.

From the VALLRITE repository root:

```bash
.venv-vpa-gpu/bin/python -m VisualPhoneme.train \
  --data-root datasets/grid-pilot \
  --output-dir checkpoints/visual-phoneme-mouth \
  --max-minutes 30
```

Run the image-pretrained, aperture-gated inner-mouth ablation:

```bash
.venv-vpa-gpu/bin/python -m VisualPhoneme.train \
  --architecture tongue-gated-fusion --coordinate-mode clip-centered \
  --coordinate-features motion --landmark-bottleneck 32 \
  --initialize-coordinate-checkpoint path/to/coordinate-best.pt \
  --initialize-image-checkpoint path/to/image-best.pt \
  --freeze-coordinate-epochs 3 --image-gate-initial-probability 0.05 \
  --inner-mouth-gate-initial-probability 0.05 \
  --top-n 5 --beam-width 16 --selection-metric oracle-per-at-n \
  --output-dir checkpoints/ablations/tongue-gated --max-minutes 30
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

Training writes `train.log`, `metrics.json`, the selected `best.pt`, and an
`epochs/epoch-NNNN.pt` checkpoint after every validation epoch. Each periodic
checkpoint contains its training and validation metrics, selection key, decoder
configuration, and learned gate values so the improvement curve can be
reanalyzed later. Use `--checkpoint-every N` to retain a coarser interval or
`--checkpoint-every 0` to disable periodic weights. Logs include source filename
and line number and are flushed at epoch boundaries. The test speakers are
intentionally absent from the training command; a separate frozen test evaluator
should be added only after model choices are complete.

Long runs also write `resume.pt` after every validated epoch and whenever they
pause at a batch boundary. Create `OUTPUT_DIR/PAUSE`, or send `SIGINT`/`SIGTERM`,
to request a clean pause. Remove the marker before resuming, then repeat the
original command with `--resume-state OUTPUT_DIR/resume.pt`. The resume state
restores weights, optimizer moments, scheduler, random-number generators,
selection history, and early-stopping counters. If paused partway through an
epoch, completed optimizer updates are retained and that epoch is replayed with
a fresh shuffle; epoch-boundary resumes are exact. `--max-minutes` uses the same
clean state-writing path, so a bounded run can be extended later.

Monitor a live run without affecting it:

```bash
tools/watch-visual-training \
  checkpoints/lrs3-pretrain-433h-large-image-gpu-20260923-v2
```

The optional second argument is the refresh interval in seconds. The monitor
shows within-epoch clip progress, the latest loss/PER line, validation results,
GPU load, pause-marker status, and recently written checkpoints. Stopping the
monitor with `Ctrl+C` does not signal or stop the trainer.

Training starts its own thermal watchdog independently of this monitor. It
checks the hottest NVIDIA GPU and CPU hwmon readings every second, pauses the
trainer and its workers at GPU **80 °C** or CPU **85 °C**, and resumes only when
the GPU is below **75 °C** and CPU below **80 °C**. Missing required sensor
readings also pause training until readings recover. CPU-only runs do not
require a GPU reading. Linux CPU drivers supported are `coretemp`, `k10temp`,
`k8temp`, and `zenpower`; CUDA runs need `nvidia-smi` available on the host.

If either temperature exceeds **95 °C**, the watchdog immediately kills the
trainer and workers on detection and writes a `PAUSE` marker. It does not wait
for a batch or save a new checkpoint; the last saved checkpoint remains the
recovery point. Sensor commands have a two-second timeout, so detection is
subject to polling and sensor latency. Thermal events are written to
`train.log`. These protections apply to newly started or resumed training;
an already-running trainer must be restarted to load them.

From the repository root, request an immediate clean pause for every active
VisualPhoneme trainer:

```bash
./pause-training
```

The command discovers each trainer's `--output-dir`, creates its persistent
`PAUSE` marker, and sends the handled interrupt signal so the main process
wakes promptly. Training finishes its current batch, writes `resume.pt`, and
exits. An explicit `./pause-training OUTPUT_DIR` creates a marker even when the
trainer is not currently visible. Remove that marker before resuming.

Resume from the repository root without reconstructing the original command:

```bash
./resume-training OUTPUT_DIR
```

The command validates `metrics.json` and `resume.pt`, refuses to duplicate an
active run, removes `OUTPUT_DIR/PAUSE`, reconstructs the recorded arguments,
adds `--resume-state`, and launches a detached user service. The resumed job
therefore survives terminal and Wi-Fi disconnections. A reboot still stops it,
but the latest `resume.pt` remains usable.

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
