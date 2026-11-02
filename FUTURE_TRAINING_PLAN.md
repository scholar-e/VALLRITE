# Future evaluation and training plan

Revised 2026-10-29 after a codebase review. Status: proposed; no runs launched
by this document. Dated results and the superseded 2026-10-26 staged plan are
in the [experiment log](EXPERIMENT_LOG.md).

## Current state

Results use the repeatedly inspected 1,140-clip development set (44,586
reference phones, 12,675 words). They are development numbers, not test results.

| System / diagnostic | Result | Role |
| --- | ---: | --- |
| Context visual head, step 750, unadjusted greedy | 82.67% exact PER | Selected visual checkpoint |
| Same model, no phone LM, beam | 70.82% PER; 69.29% oracle PER@5 | Current candidate source |
| Mixed decoder + fixed probability ranking | 93.29% WER | Retained end-to-end baseline |
| Clean reference phones + clean EOS decoder | 13.40% WER | Decoder-only diagnostic |

Preserve `checkpoints/fullclip-phone-context-20261013/phones-step-0750.pt`,
`checkpoints/vallr-eos-20261010/adapter`,
`checkpoints/decoder-current-mixed-20261027/adapter` and
`checkpoints/vallr-actual-errors-20261014/adapter`. Never overwrite them.

## Diagnosis

The word decoder is not the bottleneck: it reaches 13.40% WER on correct phones,
while the visual model gets roughly 70% of phones wrong. Decoder ranking,
calibration and adaptation experiments since 2026-10-26 have moved WER by
fractions of a point, within run-to-run noise. The visual front end is where
the remaining error lives. Specific causes found in the review:

1. **Auto-AVSR is mostly unused.** `checkpoints/pretrained/autoavsr/LRS3_V_WER19.1/model.pth`
   holds 250M parameters (published 19.1% LRS3 test WER). `AutoAvsrVisualFrontend`
   loads only the 11.2M-parameter Conv3D/ResNet-18 frontend; the 170M-parameter,
   12-layer conformer encoder and its 5,049-unit CTC head are discarded. The
   selected pipeline uses the 9.3M-parameter scratch CNN instead.
2. **Preprocessing does not match Auto-AVSR.** It expects smoothed-landmark,
   mean-face-aligned 96×96 mouth crops, centre-cropped to 88×88 and normalized
   with mean 0.421, std 0.165. This repository letterboxes to 96×96 and divides
   by 255 (`VisualPhoneme/lrs3_data.py`).
3. **The landmark mouth crop fights the signal.** `_mouth_crop` recomputes the
   box per frame without smoothing, sized `max(1.55·width, 2.30·height)` of the
   mouth, so it zooms out as the mouth opens. Pretrain rows lack landmarks and
   fall back to a fixed rectangle while validation uses landmark crops; the test
   split has no joined landmarks. A single detector run on every split is needed.
4. **Training is far too short.** 1,006,263 training rows (31.5M frames) exist;
   phone-head runs use 4,000 clips and about 6,000 presentations. 89% greedy
   blank emission is consistent with undertrained CTC.
5. **Padding changes outputs.** Temporal convolutions and BatchNorm see padded
   frames without a mask; running clips alone changes PER 82.67% → 83.19%.
6. **Development data overlaps Auto-AVSR training.** The 1,140 clips come from
   LRS3 trainval, which Auto-AVSR trained on. Its development scores will be
   optimistic. The official 1,321-clip LRS3 test set appears unused and is the
   set on which VALLR (18.7%) and Auto-AVSR (19.1%) report.

## Ordered plan

The user approves each phase's scope and budget before launch.

### Phase 0 — engineering hygiene (no GPU)

Commit outstanding work; split this plan from the dated log; replace duplicated
architecture-name sets with `VisualPhoneme/architectures.py`. Done 2026-10-29.
The untracked root `:memory:.ses` (a session ID from a library given `:memory:`
as a path) is not committed and was left in place. Padding masks
change model behaviour and need matched retraining, so they move into Phase 2.

### Phase 1 — reproduce Auto-AVSR as published (key gate)

**Budget:** about one GPU hour of evaluation after implementation.

Port the conformer encoder and its CTC head without the legacy ESPnet stack.
Obtain the `unigram5000` SentencePiece model from the Auto-AVSR repository; the
local zip contains only `model.json` and `model.pth`, and `model.json`'s output
dimension (41) does not match the 5,049-way CTC layer. Implement the upstream
preprocessing: face landmark detection, temporal smoothing, mean-face affine
alignment, 96×96 crop, 88×88 centre crop, normalization. Run it identically on
train, validation and test. Score greedy-CTC word output on the official test set.

**Gate:** WER within a few points of the published 19.1% (greedy CTC without
the attention decoder should be somewhat worse). If far off, the preprocessing
is wrong; fix it before Phase 2. That result may also explain earlier failures.

**Status 2026-11-02 — substantially met.** The full model is ported. On the
official test set it reaches 31.72% WER with greedy CTC and 26.35% with joint
CTC/attention decoding and no language model, against 93.29% for the current
pipeline on development data. The 7-point gap to the published figure is
partly the unported language model; test-crop provenance is unverified. The
test crops already match Auto-AVSR's style, so only training data still needs
the new crop pipeline. See [the reproduction record](VisualPhoneme/AUTOAVSR_REPRODUCTION.md).

### Phase 2 — phone CTC head on the Auto-AVSR encoder

**Budget:** propose a two-hour resumable run; extend only after inspecting curves.

Train a 768→40 linear phone CTC head on the frozen encoder using the full
pretrain+trainval data with Phase 1 crops. Then test unfreezing the top
conformer blocks with a small encoder learning rate. Use padding masks
throughout. Report PER on development and on a held-out portion of the official
test set, because of the overlap in diagnosis item 6.

**Gate:** clear, repeatable exact-PER gain over the current 68–71%.

**Status 2026-11-02 — met on a single seed; strategic decision pending.**
Trainval is now cropped with the Auto-AVSR procedure. On the held-out test
half, a linear head on frozen features reaches 40.34% beam PER and a two-layer
transformer head 27.99%. However, Auto-AVSR's own word output converted to
phones reaches 18.80% on the same clips. The phone-first route currently loses
to word-first decoding. Choose a direction before Phase 3. See
[the phone-head record](VisualPhoneme/AUTOAVSR_PHONE_HEAD.md).

### Phase 3 — rerun the decoder stack on the new phones

Run the clean EOS adapter and fixed ranking unchanged first. Then repeat matched
noisy-phone adaptation with cross-fitted predictions, reusing
`export_matched_phones`, the source-disjoint split and validation-WER selection.

**Gate:** actual selected WER beats both the 93.29% baseline and an unchanged
decoder on the same new phones.

**Status 2026-11-02 — unchanged-decoder step done; phone interface retained.**
On 661 held-out test clips, Auto-AVSR phones through the clean EOS decoder with
fixed ranking reach 49.83% WER, versus 25.54% for Auto-AVSR's own words. The
same decoder on correct phones reaches 15.33%, so it is not the ceiling, but it
amplifies phone errors. Next: matched-error decoder adaptation. See
[the decoding record](CompetitiveWordDecoder/AUTOAVSR_PHONE_DECODING.md).

**Update 2026-11-02 — matched-noise adaptation.** Continuing the decoder on
LRS3 transcripts with phone errors matched to the measured head reduces
held-out ranked WER from 49.83% to 40.13% (selected on test-select; single
seed). Domain/format change is confounded with noise; run the clean-only LRS3
control next. Auto-AVSR words remain ahead at 25.54%.

### Phase 4 — project-specific ideas on the stronger base

Timed/probabilistic phone input, VPA, family objectives and audio-teacher
distillation. Optionally compare a direct encoder-to-LLM baseline as a ceiling.

## Pause until Phase 2 lands

| Option | Why pause |
| --- | --- |
| Decoder continuation, ranking or calibration runs | Input phones are too noisy for differences to be informative |
| Blank-bias tuning and rolling smoothing | Operating-point fixes for an undertrained model |
| Further scratch-CNN training or family search | A far stronger pretrained encoder is already available |

Check the Auto-AVSR checkpoint licence and LRS3 dataset terms before building
on them beyond research use.

## Common measurement and safety rules

- Primary end-to-end metric: **WER of the actually selected output**. Report
  exact-phone PER, oracle@5, substitution/deletion/insertion counts, output length,
  exact-sentence hits, generation caps, and latency as secondary diagnostics.
- Keep family PER explicitly separate. Never count a merged family label as an
  exact-phone improvement. Oracle selection must never enter the inference path.
- Freeze normalization, clip eligibility, reference hashes, beam settings and
  scoring code for a comparison. Keep excluded examples and reasons visible.
- Use the existing development set for exploratory continuity. Before tuning a
  new learned ranker or selecting a longer run, establish a separate selection
  split. Audit all upstream visual/audio/decoder training exposure before calling
  another split untouched. Prefer speaker/source-video-disjoint partitions.
- Reserve a final confirmation set that is never used for training, checkpoint
  selection, calibration or hyperparameter choice. Freeze the winner before
  opening its result. Repeated evaluation makes a set development data.
- Report paired uncertainty, preferably source-video/speaker-cluster bootstrap
  intervals, plus variation across training seeds. A tiny single-seed gain is
  provisional; statistical significance alone is not practical usefulness.
- Run one GPU workload at a time. Check GPU visibility with the repository
  launcher. Create monitor directories before starting thermal protection and
  verify the watchdog is alive; the previous export had a startup failure.
- Keep existing cooling thresholds. Do not trade thermal safety for utilization.
  Test PAUSE and resume on a short smoke run before any long stage. Verify
  optimizer, scheduler, RNG and data position restoration, not only weights.
- Save initial, periodic, best-selection and final checkpoints with manifests.
  Preserve logs/metrics for failures and negative results. Time caps may exclude
  preparation/evaluation in current scripts: budget those separately and record
  both optimization and total wall time. A capped run is incomplete, not success.


## Handoff and reporting

Each run gets an ID, hypothesis, approved cap, source hashes, split manifest,
config, checkpoint curve, final metrics and a short keep/reject decision.
Report failed/capped/paused runs distinctly from completed runs. Stay through
planned evaluation unless background-only execution is requested. Do not start
the next experimental stage automatically just because the previous process exits.

Keep data and credentials local; review artifacts should contain methods and
aggregate metrics, not raw recordings or secrets. No new dependencies are
required by this plan; verify any future dependency and justify it before adding.


## Independent review

The previous review command points at a non-existent launcher path, so no
critique has ever been produced. Use the `deepseek-review` skill before
launching Phase 2's training budget.
