# Four blank-mitigation pilots, 2026-10-28

Predeclared before launch. The user authorized all four strategies. No new
dependencies or model downloads are needed. This is a single-seed development
pilot, not independent test confirmation or permission for overnight runs.

1. **Separate decoder calibration.** Reuse the previously measured greedy
   calibration (-2, selected on 256 training-side clips). Independently compare
   beam blank biases 0, -0.5 and -1 on those same cached calibration logits, with
   no phone LM, beam 64, token top-k 12 and five outputs. Select lowest rank-one
   calibration PER, then freeze the choice for full development evaluation.
   Never select using development references or call oracle selection deployable.
2. **Longer clips and consistent crops.** Continue the existing step-750 phone
   head for 300 updates on 512 selected training clips. Compare the legacy
   sampling policy with 512 landmark-equipped clips of at least 60 frames. For
   the longer clips, compare fixed crops with landmark crops on identical IDs.
   Long-vs-legacy also changes source composition; only the crop comparison is
   a matched same-data ablation. Exclude development source-video IDs.
3. **Weak phone-count loss.** On the same longer/landmark data, add 0.1 times
   smooth-L1 loss on expected collapsed CTC count divided by reference count.
   The expectation is sum over time and nonblank class of
   p_t(c) * (1 - p_(t-1)(c)), with previous probability zero at t=0 and padding
   excluded. This counts repeated phones separated by blank correctly.
4. **Filtered teacher supervision.** On those exact same longer/landmark IDs,
   add 0.1 times soft-label cross entropy for accepted temporal windows only.
   Require clip quality >=0.95, forced-alignment score >=0.8, teacher agreement
   >=0.95, transcript agreement >=0.95, in-bounds intervals and >=10% supervised
   frames. Windows are 80 ms wide, require >=20 ms labelled overlap and >=70%
   of overlap mass on one phone. Unknown frames remain masked; no silence or
   blank labels are invented. Window targets are interval-derived distributions,
   not saved acoustic posteriors or calibrated confidence estimates.

Training arms: legacy, long_fixed, long_landmark (CTC control), count, teacher.
All start at the same checkpoint, use AdamW lr 3e-5 / weight decay 0.01, batch
eight, seed 20261028, 300 updates, checkpoints/evaluation every 100 updates and
five-minute training cap per arm. No checkpoint is chosen on development WER.
Encoder features are cached with fixed batch membership/padding; only the
218,192-parameter context phone head changes. Development batch eight must
reproduce the original 36,860 greedy errors before training is permitted.

Preparation and final evaluation are overhead outside each training cap. Use
host GPU, thermal guard, repository/run PAUSE checks and atomic resume state.
Resume with the same output and arm plus --resume. Partial preparation must be
rerun; completed preparation has PREPARED. Completed training has metrics.json
and COMPLETE; process termination is not evidence of completion.

Report exact PER, substitutions/deletions/insertions, blank fraction, output
length and full checkpoint curves. Run the frozen mixed word decoder on the
final training-arm greedy predictions and selected beam output, using the
existing generation settings. Keep calibration, model and source hashes, IDs,
teacher coverage and thresholds. No automatic promotion from PER alone.

Planned artifacts: `checkpoints/blank-strategies-20261028` and
`checkpoints/beam-calibration-20261028`.

## Results

All five training arms completed 300 updates. Exports and optimizer states
confirm completion; all non-phone-head tensors remain identical to the source
checkpoint. Caching frozen features made the training/evaluation loop only
1.6–2.0 seconds per arm; data preparation, model loading and word evaluation
are separate overhead. These are small head-continuation pilots, not minutes
of end-to-end encoder fine-tuning.

All results below use 1,140 development clips, 44,586 phones and 12,675 words.
Word results use one greedy phone path with the same frozen mixed decoder;
adapter hashes, generation settings, clip IDs and word references were checked.

| Arm | Exact greedy PER | Frozen-decoder WER | Interpretation |
|---|---:|---:|---|
| Starting checkpoint | 82.67% | 95.70% | Retained matched greedy baseline |
| Legacy continuation | 83.13% | 94.90% | Training control |
| Longer clips, fixed crops | 73.69% | 96.03% | Phone gain, no word gain |
| Same longer clips, landmark crops | 83.48% | 96.21% | Matched CTC control for losses |
| Landmark crops + count loss | 83.52% | 96.25% | No benefit at this weight |
| Landmark crops + teacher loss | 82.81% | 97.28% | Small PER gain versus matched control; WER regression |

### 1. Separate beam calibration

| Blank bias | Calibration beam PER |
|---|---:|
| 0 | 61.72% |
| -0.5 | 58.16% |
| **-1** | **57.77%** |

The frozen choice -1 reaches **67.71% rank-one PER** and **66.13% oracle PER@5**
on development. This improves the previous no-prior beam's 70.82% / 69.29%
without model training or family merging. The independently selected greedy
setting remains -2 at 68.85% PER. The shared cached calibration data has 256
clips; it is not an untouched test set or out-of-fold encoder output.

The selected beam's frozen mixed-decoder WER is 106.91% before ranking
(13,551 errors), **97.59% after fixed ranking** (12,370 errors), and 92.21%
oracle@5 (11,687 errors). There were 29 capped generations. Ranking preserved
all candidate strings and the oracle count. This is worse than the retained
93.29% ranked pipeline despite better phone PER; no end-to-end replacement is
justified. Generation took about five minutes and ranking about 49 seconds.

### 2. Data and crop comparison

The selected legacy clips have median 28 frames and only 5/512 landmark rows.
The longer set has median 122 frames, all with landmarks. Development median
is 73. The longer set was also restricted to teacher-eligible examples so all
loss arms could share exact IDs; source and quality composition therefore
change with length. The comparison does not isolate duration alone.

Fixed crops on longer clips improve phone PER but not WER. Landmark crops on
the identical IDs do not improve this frozen-encoder pilot. That rejects an
immediate crop-only fix for these weights; it does not establish that landmark
crops are inherently worse after end-to-end retraining.

| Updates | Legacy | Long fixed | Long landmark | Count | Teacher |
|---|---:|---:|---:|---:|---:|
| 0 | 82.67% | 82.67% | 82.67% | 82.67% | 82.67% |
| 100 | 83.45% | 71.97% | 83.85% | 83.91% | 82.72% |
| 200 | 83.46% | 71.59% | 83.71% | 83.75% | 82.84% |
| 300 | 83.13% | 73.69% | 83.48% | 83.52% | 82.81% |

The table reports the predeclared final step, not the lowest development point.

### 3. Expected count versus greedy collapse

The original model already has an expected collapsed count of **47,772.6**
against 44,586 reference phones, even though greedy emits only 10,514 phones.
Expectation over CTC alignments and the count from the most likely per-frame
labels differ substantially. Consequently the weak expected-count loss is
already nearly satisfied and does not resolve greedy underproduction. The
implementation agrees with exhaustive enumeration and excludes padded frames;
the negative result is not evidence that it counts nonblank frames by mistake.

### 4. Teacher filtering and limitations

6,056 training clips passed selection; 512 were sampled. Window supervision
covers 22,125/59,107 frames (37.43%) in this set. The remaining frames are masked.
429 selected alignment files originate from official-transcript CTC alignment,
and 83 from the ASR-teacher labeler. For official-transcript files, transcript
agreement is not independent teacher agreement. Forced-alignment confidence is
a clip-level proxy, not calibrated per-phone correctness. No raw acoustic
posterior distributions were available in these interval exports.

The prior alignment audit found no interior label-mapping mismatches, but
audio/video stream offsets were not verified there (ffprobe unavailable).
This run checks interval bounds and transcript agreement, not manual audiovisual
synchronization. Interpret the teacher arm as a filtered interval-supervision
pilot, not proof that fully verified audio distillation fails.

Preparation initially encountered older records with confidence stored inside
the alignment document; it now supports that format. Preparation was paused
and restarted to vectorize window construction; no training weights were
changed during a running arm. Completed preparation and all checkpoints are
preserved. No new dependencies were installed.

### Final decision

All four strategies have completed their planned comparisons. Retain the -1
no-prior beam as the best tested *phone* N-best configuration, with explicit
calibration provenance. Keep the current word pipeline unchanged. The longer
fixed-crop head is a useful phone-training result but has no demonstrated WER
benefit. Do not adopt the count loss or this teacher-loss setting as defaults.
These results do not rule out differently calibrated losses, verified teacher
posteriors, or end-to-end encoder adaptation; those would be new experiments.

44 targeted tests pass. Baseline error reproduction, final 300-step states,
frozen encoder tensors, input/reference identities, decoder settings and
ranking-oracle invariance were independently checked. No model weights or
decoding defaults outside the experimental outputs were replaced.

## Reproduction and pause/resume

Use a fresh output directory for preparation:

```bash
tools/with-vpa-gpu .venv-qwen/bin/python -m VisualPhoneme.test_blank_strategies \
  --output checkpoints/NEW-STRATEGY-RUN --prepare
tools/with-vpa-gpu .venv-qwen/bin/python -m VisualPhoneme.test_blank_strategies \
  --output checkpoints/NEW-STRATEGY-RUN --arm count --max-minutes 5
```

Choose each arm from `legacy`, `long_fixed`, `long_landmark`, `count`, `teacher`.
After a paused training arm, remove the PAUSE file you created and repeat the
arm command with `--resume`. The latest atomic optimizer/head state is used;
the source manifest must match. The paired frozen-feature preparation must be
complete first. Do not delete another task's PAUSE request.

Beam calibration is CPU-only over saved GPU logits:

```bash
.venv-qwen/bin/python -m VisualPhoneme.calibrate_phone_beam \
  --output checkpoints/NEW-BEAM-CALIBRATION --max-minutes 15
```

It checks PAUSE between clips and reuses validated partial JSONL outputs on
restart with the same output path. Word evaluations use
`CompetitiveWordDecoder.evaluate_llama_visual`; training-arm exports require
`--visual-top-one`, while beam exports contain five paths. Ranking uses the
unchanged `CompetitiveWordDecoder.evaluate_probability_ranker` rule.
