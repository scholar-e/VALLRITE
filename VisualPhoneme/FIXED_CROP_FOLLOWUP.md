# Fixed-crop continuation, 2026-10-28

Predeclared follow-up to the four blank-strategy pilots. User authorized fixed
crops and further improvements. No deployment changes or new dependencies.

The previous long_fixed arm trained on fixed crops but evaluated on landmark
crops. First measure that same final step-300 head on both validation crop
policies; this is an inference-policy ablation, not proof that either framing
is intrinsically better.

Continue the final step-300 fixed-crop head on its same 512 clips, using the
cached encoder features. Compare constant lr 3e-5 with cosine lr 1e-5 → 1e-6,
both for 600 additional updates, batch eight, weight decay .01, norm cap five,
seed 20261029, checkpoint/evaluation every 100 steps. Both reset AdamW state
and use identical deterministic batch order. Each training arm is capped at
five minutes; preparation and word/beam evaluations are separate overhead.
Keep the visual encoder frozen; do not reintroduce unsuccessful count/teacher
losses. This tests learning-rate stability, not end-to-end visual fine-tuning.

Sample 128 eligible long (>=60 frame) training-side calibration clips with
landmarks available, one per source video, excluding development source videos
and source videos in both prior phone-head training subsets. Apply fixed crops
to these clips. Encoder pretraining exposure remains possible. The existing
1,140-clip development set is repeatedly inspected, not an independent test set.

Select checkpoint/arm only by calibration greedy PER, including step zero as
a contender and favoring earlier checkpoints on ties. Report training CTC loss
and calibration curves to investigate the earlier late regression. Evaluate
both final arms and the calibration-selected checkpoint on development, but
never use that result to select a checkpoint.

Calibrate greedy blank bias on the selected checkpoint using 0,-.5,-1,-2.
Separately calibrate beam bias with 0,-.5,-1, no phone LM, beam 64/top-k12/five
paths. Freeze selections before development decoding. Cache logits and record
all hashes/IDs. Measure frozen mixed-decoder WER on the fixed-crop starting
head and selected calibrated greedy output, then evaluate/rank selected beam
candidates using the existing fixed rule. Do not promote based on PER alone.

Preserve root/run PAUSE checks, thermal guard, atomic optimizer/head resume
state, exported checkpoints and completed metrics. Partial preparation can be
repeated; completed caches are reusable. Stop after these comparisons.

## Results

Both continuations completed all 600 updates. The frozen encoder tensors were
verified unchanged. Each arm saved seven curve points and resumable optimizer
state. Calibration contains 128 unique source videos, disjoint from both
phone-head training subsets and development (median 116 frames).

| Additional updates | Constant calibration greedy PER | Low-cosine calibration greedy PER |
| --- | ---: | ---: |
| 0 | 83.72% | 83.72% |
| 100 | 84.15% | 83.98% |
| 200 | 84.11% | 83.89% |
| 300 | 84.51% | 84.05% |
| 400 | 85.22% | 84.17% |
| 500 | 84.91% | 84.26% |
| 600 | 85.68% | 84.26% |

Training/calibration CTC loss fell from 2.7647/2.9630 to 2.5805/2.8036
(constant), or 2.7154/2.9200 (low cosine). Greedy PER worsened despite improved
calibration loss; this does not establish ordinary validation overfitting.
The original greedy-selection protocol selected step zero.

| Head | Development greedy PER, fixed crops | Development greedy PER, landmark crops |
| --- | ---: | ---: |
| Starting head | 80.74% | 73.69% |
| Constant final | 82.53% | 80.42% |
| Low-cosine final | 81.18% | 75.33% |

These are the same 1,140 clips / 44,586 reference phones. Simply switching
inference to fixed crops worsened the starting head. Calibrating its greedy
blank bias selected -2 and reduced fixed-crop PER to 72.98%, but frozen mixed
decoder WER rose from 100.97% (12,798 errors) to 115.64% (14,657 errors), on
12,675 words. Generation-cap hits rose from one to eight. WER can exceed 100%
when insertion errors are numerous.

The exploratory beam comparison selected the constant final head with bias
-1: calibration PER 70.14%, versus 72.44% for the starting head at -.5 and
72.18% for low cosine at -.5. This choice was recorded in
`beam-head-selection.json` before challenger development decoding.
The starting head's separate beam development result is 71.35% PER / 69.85%
oracle PER@5. The selected constant continuation reaches 68.70% PER
(30,629 errors) / 67.16% oracle PER@5 on development. It improves over this
fixed-crop starting beam, but does not beat the earlier 67.71% PER / 66.13%
oracle PER@5 result.

The frozen mixed word decoder produces 111.40% WER (14,120 errors), falling
to 95.64% (12,122 errors) under the existing probability ranker. Word oracle
WER@5 is 92.36% (11,707 errors); ranking preserves this oracle exactly.
Thirty of 5,700 generations reached the 256-token cap. The diagnostic
length-normalized ranker gives 117.95% WER; it was not selected or tuned.
The retained pipeline remains better at 93.29% ranked WER and 90.12% oracle
WER@5. No model or decoding defaults were promoted.

## Conclusion and next priorities

Fixed crops plus continued head training improve this experiment's beam PER,
but do not improve the strongest existing phone or word result. Greedy PER
and CTC loss rank these checkpoints differently, so future selection should
use the intended sequence decoder on calibration data. Lower PER alone is
not evidence of improved phoneme-to-word usefulness.

A future experiment should first examine paired deletion, substitution and
insertion patterns and decoder behavior on the changed phone distribution.
Then test matched noisy-phone decoder adaptation, with a fresh untouched
evaluation split and WER as the endpoint. Broader full-clip training or encoder
fine-tuning may be worth testing, but this frozen-feature pilot cannot establish
their benefit. No further training was launched by this recommendation.

Artifacts: `checkpoints/fixed-crop-followup-20261028/`, including per-arm
checkpoints/curves, greedy selection, `beam-head-selection.json`, source hashes,
and `beam-constant/{metrics.json,word-eval/metrics.json,word-ranked/metrics.json}`.
All 1,140 IDs and phone/word references were checked against the earlier
evaluation, with five paths per clip. All 44 relevant regression tests passed;
`git diff --check` passed. GPU evaluations retained PAUSE and thermal protection;
brief CPU thermal pauses resumed after cooling. Results are development
measurements, not official test accuracy or manually verified phone-label error.

## Diagnostic extension declared after continuation loss curves

Both final arms reduced calibration CTC loss while worsening greedy PER.
Before looking at their beam development results, add a calibration-only
comparison of the two final (600-update) heads against the starting head.
Use the same 128 calibration clips and beam-bias grid for all three. Select
head and bias by calibration rank-one beam PER, then evaluate only that frozen
choice on development and with the frozen word decoder/ranker. No intermediate
checkpoint beam search or additional training is introduced. This is an
exploratory extension motivated by the loss/greedy discrepancy, not part of
the initial checkpoint-selection protocol. Preserve both sets of results.
