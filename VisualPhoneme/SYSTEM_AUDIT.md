# Phoneme system audit and calibration, 2026-10-28

Latest: [architecture audit and sequential corrections](../CompetitiveWordDecoder/ARCHITECTURE_AUDIT.md)
records exact-token ranker repair, baseline-inclusive validation selection, and
the detached family-loss accounting fix. Padding sensitivity remains unresolved;
no visual weights were changed by those corrections.

The [fixed-crop follow-up](FIXED_CROP_FOLLOWUP.md) is complete: calibration-selected
continuation reaches 68.70% beam PER / 67.16% oracle PER@5, but 95.64% ranked
WER. The retained word pipeline remains better at 93.29%. No defaults changed.

Follow-up: [all four blank-mitigation strategies have now been tested](BLANK_STRATEGIES.md).
The best tested beam reaches 67.71% exact PER and 66.13% oracle PER@5, but the
retained end-to-end word pipeline remains better. See the follow-up for matched
training controls, full curves and negative results.

## Scope and retained system

This audit follows the [candidate deletion analysis](../CompetitiveWordDecoder/PHONE_CANDIDATE_AUDIT.md).
It examines the current `phone-family-context` checkpoint at step 750, rather
than claiming to evaluate every historical model. It contains 9,477,186
parameters, including 218,192 in the phone head. A frozen pretrained image
encoder and ten temporal convolution blocks feed a linear phone head plus a
nine-frame residual correction. The theoretical temporal receptive field is
257 frames, longer than every clip in the 1,140-clip evaluation set.

Phone CTC supervision is transcript-derived pronunciation, not manually timed
ground truth. Adjacent frames' logits are repeated twice for CTC; this creates
additional alignment states, not new visual evidence. Family auxiliary loss
on a frozen family head is constant with respect to the trainable phone head.
Grouped targets must not be reported as exact-phone accuracy.

## Confirmed findings

1. **Blank overprediction is a major operating-point problem.** Greedy blank
   argmax occupies 89.02% of development emissions. Predicted sequences contain
   only 10,514 phones against 44,586 references. This is a measured property of
   logits, distinct from silence detection: CTC blank is not a silence label.
2. **The phone LM previously worsened deletion bias.** Removing its fixed 0.5
   weight improved beam PER from 82.12% to 70.82%, but did not improve ranked
   word output with the frozen mixed decoder. The prior adds negative scores
   per emitted phone, so it can favor short sequences. No reference-tuned token
   bonus was introduced.
3. **Training and evaluation preprocessing differ.** Only 106/4,000 selected
   phone-head training rows have landmarks. All 1,200 validation-manifest rows
   have them (1,140 pass eligibility and are evaluated). Thus most training rows
   use the fixed mouth rectangle while validation uses landmark-dependent mouth
   crops. Inspection of four training and four validation midframes confirms
   visibly different crop scale and framing; it does not quantify the entire
   corpus. The crop contact sheet is saved with audit artifacts.
4. **Training segments are short.** Median selected training length is 28
   frames; evaluated development median is 73 (range 17–155). Earlier wording
   "full training clips" means complete prepared rows; many rows are already
   pretraining segments, not full sentences. Step 750 at batch eight is only
   about 6,000 sample presentations over the selected 4,000 rows.
5. **Padding changes inference.** Temporal convolutions and BatchNorm operate
   without a padding mask; padded image frames also pass through the encoder.
   Running clips alone changed development greedy PER from 82.67% to 83.19%.
   On calibration clips it changed 73.54% to 76.53%. Removing padding is not an
   immediate accuracy fix for these existing weights. Batch construction must
   be controlled, and any masked-model change needs matched retraining.
6. **No beam recurrence defect was found in exhaustive tiny cases.** Independent
   enumeration of all three-frame/three-class alignments agrees with beam
   scores, with and without a bigram prior, including blank-separated repeats.
   This is a focused correctness check, not proof for all pruning settings.

## Calibration protocol and results

`audit_phone_system.py` caches padded and individual-clip logits. It selects
256 eligible training-side rows with seed 20261028, excluding the exact IDs
used by this phone-head run. These clips can still have pretrained-encoder
exposure or source-video overlap; they are calibration data, not independent
test data. The reference development set is repeatedly inspected and also is
not an untouched test set.

The predeclared blank-logit adjustments are 0, -0.5, -1, -2, -3. Choose by
calibration greedy PER only, breaking ties in favor of the earlier setting.
No development labels choose this adjustment. No training occurs.

| Blank-logit adjustment | Calibration greedy PER |
|---|---:|
| 0 | 73.54% |
| -0.5 | 67.70% |
| -1 | 62.26% |
| **-2** | **58.96%** |
| -3 | 72.10% |

Applying the selected -2 adjustment to development reduces greedy PER from
82.6717% to **68.8490%** (36,860 to 30,697 edits), a 13.82-point improvement.
Predicted phone count increases to 41,101 and blank argmax falls to 56.60%.
Deletions fall from 34,085 to 6,277; substitutions rise from 2,762 to 21,628;
insertions rise from 13 to 2,792. Total errors improve, but the dominant remaining
problem is choosing the correct phone. Common substitutions include IH→AH,
T→N, T→AH and Z→S; these alignments are diagnostic, not unique ground-truth
confusion labels.

**Do not reuse greedy calibration blindly for beam search.** With the same -2
adjustment and no phone LM, beam PER worsens to 82.94% and oracle PER@5 to
81.37%. It emits 54,877 phones and makes 11,675 insertions. The unadjusted no-LM
beam remains preferable for N-best output (70.82% / 69.29%). A future beam
calibration must be selected separately on training-side data and then frozen.
`evaluate_nbest.py` exposes `--blank-logit-bias` and records it in exports and
reports; its default is still zero.

## Downstream word check

Completed matched baseline/calibrated greedy evaluation with the unchanged
mixed word adapter. The new `--visual-top-one` evaluator mode reports one visual
hypothesis without fabricating extra candidates or an oracle@5 result. Adapter
hash, generation settings, all 1,140 IDs and 12,675 reference words match.

| Input to the same mixed word decoder | Greedy phone PER | Word WER |
|---|---:|---:|
| Unadjusted greedy | 82.67% | 95.70% |
| Training-calibrated greedy (-2 blank bias) | 68.85% | 102.45% |

Calibrated output hit the generation cap twice versus zero for baseline.
WER above 100% is possible because insertions count. The decoder was adapted
to much shorter prior-weighted phone sequences; distribution mismatch is a
plausible contributor, not a proven sole explanation. Many of the newly emitted
phones are substitutions. Reduced PER alone does not establish useful sentence
recognition or warrant replacing the current 93.29% ranked word pipeline.

Decision: keep blank-calibrated greedy as an experimental exact-phone operating
point, retain unadjusted no-LM beam for the best current phone N-best results,
and preserve the existing ranked word configuration. These are different
operating points, not interchangeable best-model claims. Forty targeted tests
pass, including single-visual-input scoring, beam enumeration and phone-head
checks; whitespace validation also passes.

## Recommended next training design

First standardize mouth framing and verify a representative audiovisual sample
from each source. Prefer reliable crop coordinates with explicit quality checks;
do not silently assume every fixed crop is wrong or every landmark crop correct.
Then construct training/calibration/confirmation splits by source video, with
recorded prior encoder exposure. Sample a meaningful proportion of longer
sentence clips, documenting duration and phone-count distributions.

With those inputs fixed, compare the current frozen-encoder head with limited
unfreezing of the last temporal block, retaining pause/resume, bounded runtime,
checkpoint curves, exact PER and a frozen word decoder. Measure blank rate and
insertion/deletion/substitution counts alongside loss. Do not select new family
groups or increase model size before testing these concrete distribution and
discrimination issues. Keep crop, padding and calibration changes as separate
ablations so improvements remain attributable.

## Artifacts and review

`checkpoints/phone-system-audit-20261028` contains metrics, selected calibration
IDs, cached logits, edit breakdown, crop images and single-path exports.
`checkpoints/context-calibrated-blank-20261028` contains the rejected beam trial.
All GPU evaluations use the host launcher, thermal guard and a ten-minute cap.
The audit encoding loop observes PAUSE. No model weights were changed.

Local DeepSeek was delegated the exhaustive test (job
`674f8ad4-2c0d-48f2-9d64-86fd4aab3828`). Its draft incorrectly collapsed repeated
labels and summed log probabilities; it was cancelled, and that draft was
replaced with a verified independent enumeration. No worker claim was accepted
as evidence. This audit is supervised local analysis, not independent external
validation or an official comparison against original VALLR.
