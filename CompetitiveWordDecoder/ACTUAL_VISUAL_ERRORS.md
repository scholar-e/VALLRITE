# Actual visual-error continuation pilot

2026-10-14. Continue the retained clean EOS Llama adapter, without overwriting it.
This tests transfer from older visual-model errors to the newer context model.

Data: `datasets/decoder-training/lrs3-full-forced-direct-train-5k/phone-hypotheses.jsonl`.
Every source ID must occur in the LRS3 training manifest and must not occur in
the matched 1,140-clip development reference. Remove exact normalized transcript
duplicates, including matches to that development reference. Shuffle eligible
clips with seed 42; select 1,000 for optimization and 100 distinct clips for
internal loss evaluation. Each contributes one clean reference-phone prompt and
one actual rank-one visual-phone prompt, both targeting its transcript. No word
boundaries or reference information are added to noisy prompts.

All 2,000 training and 200 internal-validation pairs fit 512 tokens including
supervised EOS. Prompt and padding loss are masked. One epoch gives 500 updates
at batch four, learning rate 1e-5 with 3% warmup and linear decay, LoRA rank16,
alpha32, dropout0.05, BF16, gradient checkpointing, and weight decay0.01. Save
resume checkpoints at steps250 and500; retain both. Thermal protection and root/
run PAUSE markers remain active; cap training at15 minutes excluding preparation
and evaluation. No early selection based on external WER is performed.

The older visual model saw this training corpus; these predictions are not
out-of-fold. The internal split is held out from this decoder continuation, not
from visual-model training, and is clip-level rather than speaker-level.
Transfer mismatch and in-sample errors limit conclusions. No claim of a clean
causal comparison against synthetic corruption is intended: training corpus,
data size, and precision differ from earlier synthetic experiments.

Training artifact: `checkpoints/vallr-actual-errors-20261014`. Manifest records
source/reference/training-manifest hashes, selected IDs, encoded-row hashes,
initial adapter hash and optimizer settings. Inherited `duplicates_removed`
describes the unused WikiText preparation, not this visual-error filtering;
the `visual_errors` field and split hashes identify the actual training data.

Planned evaluation: unchanged `evaluate_llama_visual` on all context-model
five-candidate outputs, then the same clips with reference phones as a clean
regression diagnostic. Compare to95.48% visual WER/92.53% oracle WER@5 and13.40%
reference-phone WER. Oracle scores are reference-aware bounds. A visual gain
must be weighed against degradation on clean phones. No deployment promotion
is automatic, and no fresh held-out test is being consumed here.

## Results

Completed all500 updates and one epoch; saved trainer state confirms completion.
Training runtime280.8 seconds, internal validation loss2.084. Cooling pauses
occurred and resumed automatically; thresholds were not relaxed. Twelve existing
decoder tests pass, as do dataset split assertions and whitespace checks.

| Matched development metric | Clean EOS baseline | Actual-error continuation |
| --- | ---: | ---: |
| Context visual-input top-ranked WER | 95.48% | 97.22% |
| Context visual-input oracle WER@5 | 92.53% | 91.53% |
| Reference-phone WER | 13.40% | 9.31% |
| Capped visual generations /5700 | 11 | 4 |

Visual top-ranked errors rise from12,102 to12,323 (+221). Oracle errors fall
from11,728 to11,602 (-126), with one exact sentence somewhere in five. The
reference-phone diagnostic improves, but is not a lipreading result. This
suggests useful decoder/domain adaptation without solving the noisy-input
top-ranked output problem; no causal isolation of domain and error adaptation
was performed. Both evaluations use all1,140 clips and12,675 reference words.

Do not promote this adapter for the visual pipeline: its actual top-ranked WER
regresses. Preserve it as a decoder candidate and retain the clean EOS baseline.
Next proposed check is fixed probability ranking of these existing candidate
sentences, using the prior scoring rule without tuning to references. If that
does not recover deployable WER, generate training errors from the current
context model on an appropriate held-out/out-of-fold training partition rather
than extending this mismatched pilot blindly. No additional run was launched.

Evaluation artifacts: `visual-eval/metrics.json`, `reference-eval/metrics.json`,
and their manifests and generated sentences under the training artifact folder.
