# Controlled phoneme-error continuation

Starts from `checkpoints/vallr-eos-20261010/adapter`; does not alter that baseline.
One epoch, learning rate 1e-5, effective batch four (physical batch4, accumulation1),
60-minute training cap, checkpoints every250 updates with the latest two retained.
EOS supervision and complete-pair length filtering remain enabled.

Each retained training transcript appears once. Exactly floor(N/2) rows receive
synthetic errors, the rest remain clean. A deterministic hash assigns noisy rows
5%, 10%, or 20% nominal edit budgets with 50/40/10 weighting. Edit types use
60% substitution, 25% deletion, 15% insertion. Substitutions are uniform among
the other CMU phones, not calibrated visual confusions. Very short inputs receive
at least one edit; actual phone edit distance is recorded, rather than assuming
nominal rate equals PER. Tags, targets and UNK(word) spelling remain untouched.
Both clean and corrupted complete pairs must fit512 tokens including EOS.

Validation uses the same100 deterministically selected examples under three
conditions: clean, 10%, and20% nominal corruption. All variants must fit the
length budget; generation cap256, greedy decoding, identical normalization to
the clean experiment. Validation error seeds differ from training seeds. Baseline
WER is measured before optimization and continuation WER after the epoch. The
test split is not used for scoring in this pilot. Unknown spelling leakage is
retained and dictionary-only scores are reported separately.

`synthetic-data.json` saves exact validation inputs, training-data hash, counts,
and measured input PER. `synthetic-results.json` combines before/after WER.
Per-condition predictions and scores live under `baseline-*` and `final-*`.
Files named `clean-test-*` are inherited from the shared evaluator; their split
metadata correctly says validation for this experiment.

```bash
tools/with-vpa-gpu .venv-qwen/bin/python -m CompetitiveWordDecoder.train_vallr_clean \
  --output checkpoints/vallr-noise-20261010 \
  --initial-adapter checkpoints/vallr-eos-20261010/adapter \
  --end-of-answer --synthetic-noise --epochs 1 --learning-rate 1e-5 \
  --batch-size 4 --gradient-accumulation 1 --test-limit 100 --max-minutes 60
```

Pause: `./pause-training checkpoints/vallr-noise-20261010`. Remove the intended
marker and repeat the command with `--resume` to restore the optimizer/checkpoint
state. If paused during the initial baseline evaluation, no training checkpoint
exists yet: preserve that partial directory and start in a new output directory.
The cap applies to training, not the before/after evaluations. No thermal watchdog
has been added. Do not automatically promote this model or proceed to visual
errors without reviewing both robustness and clean-WER changes.

## 75/25 comparison

`checkpoints/vallr-noise75-20261010` uses the same command and original EOS
adapter, with `--clean-fraction 0.75` and the new output directory. All other
settings stay fixed. The default remains 0.5, retaining compatibility with the
first run. The eligible transcript pool and evaluation variants are unchanged;
the 25% noisy rows are a deterministic subset of the earlier 50% noisy rows.
Counts round down for the noisy portion. The manifest records the fraction and
resume rejects a mixture change. Monitor the detached service with:

```bash
journalctl --user -u vallrite-vallr-noise75-20261010.service -f
```

## Frozen-decoder evaluation on actual visual predictions (2026-10-11)

`CompetitiveWordDecoder.evaluate_llama_visual` evaluates saved Llama adapters on
ranked visual ARPAbet hypotheses using the trained tag prompt (no chat template
or extra BOS), greedy decoding, EOS termination, a256-token generation cap,
and lowercase word/apostrophe normalization matching the earlier direct decoder
evaluator. No references are provided to generation. Oracle@5 is the minimum
word edit count across five decoded visual hypotheses, scored with references;
it is not a deployable selection method.

Both models used the same1,140 LRS3 development-validation clips and12,675 words
from `datasets/decoder-training/lrs3-full-forced-direct-validation/phone-hypotheses.jsonl`.
These are stored predictions from the older gated-fusion epoch3 checkpoint
`lrs3-full-forced-finetune-hybrid-gpu-20260913`, not the newer grouped-phone
encoder. The exported input oracle PER@5 is78.14%. Source hashes, adapter hashes,
normalization, per-hypothesis outputs and errors are recorded in each report.

| Frozen decoder | Top-1 WER | Oracle WER@5 | Capped generations /5,700 |
| --- | ---: | ---: | ---: |
| Original EOS baseline |99.59% |96.44% |24 |
| 50/50 synthetic continuation |103.76% |95.87% |18 |

Neither model produced an exact oracle word sequence on this set. The synthetic
continuation improved oracle WER by only0.57 points while worsening top-1 by4.17
points. Do not promote it as a better visual-input decoder on this evidence.
Reports are `checkpoints/vallr-visual-eos-20261011/metrics.json` and
`checkpoints/vallr-visual-noise50-20261011/metrics.json`.

## Reference-phone diagnostic (2026-10-11)

Completed the same evaluation with `--reference-phones`: one greedy generation
from each stored `reference_phones` sequence, with no transcript words passed to
the decoder. All 1,140 clips and 12,675 normalized reference words match the visual
evaluation, verified by source hash and ordered clip/reference pairs. Prompt,
adapters, generation cap, and scoring are unchanged. No training was performed.

| Frozen decoder | Reference-phone WER | Exact clips | Capped /1,140 | Visual top-1 WER |
| --- | ---: | ---: | ---: | ---: |
| Original EOS baseline | 13.40% | 502 (44.04%) | 0 | 99.59% |
| 50/50 synthetic continuation | 18.88% | 467 (40.96%) | 2 | 103.76% |

Reports: `checkpoints/vallr-reference-eos-20261011/metrics.json` and
`checkpoints/vallr-reference-noise50-20261011/metrics.json`. Word error counts are
1,699 and 2,393 respectively. GPU generation/scoring took 57 and 77 seconds.
There is no oracle@5 for this single-input diagnostic. Stored reference labels
are treated as targets, not independently verified error-free audio annotations;
these numbers are not end-to-end visual or official VALLR test WER.

The large reference-versus-visual gap strongly implicates the older visual phone
predictions as the main bottleneck on this set. Residual reference-phone WER can
reflect ambiguity, label conventions, and decoder/domain errors; this experiment
does not distinguish those causes. The synthetic continuation also worsens
reference-phone WER by 5.48 percentage points. Retain the clean EOS baseline.
Next recommended work is evaluating the latest visual encoder on this matched
split and, if adapting the decoder, using real training-split visual confusions
instead of uniformly sampled phone edits. Keep this validation split out of
training and use an untouched test split for final confirmation.

## Matched visual-family comparison (2026-10-11)

The latest retained visual checkpoint predicts 17 families plus REST, not the
39 individual ARPAbet phones expected by the Llama decoder. A direct decoder WER
test would therefore require an explicit family-aware decoder or probabilistic
phone expansion. No arbitrary representative phone was substituted for a family.
Instead, `VisualPhoneme.evaluate_matched_groups` compared three frozen visual
checkpoints on the exact 1,140 full validation clips from the preceding tests.
Reference phone sequences were checked against the manifest before inference.

| Visual checkpoint | Parameters | Greedy family error rate |
| --- | ---: | ---: |
| Older hybrid, epoch 3 | 157,849 | 78.11% |
| Pretrained large image, epoch 25 | 9,258,994 | 71.98% |
| REST continuation, epoch 9 | 9,259,379 | 74.86% |

All share 44,586 reference tokens. Scoring applies greedy CTC collapse first,
maps individual phone emissions to families where necessary, removes REST, and
does not merge adjacent equal families. These are family-token error rates,
not exact-phone PER, oracle@5, or word error. Full clips and transcript/CMUdict
labels differ from the short forced-aligned chunks used in earlier REST tests;
the apparent REST gain there does not establish a full-clip gain here.
The pre-REST image model improves by 6.13 percentage points over the older
hybrid on this matched family metric, but remains highly error-prone.

Results and checkpoint hashes:
`checkpoints/visual-matched-families-20261011/metrics.json`.
The run used batch size 8, four loader workers, the GPU launcher, thermal guard,
and PAUSE checks; no checkpoint was changed or trained. The failed October 2
distillation pilot was not promoted and is not treated as the preferred model.

Original VALLR comparison: its paper reports 18.7% end-to-end LRS3 WER,
not a directly comparable sequence PER. The saved local diagnostic at
`evaluation/original-vallr-vs-vallrite-lrs3-word-context.json` reports 98.42%
exact-phone PER on 240 forced-aligned isolated words (822 reference phones),
and family errors of 95.50% for original VALLR versus 94.28% for our REST model.
That word-segment test uses 16 frames and the original checkpoint's eight CTC
steps; it is not the paper's sentence pipeline or a matched comparison with the
1,140-clip table above. A locally reproduced original-VALLR advantage has not
been established. Source: https://arxiv.org/html/2503.21408v2 .
