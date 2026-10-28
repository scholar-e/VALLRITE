# Probabilistic phone-path sentence ranking

## Current context-model comparison, 2026-10-28

The same fixed rule was evaluated on cached context-model sentences from the
retained clean EOS adapter and the current-error mixed adapter. Each adapter
scored its own sentences. No training, candidate generation, weight fitting or
reference-dependent selection was performed. Both runs used batch size 4,
thermal protection, PAUSE checks and a ten-minute process cap.

| Adapter | Unranked WER (errors) | Fixed probability-ranked WER (errors) | Mean-score diagnostic WER | Unchanged oracle WER@5 |
|---|---:|---:|---:|---:|
| Retained clean EOS | 95.48% (12,102) | 95.29% (12,078) | 95.47% | 92.53% |
| Current-error mixed | 95.23% (12,070) | 93.29% (11,824) | 95.12% | 90.12% |

Both contain the same 1,140 clips and 12,675 reference words. Every cached
candidate string was checked against its original report, and baseline and
oracle error counts match exactly. The five ranker unit tests pass.

The primary rule removes 246 errors (1.94 percentage points) for mixed and
24 errors (0.19 points) for clean. Mixed plus ranking is 2.19 points better
than the original unranked clean baseline and 2.00 points better than ranked
clean. Retain the fixed rule as a promising experimental configuration; this
single-seed, repeatedly inspected development result is not confirmation on an
independent test set. Even perfect selection from the mixed pool leaves 90.12%
WER, so the next priority is candidate quality, not tuning ranking weights on
these references. No additional experiment was launched.

Artifacts: `checkpoints/context-baseline-ranked-20261028` and
`checkpoints/context-mixed-ranked-20261028`, each with manifest, per-candidate
scores, metrics and logs. Both evaluations finished in under one minute.

## Frozen pilot, 2026-10-12

This validates a low-cost probability-aware selection pipeline before changing
the visual model. It reuses five concrete ARPAbet paths per clip from the older
157,849-parameter gated-fusion epoch-3 model and the exact cached sentences from
the clean EOS Llama decoder. Neither model is trained. No reference transcript
or reference phone sequence enters the ranking decision.

Each generated sentence is scored under its own phone prompt using the frozen
decoder's conditional token log probabilities. EOS is scored for terminated
outputs; capped outputs score their observed prefix only. Saved text is
retokenized, rather than claiming to recover its original generation token IDs.
The primary rule adds the phone-path beam log score and summed decoder token
log probability, with fixed weight 1. Identical normalized sentences from
different phone paths have their scores combined with log-sum-exp.

The phone scores already include a phone bigram prior with weight 0.5. They
are not pure visual CTC probabilities. Normalized masses are relative only to
the retained candidate set and are not calibrated correctness probabilities.
A separately predeclared diagnostic uses mean decoder token log probability
instead of its sum; it is a length-normalized heuristic, not a joint probability.
No weights were fit or selected using validation references. References enter
only after sentence selection, for WER and oracle scoring.

## Result

All 1,140 development-validation clips, 5,700 candidates, and 12,675 reference
words match the previous cached evaluation. Runtime was about 54 seconds on the
host GPU, including loading, with thermal protection and PAUSE checks.

| Fixed selection rule | Word errors | WER |
| --- | ---: | ---: |
| Original highest-ranked phone path | 12,623 | 99.59% |
| Path score + summed decoder log probability | 12,478 | 98.45% |
| Path score + mean decoder log probability | 13,291 | 104.86% |
| Reference-selected oracle among the same five sentences | 12,224 | 96.44% |

The primary rule removes 145 word edits, a 1.14 percentage-point improvement.
That is a small ranking gain, not useful lipreading accuracy. The unchanged
96.44% oracle WER shows that this candidate pool is fundamentally weak; ranking
alone cannot recover sentences absent from it. These are older-model visual
inputs, not reference-family inputs or an official VALLR benchmark. Do not
compare this table directly with the reference-family reconstruction pilots.

No new per-member family probabilities have been learned. A subsequent model
with an individual-phone head (and optional auxiliary family loss) is still
needed to obtain those probabilities from the newer visual encoder.

## Artifacts and reproduction

Module: `CompetitiveWordDecoder.evaluate_probability_ranker`.
Run: `checkpoints/vallr-probability-ranker-20261012` contains `manifest.json`,
`scores.jsonl`, `metrics.json`, and logs. Source, adapter, and cached report
hashes are checked before scoring. Unit tests cover fixed evidence combination,
duplicate-sentence aggregation, normalized mass, and reference-independent
selection. The saved baseline and oracle counts were verified to match the
original cached decoder report exactly.

```bash
tools/with-vpa-gpu .venv-qwen/bin/python -m CompetitiveWordDecoder.evaluate_probability_ranker \
  --hypotheses datasets/decoder-training/lrs3-full-forced-direct-validation/phone-hypotheses.jsonl \
  --decoder-report checkpoints/vallr-visual-eos-20261011/metrics.json \
  --adapter checkpoints/vallr-eos-20261010/adapter \
  --output checkpoints/NEW-PROBABILITY-RUN --batch-size 4
```
