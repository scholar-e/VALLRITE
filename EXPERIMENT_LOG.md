# Experiment log

Dated results moved from [FUTURE_TRAINING_PLAN.md](FUTURE_TRAINING_PLAN.md) on
2026-10-29. Newest first. Entries are records, not current recommendations;
see the plan for what to do next.

## Dated results

2026-10-29 timed input pilot: [numeric evidence adapter](CompetitiveWordDecoder/TIMED_PHONE_INPUT.md)
is implemented and trained with both upstream models frozen. The initial
coordinate-wise bound allowed residual vectors comparable to the original
embeddings and regressed validation WER. A5%-of-embedding vector bound reduces
that regression to93.68% versus93.31% baseline. Training from zero under that
bound is also complete: 93.53% at the final checkpoint versus 93.31% baseline,
three additional word errors. Baseline remains selected; no accuracy gain or
evaluation-set promotion is claimed. Full-frame uncertainty and missing-phone
positions remain limitations of the per-predicted-phone input summaries.

2026-10-29 rolling representation: [timed phone distributions](VisualPhoneme/ROLLING_PHONES.md)
are implemented and tested. Plain three/five-frame averages worsen validation
PER to 73.43%/76.66%; preserving blank mass yields 72.85%/74.85%, versus 70.60%
without smoothing. Retain unsmoothed timed distributions and test a decoder that
actually consumes their uncertainty; do not deploy uniform smoothing as an
accuracy improvement. No new decoder training was run in this follow-up.

2026-10-29 architecture audit: [sequential corrections and controlled continuation](CompetitiveWordDecoder/ARCHITECTURE_AUDIT.md)
fix exact generated-token scoring, add validation-WER selection with a baseline
candidate, and remove a detached family loss from optimization accounting.
The matched smaller-learning-rate run completes 375 updates but loses on
validation: 113.09%/113.46% ranked WER at steps 250/375 versus 93.46% baseline.
It is rejected without another evaluation-set pass. Next investigate early
adaptation drift and retention regularization; batch padding and timing/confidence
loss remain separate unresolved architectural issues.

2026-10-29 matched adaptation completed: the [source-disjoint pilot](CompetitiveWordDecoder/MATCHED_ADAPTATION.md)
regresses ranked WER from 95.11% to 128.95% (clean) and 126.53% (mixed) on the
same new 100-clip evaluation sample. Both complete 375 updates. Insertions and
substitutions increase sharply; neither adapter is promoted. Do not scale this
recipe. Prioritize validation-WER monitoring, smaller updates and retention of
the starting decoder's behavior before another continuation. These new-split
scores must not be compared directly with the older 1,140-clip scores.

2026-10-29: [paired error analysis and matched adaptation preparation](CompetitiveWordDecoder/MATCHED_ADAPTATION.md)
show that recovered phone length trades deletions for substitutions. The previous
decoder continuation used only eight videos, shared across its internal splits.
A broader 750/100/100 source-video-disjoint full-clip split is frozen and verified;
matched phone export and controlled decoder continuation are now complete.
Its historical pretraining exposure is not fully audited, so it is not an
untouched end-to-end test set.

Fixed-crop follow-up completed 2026-10-29: [matched continuations and evaluation](VisualPhoneme/FIXED_CROP_FOLLOWUP.md)
reach 68.70% beam PER / 67.16% oracle PER@5 and 95.64% ranked WER. Neither
the best existing phone result nor the retained 93.29% word pipeline improves.
Greedy checkpoint selection disagrees with beam selection despite improving
calibration CTC loss. Use the intended decoder for future calibration selection;
prioritize paired error analysis and matched noisy-phone word-decoder adaptation
on a fresh evaluation split before committing to longer runs of this recipe.

2026-10-28 four-strategy update: [all blank-mitigation pilots](VisualPhoneme/BLANK_STRATEGIES.md)
are complete. Separate beam calibration reaches 67.71% PER / 66.13% oracle
PER@5, but ranked WER worsens to 97.59%. Longer fixed-crop training reaches
73.69% greedy PER without a WER gain; matched landmark crops, weak expected
count loss and filtered teacher supervision do not improve the retained word
pipeline. Preserve the 93.29% ranked baseline and the negative results. Do not
extend these exact loss settings or silently combine changes based on PER alone.

2026-10-28 system audit: [blank calibration and input-distribution analysis](VisualPhoneme/SYSTEM_AUDIT.md)
improves exact greedy PER from 82.67% to 68.85% using training-side calibration,
but the same adjustment fails for beam search. Standardize mouth crop policy
and increase sentence-length training coverage before expanding capacity.
Treat greedy and beam calibration separately; retain the current word pipeline
until matched WER and independent confirmation support replacement.

2026-10-28 candidate audit: [removing the phone prior](CompetitiveWordDecoder/PHONE_CANDIDATE_AUDIT.md)
reduced rank-one beam PER from 82.12% to 70.82% and oracle PER@5 to 69.29%.
However, mixed-decoder ranked WER was 93.33% versus 93.29% with the prior.
Keep the current pipeline. Test clean-decoder compatibility with these new
candidates before adapting the decoder or expanding visual training.

2026-10-28 update: [fixed probability ranking](CompetitiveWordDecoder/PROBABILITY_RANKING.md)
is complete for the retained and mixed adapters. Ranked WER is 95.29% and
93.29%, respectively, with unchanged candidate-set oracles. Keep mixed plus
fixed ranking as an experimental candidate; prioritize candidate quality next,
since its oracle still has 90.12% WER. No ranking weights were tuned.

2026-10-27 update: the [matched current-error pilot](CompetitiveWordDecoder/CURRENT_ERROR_COMPARISON.md)
is complete. Mixed training reached 95.23% selected visual WER versus 95.48%
retained baseline and 95.74% clean-only control; oracle WER@5 reached 90.12%.
The small single-seed development gain does not yet justify promotion. Keep the
baseline and prioritize fixed-candidate ranking before a larger training run.

Update: the [decoder contract audit and tolerance curves](CompetitiveWordDecoder/DECODER_TOLERANCE.md)
are complete. No simple word-separator mismatch was found; both decoders are
fragile to phone errors, especially the latest adapted model. Use the retained

## Superseded run plan (2026-10-26)

The staged plan below guided the 2026-10-26 to 2026-10-29 experiments. Its
stages 0–3 were executed (see entries above). It is retained for provenance.


Start with fixed candidate ranking and data/split checks. Then run a small
matched decoder experiment and a separate visual-head ablation. Scale only the
branch that improves actual selected WER without unacceptable regressions.
Do not spend hours extending the latest adapter or searching more family groups
until these smaller experiments explain which component helps.

The user approves each stage's scope and total budget before launch. Budgets
below are proposed ceilings, not runtime forecasts or authorization to run.
The near-term goal is reproducible improvement, not a promised target WER.

### Baselines to preserve

Results below use the repeatedly inspected 1,140-clip development set, with
44,586 reference phones and 12,675 words. They are not fresh test results.

| System / diagnostic | Result | Role |
| --- | ---: | --- |
| Context visual head, step 750 | 82.67% greedy exact PER | Best observed selected visual checkpoint |
| Same visual model, five phone candidates | 79.96% oracle PER@5 | Candidate-quality diagnostic, not deployable selection |
| Context model + clean EOS word decoder | 95.48% WER; 92.53% oracle WER@5 | Current selected-output comparison baseline |
| Context model + actual-error-adapted decoder | 97.22% WER; 91.53% oracle WER@5 | Experimental candidate; not promoted |
| Clean reference phones + clean EOS decoder | 13.40% WER | Decoder-only diagnostic |
| Clean reference phones + adapted decoder | 9.31% WER | Decoder/domain gain, not lipreading performance |

Preserve `checkpoints/fullclip-phone-context-20261013/phones-step-0750.pt`,
`checkpoints/vallr-eos-20261010/adapter`, and
`checkpoints/vallr-actual-errors-20261014/adapter`. Never overwrite them.
Retain the older hybrid and all manifests as historical controls.

Evidence: [context experiment](VisualPhoneme/MULTIFRAME_CONTEXT.md),
[family sweep](VisualPhoneme/FAMILY_OBJECTIVES.md),
[word evaluation](VisualPhoneme/CONTEXT_WORD_EVALUATION.md), and
[actual-error continuation](CompetitiveWordDecoder/ACTUAL_VISUAL_ERRORS.md).

### Ordered run plan

#### 0. Reproducibility and data audit — no training

**Budget:** up to 30 minutes of checks; no new dataset downloads.

Verify baseline artifacts, hashes, current evaluator compatibility, selected clip
IDs and source-video/speaker overlap. Inspect representative failures from the
existing development outputs: empty/short phones, repeats, substitution-heavy
sequences, word insertions and capped generations. Produce counts by clip length
and error type, not just memorable examples. Distinguish pipeline defects from
genuinely weak recognition.

Define future train/selection/confirmation manifests and audit upstream exposure.
If no genuinely unused data is available, report that rather than inventing a
holdout guarantee. This stage can uncover indexing, cropping, or reference bugs;
fix and rerun the affected baseline before comparing new training.

**Exit:** reproducible baseline, usable split plan, safety smoke test and a short
error breakdown. Stop subsequent stages if these checks fail.

#### 1. Fixed probability ranking — cheapest next experiment

**Budget:** two evaluations, up to 10 minutes each including model loading;
no training and no new candidate generation.

Run the existing fixed ranking rule on cached context-model candidates from
both the clean and adapted decoders. Use each adapter to score its own generated
sentences. Keep the previously declared path-score plus summed decoder log
probability rule, EOS handling and duplicate aggregation. Do not tune weights
or length penalties on evaluation references. Relative scores are not calibrated
confidence; the path score already includes a phone language model.

Compare actual selected WER against both each adapter's unranked output and the
95.48% clean-decoder baseline. Candidate-set oracle WER must remain unchanged.

**Decision:** retain a ranking rule only if it improves actual output. Oracle
WER of roughly 92% shows ranking alone cannot make this pool good. If ranking
fails, preserve the result and move to candidate quality rather than tuning
many score weights on the same set.

#### 2. Training predictions matched to the current visual model

**Budget:** initial data/export pilot up to 30 minutes; any fold retraining needs
a separately approved budget.

Produce current-model phone predictions on training-side clips, keeping targets
separate from inference. Save logits or sufficient scores, greedy phones,
five-best phones, checkpoint hashes, crop policy and beam configuration. Audit
error rates and length distributions against the deployment-like selection set.

Prefer clips excluded from every training stage of the visual model. If that is
impossible, use cross-fitting: each fold's predictions come from a visual model
trained without that fold. Holding out clips only from phone-head adaptation
does **not** remove exposure through pretrained visual weights. When full
cross-fitting is too expensive, explicitly label the export in-sample and treat
it as a pilot with limited evidence, not an unbiased error distribution.

**Exit:** verified provenance, no decoder evaluation-target leakage, and an
eligible 1,000-clip pilot pool. Do not scale to the entire corpus yet.

#### 3. Matched decoder adaptation — isolate domain and noise effects

**Budget:** three arms, each at most 15 minutes training plus separately budgeted
evaluation. Start with one seed; replicate only a promising configuration.

Start every arm from the same clean EOS adapter. Use identical transcript pools,
eligible lengths, total presentations, optimizer, precision and update count:

1. Clean reference-phone continuation: domain-adaptation control.
2. 50% clean / 50% current-model visual-phone inputs.
3. 75% clean / 25% current-model visual-phone inputs.

Use deterministic assignments, target-only loss and explicit EOS. Keep rank-one
input first; introducing all five candidates or a new prompt is a separate
ablation. Pilot around 1,000 clips and 500 matched updates, with learning rate
1e-5, checkpoints every 100–250 updates. These are starting settings, not claims
of optimality. Select using the new selection split, then evaluate the frozen
choice with identical generation/ranking settings.

**Gate:** actual visual WER must improve over both the retained baseline and
the clean-domain control. Proposed practical threshold: at least 1 absolute WER
point on selection data, with no more than 1 point clean-phone regression versus
the retained decoder. Confirm across three seeds before larger investment.
These thresholds are proposed decision rules, not guarantees of significance.
If only reference-phone WER improves, retain that as a diagnostic—not a visual
pipeline success.

#### 4. Visual context versus capacity — controlled PER experiment

**Budget:** two heads × three seeds; at most 10 minutes training per run, with
evaluation overhead budgeted separately. Run this independently of decoder changes.

Compare the current nine-frame residual context head with a framewise nonlinear
head matched as closely as possible in parameter count. Keep full training clips,
initial linear weights, frozen encoder, losses, batch size and update order fixed
within each seed. Report exact parameter counts and FLOPs; matching parameters
does not automatically match compute.

Train through a predeclared 1,000-update horizon and save every 100 updates. Select
checkpoints on the separate selection split; show the complete curve. All recent
curves regressed after step 750, so measure training/selection loss, blank rate,
output length and edit breakdown before calling that overfitting. Choose any
scheduler on selection data only and apply it equally to both heads.

**Gate:** repeatable exact-PER gain plus non-worsening actual WER under the frozen
decoder. A proposed practical PER threshold is 1 absolute point relative to the
matched control. If context does not beat matched nonlinear capacity, keep the
simpler head and stop expanding temporal windows.

#### 5. Representation adaptation or teacher supervision — conditional

**Budget:** initially two arms capped at 30 minutes each; no overnight run.

Only after stages 2–4 identify a plausible representation bottleneck, compare a
frozen baseline with the last temporal block unfrozen. Start with a conservative
encoder learning rate (for example 1e-6) and the selected head rate. Keep family
retention diagnostics, exact PER and frozen-decoder WER. Expand unfreezing only
if the small change helps without destabilizing the representation.

If revisiting audio supervision, first audit 30–50 training-side clips manually
for stream offsets and phone timing. Earlier labels cover only 27.4% of frames;
44.4% of inspected intervals contain no frame center. Do not relabel missing
frames as silence or stretch sparse acoustic spikes blindly. Test confidence-
masked/windowed soft supervision against matched CTC-only training. Audio may
teach training targets but must not become a hidden inference input.

**Gate:** improvement beyond the matched no-teacher/frozen control. A larger
teacher, new architecture or new download is a separate approval decision.

#### 6. Scale the proven configuration, then confirm once

**Budget:** propose a two-hour resumable run first. Extend to six hours only
after inspecting selection curves and thermal logs; neither is authorized here.

Increase data in stages, e.g. 1k → 5k → 20k eligible clips, rather than adding all
available material at once. To attribute gains to data, compare equal-update
budgets; report equal-epoch results separately because they spend more compute.
Track quality filters and corpus composition at each size. More weak labels can
amplify an error rather than cure it.

Use the established checkpoint-selection/stopping rule, bounded total training
time, preserved best checkpoint and resumable optimizer. Evaluate the locked
winner on the reserved confirmation set once. Report actual WER, exact PER,
oracle gaps, uncertainty, parameter count, memory and measured latency. Do not
claim phone/iPhone deployment readiness from desktop GPU speed. Deployment
conversion and device benchmarking are a separate phase.

### What not to prioritize now

| Option | Why defer it | Reconsider when |
| --- | --- | --- |
| Broader family search | Ten versus 17 families differed by only 26 edits; none beat 82.67% PER | Repeated-seed evidence supports a useful regularizer |
| Longer latest decoder run | Actual visual WER regressed despite better clean decoding | Matched-error pilot beats clean-domain control |
| Larger visual transformer immediately | Changes capacity, training recipe and compute together | Small representation ablation shows a capacity bottleneck |
| Full corpus / overnight training | Expensive before label and distribution issues are resolved | Controlled gains survive replication |

Keeping the current baseline is a legitimate outcome. All alternatives remain
reversible because candidate checkpoints and data manifests are versioned
separately. Do not delete negative results or overwrite raw data.

### Independent review

Attempted on 2026-10-26 using:

```bash
.venv-vpa-gpu/bin/python /home/elinzi/.codex/skills/write-design-doc/scripts/review_design.py \
  --design FUTURE_TRAINING_PLAN.md --output future-training-plan.feedback.md \
  --launcher /Users/haha/github/ai-skills/bin/claude_deepseek.py
```

The required launcher is absent; no independent critique was produced. The
[attempt log](future-training-plan.feedback.md.log) records the failure.

| Item | Disposition | Reason |
| --- | --- | --- |
| Independent technical review | Deferred | Reviewer infrastructure unavailable; no findings to incorporate |

This plan is based on repository evidence and the main agent's review, not an
independent endorsement. Resolve this limitation before a large training budget.
