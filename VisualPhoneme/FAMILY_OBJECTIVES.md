# Probabilistic family-loss sweep

2026-10-13. Follow-up to [MULTIFRAME_CONTEXT.md](MULTIFRAME_CONTEXT.md).

## Predeclared protocol

Keep the nine-frame residual phone head, frozen pretrained visual backbone,
initial step-100 phone checkpoint, 4,000 full training clips, seed 42, batch eight,
and 1,140-clip development evaluation fixed. Each arm runs 1,000 updates with
checkpoints every 250 updates. Learning rate follows cosine decay from 3e-4 to
3e-5; this schedule depends only on step, not validation performance. The final
step is the primary comparison; best-checkpoint scores are secondary and
development-selected. Each arm has a five-minute training-loop cap with the
existing thermal/pause/resume protections. Arms run sequentially, not concurrently.

Test four conditions: no added family loss; existing 17 families; a finer
31-family partition; and a coarse 10-family partition. All predict the identical
39-phone vocabulary plus CTC blank. No family partition changes the reference
used for exact PER. These are hand-specified hypotheses, not empirically proven
equivalence classes and not partitions learned from validation confusions.

Fine groups retain only B/P, F/V, DH/TH, D/T, S/Z, CH/JH, SH/ZH, G/K as pairs;
every other phone is a singleton. Coarse groups are B/M/P; F/V;
DH/TH/D/N/T/S/Z; CH/JH/SH/ZH; G/K/NG; L/R/ER; UH/UW/W/AW/OW/OY;
IH/IY/Y/AY/EY; AA/AH/AO/AE/EH; HH. The original partition remains in
`visemes.py`; the new experimental definitions do not modify it globally.

Family probabilities are sums of normalized phone probabilities, implemented
with log-sum-exp. CTC blank stays its own class. Map target phones tokenwise to
families, preserving adjacent repeats. Add 0.1 times this grouped CTC loss to
exact-phone CTC. This gradient reaches the phone head; unlike the retained
frozen source family-head loss, it is not constant. Grouped CTC is an auxiliary
objective, not mathematically identical to marginalizing all exact-phone CTC
sequences: merging labels changes CTC repeat/blank constraints.

Primary optimization target: lower exact-phone PER at the final matched step.
Secondary measures: checkpoint curve and PER under the **fixed original** family
mapping. Do not rank conditions by their own differently sized family inventories.
Changing the auxiliary grouping cannot directly change which exact-phone
prediction is counted correct. No WER gain or calibrated confidence is assumed.

The cosine/no-family arm isolates scheduling relative to the preceding
constant-rate context run. The other three isolate their family regularizer
relative to cosine/no-family. This sweep does not include the previously proposed
parameter-matched framewise head or fresh held-out testing; those remain separate
checks. It also does not retrain the pretrained family encoder from scratch, so
all arms inherit the same original-family representation.

## Artifacts and reproduction

Run directories: `checkpoints/context-family-{none,original,fine,coarse}-20261013`.
Use `VisualPhoneme.train_phone_head` with the previous context experiment's
arguments plus `--lr-schedule cosine --phone-family NAME
--phone-family-weight 0.1 --max-minutes 5`. Exports retain the explicitly tagged
context architecture. Resume with the same arguments and `--resume`; the schedule
is reconstructed from the saved update count. Changing the planned number of
steps changes the schedule and is rejected by the manifest check.

Tests verify partition coverage, class counts, conserved probability mass,
separate blank probability, and nonzero gradients, in addition to the six
existing phone-head tests. Seven tests pass before training. No new dependencies
are needed. No baseline checkpoint is overwritten.

## Results

All four arms completed 1,000 updates and final evaluation. Manifests match
except for family-objective fields. Frozen encoder/family-head equality checks
pass in every arm. Recorded loop/final-evaluation time is 65–76 seconds per arm,
excluding preparation and initial evaluation.

| Auxiliary objective | Step 250 exact PER | Step 500 | Step 750 (best) | Step 1,000 (primary) |
| --- | ---: | ---: | ---: | ---: |
| None | 86.1436% | 85.0760% | 83.5823% | 84.0062% |
| Original, 17 families | 85.9238% | 84.7800% | 83.3468% | 83.7684% |
| Fine, 31 families | 86.0562% | 84.9751% | 83.4903% | 83.8783% |
| Coarse, 10 families | 85.9104% | 84.7351% | 83.2840% | 83.7101% |

All start at 87.7540% exact PER. Final phone-prediction PER under the fixed
original family mapping is respectively 80.9335%, 80.5522%, 80.7563%, and
80.5073%. The separate frozen family head remains at 71.9822% throughout.

The coarse objective wins numerically: 37,323 final phone errors versus 37,455
without family loss, a reduction of 132 edits / 44,586 phones (0.2961 percentage
points). Its advantage over the original grouping is only 26 edits (0.0583
points). A single seed does not establish a robust winner. Auxiliary objectives
also change gradient scale; this is not a pure test of grouping semantics.

Cosine scheduling improves the final no-family result from the prior constant
rate's 84.6297% to 84.0062%, but none beats the prior best saved context
checkpoint at 82.6717%. All four curves regress between 750 and 1,000 updates.
Decreasing the learning rate alone has not removed that pattern.

Retain `fullclip-phone-context-20261013/phones-step-0750.pt` as the best observed
exact-PER candidate; no deployed model is changed. The coarse arm's best export
is `context-family-coarse-20261013/phones-step-0750.pt`. No WER evaluation or fresh
held-out confirmation was performed. Recommended next priorities are evaluation
through the unchanged word decoder, investigation of the later-step regression,
and a parameter-matched framewise-head control. Compare original/coarse families
across seeds before undertaking a larger grouping search. These are proposals,
not additional launched experiments.

This is a development sweep;
selection among four alternatives introduces additional selection bias and
requires confirmation on fresh data before a generalization claim.
