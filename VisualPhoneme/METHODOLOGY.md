# Visual phonemes and sentence decoding: methodology and decision record

Updated 2026-10-13. This document consolidates the recent LRS3 visual-phone,
family-probability, alignment, and phoneme-to-word experiments. It is an
experimental record, not a claim that the current system is production-ready.
Earlier GRID work and data preparation remain documented in [README.md](README.md).

## Decision and scope

The [actual visual-error decoder continuation](../CompetitiveWordDecoder/ACTUAL_VISUAL_ERRORS.md)
completed one bounded epoch and both evaluations. It improves reference-phone
WER to9.31% but worsens actual visual top-ranked WER to97.22%; it is not promoted.

The [frozen word-decoder follow-up](CONTEXT_WORD_EVALUATION.md) now confirms a
downstream development improvement: top-ranked WER falls from 99.59% to 95.48%
with the selected context checkpoint. This remains poor absolute performance;
no deployment promotion or fresh held-out validation is claimed.

The subsequent [probabilistic family-loss sweep](FAMILY_OBJECTIVES.md) compares
17-, 31-, and 10-family auxiliary objectives without changing the 39-phone
scoring inventory. The coarse variant narrowly wins that sweep, but none beats
the 82.67% development-selected context checkpoint. Existing baselines remain
preserved; no deployment promotion follows from this result.

**Follow-up:** the subsequently authorized
[multi-frame context-head experiment](MULTIFRAME_CONTEXT.md) improves exact
development PER to 84.63% at the fixed final step versus 85.78% for its matched
linear control. Its development-selected best checkpoint reaches 82.67%.
This is a new experimental candidate, not a deployment promotion; the remainder
of this record describes the preceding experiments and their decisions.

Retain the existing baselines. None of the recent family-text decoders or
individual-phone-head adaptations earns promotion. The current priority is to
understand label timing and representation quality before scaling training.
More hours of the same training cannot be assumed to fix a mismatched target.

The intended inference interface is face video to phoneme probabilities to a
sentence. Audio and transcripts supply training supervision only. Experiments
that feed reference phones into a word decoder measure that decoder's ceiling
under cleaner input; they are not visual speech-recognition results.

The immediate approved follow-up is a timing audit and a small, training-only
overfit diagnostic. It does not authorize another large training run. Existing
checkpoints are preserved, so rollback means selecting a previous artifact,
not attempting to undo optimizer updates.

## Data, labels, and separation

LRS3 manifests live in `datasets/lrs3/manifests`. The preparation workflow uses
the published AV-HuBERT validation-ID list, with 1,140 eligible clips remaining
in the shared evaluation here. The scored references contain 44,586 phones and
12,675 normalized words. These are development-validation results: this split
has been inspected repeatedly and is no longer an untouched final test.

Transcript-derived references use CMUdict pronunciation conversion with stress
removed. A dictionary pronunciation is not a manually verified recording-level
phone annotation: pronunciation variants, reductions, and dictionary coverage
can affect measured PER. Word normalization uses alphabetic words and internal
apostrophes. Scores must retain their exact reference and normalization policy.

The local pretraining corpus combines the prepared approximately 403-hour
pretrain portion and 30-hour trainval portion. Availability of these sources
does not mean every experiment below trains on all of them. The initial head
pilot samples 4,000 training clips; the paired alignment experiment samples
4,000 accepted training chunks. Those are different training units.

Audio-teacher labels are under `datasets/lrs3/audio-teacher-full-forced`.
Inspected provenance records identify faster-whisper/large-v3-turbo for ASR and
CTC Viterbi alignment using `facebook/wav2vec2-lv-60-espeak-cv-ft`. These labels
are machine-generated, not ground truth. Transcript agreement must be at least
0.8 for the aligned-chunk loader. Agreement filters obvious disagreement but
does not certify phone boundaries or audiovisual synchronization.

For current aligned training, words are packed to a nominal maximum of 40 phones
per chunk, with 40 ms context padding on each side. Start and end frames use
floor and ceiling respectively. A frame is labelled by the interval containing
its center, `(frame_index + 0.5) / fps`. Phones belonging to neighbouring words
are excluded from the padded margins. Missing frame labels are `-100` and ignored
by frame cross-entropy; they are not automatically silence or CTC blank.

Training source clip IDs are checked against validation IDs. Manifests retain
selected IDs and hashes for paired comparisons. The older 5,000-clip noisy
training export comes from training data seen by its visual model; it is not
out-of-fold supervision. Future realistic-noise training should use out-of-fold
or otherwise independent visual predictions. Never train from evaluation
transcripts or use reference-aware oracle selection at inference.

## Visual models and target inventories

The older compact hybrid has 157,849 parameters and combines normalized
landmark motion with image features. The larger image model has approximately
9.26 million parameters: grayscale frame CNN features feed a 384-wide temporal
network with ten dilated blocks. Landmarks locate the crop; they do not give a
direct measurement of a hidden tongue. Neither architecture observes internal
articulation that the camera cannot see.

The retained pretrained family source is
`checkpoints/lrs3-pretrain-433h-large-image-gpu-20260923-v2/best.pt`.
It predicts 17 visual families plus CTC blank. Families group visually similar
phones; reducing the inventory makes the task easier but loses distinctions.
REST is a separate experimental target derived from gaps. CTC blank means no
output token at a timestep and must not be interpreted as physical silence.

CTC marginalizes frame-to-token alignments. Greedy decoding collapses consecutive
repeated predictions before removing blank. The phone-head experiments upsample
logits by a factor of two before CTC. This creates additional alignment steps,
not additional observed video frames.

The new individual-phone classifier has 39 phones plus blank. Each phone starts
with its family's weight vector and the family bias minus log family size.
Thus member probabilities initially sum to the original family probability.
Within-family probabilities begin uniform; initialization alone does not learn
which member was spoken. The family head remains available as an auxiliary loss.
Implementation and detailed results: [INDIVIDUAL_PHONE_HEAD.md](INDIVIDUAL_PHONE_HEAD.md).

## Metric contracts

| Metric | Meaning and limitation |
| --- | --- |
| Exact PER | Phone substitutions + deletions + insertions, divided by reference phones. |
| Group PER | The same edit calculation after mapping phones to families; adjacent mapped tokens are not merged. It is not exact-phone accuracy. |
| WER | Word substitutions + deletions + insertions, divided by reference words; insertions can make it exceed 100%. |
| Oracle PER@5 / WER@5 | Reference-aware minimum error among five candidate sequences. An upper bound on selection quality, not deployable top-1 performance or token top-5 accuracy. |
| Frame accuracy | Accuracy only on labelled frames; sparse labels can make it unrepresentative of the whole clip. |

Always distinguish sequence PER from frame accuracy, family PER from exact PER,
and actual selected output from an oracle. A family result is not “fake” if
labelled accurately, but using it as exact PER would be misleading. These runs
do not include uncertainty intervals or establish statistical significance.

## Word-decoder supervision

The local clean decoder follows a VALLR-style Llama-3.2-3B-Instruct recipe with
WikiText-2 text converted to phones, using task LoRA (rank 16, alpha 32, dropout
0.05 on attention and MLP projections). Input is a tagged phone prompt; only
target text and its end-of-answer token contribute to the loss.

The initial recipe omitted explicit target EOS supervision and could truncate
answers. The repaired continuation keeps only complete prompt/answer/EOS pairs
within 512 tokens, masks prompt loss, and supervises EOS even when padding uses
the same token ID. It continues from the saved adapter for one epoch at learning
rate 2e-5, effective batch four. See
[VALLR_CLEAN.md](../CompetitiveWordDecoder/VALLR_CLEAN.md) for pinned revisions,
eligibility counts, artifacts, and reproduction details.

The resulting clean WikiText evaluation is 7.32% WER on 1,489 eligible examples.
The dictionary-only stratum is 9.42%; spelling embedded in `UNK(word)` is a
leakage caveat for the broader stratum. Neither number is LRS3 lipreading WER.
On the shared LRS3 reference-phone input, the clean EOS decoder reaches 13.40%
WER. That is a useful decoder diagnostic, not an attainable visual result with
the current phone recognizer.

Synthetic-noise continuation uses deterministic substitutions, deletions, and
insertions on known phones, with 5/10/20% corruption rates sampled 50/40/10.
Edit types are sampled 60/25/15. Clean/noisy mixtures of 50/50 and 75/25 are
compared on the same eligible pool. Synthetic validation improves under noise,
but real visual-phone performance does not improve enough to promote a model.
See [SYNTHETIC_NOISE.md](../CompetitiveWordDecoder/SYNTHETIC_NOISE.md).

## Sentence context, families, and probabilities

The clean decoder already consumes multiword phone sequences. Sentence context
does not require treating words independently. Two further experiments encode
family sequences as fixed A–Q symbols or explicit alternatives such as
`[B/M/P]`. Neither includes word boundaries or a marker revealing the actual
phone. Repeats are retained. Dictionary-covered multiword training text is
filtered against exact normalized evaluation-transcript duplicates.

The explicit-alternative representation needs more context space: its run uses
768 tokens instead of 512 and therefore has a different eligible training pool.
This is not a pure encoding-only ablation. Both representations fail even on
reference-derived families, so sending noisier visual families through them is
not a justified next step. See
[FAMILY_SENTENCES.md](../CompetitiveWordDecoder/FAMILY_SENTENCES.md).

Probability ranking keeps five concrete phone paths and their cached decoded
sentences. The fixed score is phone-path beam score plus summed conditional
decoder token log probability, with EOS scored only for terminated outputs.
Normalized duplicate sentences are aggregated with log-sum-exp. The phone beam
already includes a phone-language-model weight of 0.5; the resulting candidate
mass is relative and uncalibrated, not a pure visual posterior. A separate
mean-token-score heuristic is evaluated without tuning on these results.
This reorders the existing pool; it cannot repair missing candidates or improve
its oracle. See [PROBABILITY_RANKING.md](../CompetitiveWordDecoder/PROBABILITY_RANKING.md).

## Matched development results

The following LRS3 results use the shared 1,140 full clips unless stated otherwise.
Do not compare columns as if their target inventories were equivalent.

| Condition | Exact PER | Family PER | Word WER / oracle WER@5 |
| --- | ---: | ---: | --- |
| Older hybrid visual model | 84.67% | 78.11% | — |
| Pretrained larger image family model | unavailable | 71.98% | — |
| REST image model, REST removed for scoring | unavailable | 74.86% | — |
| Old visual paths + clean EOS decoder | — | — | 99.59% / 96.44% |
| Old visual paths + 50/50 synthetic-noise decoder | — | — | 103.76% / 95.87% |
| Fixed probability ranking of clean candidates | — | — | 98.45% / 96.44% |
| Reference families, A–Q decoder | — | — | 108.05% / not applicable |
| Reference families, explicit alternatives | — | — | 115.07% / not applicable |

The old visual phone beam uses width 64 and token top-k 12; its oracle exact
PER@5 is 78.14%. Probability ranking removes 145 word edits relative to baseline
selection, but the very poor oracle shows that selection is not the main bottleneck.

The new phone-head pilot samples 4,000 training clips, batch eight, for 400
updates. The first 100 train only the phone head with the frozen backbone in
evaluation mode. The following 300 update the backbone as well. AdamW uses
phone learning rate 3e-4, encoder 1e-5, weight decay 0.01, gradient clipping 5,
and phone CTC + 0.3 family CTC. Initialization gives 92.05% exact PER; step 100
improves to 87.75%; step 400 gives 88.88% and worsens family PER to 74.49%.
This pilot has no no-auxiliary-loss control and does not beat the older hybrid.

The next paired experiment starts both arms from step 100, freezes all encoder
and family-head parameters and buffers, and uses the same 4,000 aligned chunks.
Both run 1,000 updates, batch eight, head learning rate 3e-4, with evaluation
and checkpoints every 250 updates. Control uses phone CTC; treatment adds
0.1 times masked frame cross-entropy. Family loss is constant when its complete
path is frozen and cannot train the phone head.

Control finishes at 88.51% exact PER; aligned treatment at 89.69%. Treatment
adds 526 errors, or 1.18 percentage points, relative to control. Both are worse
than their 87.75% start. Tensor equality verifies the frozen model components
remain unchanged, with family PER still 71.98%. Artifacts are
`checkpoints/frozen-phone-{control,aligned}-20261013`. These results do not isolate
timing quality, frozen-feature limitations, chunk/full-clip mismatch, or loss weight.

## Timing audit and tiny-sample diagnostic

`diagnose_phone_alignment.py` audits label coverage over the same 4,000 chunks,
independently remaps timestamps for 128 chunks, and compares direct chunk decoding
against full-video decoding followed by slicing on 16 chunks. Alignment-file
hashes are retained. The audit distinguishes known-label disagreements from
unlabelled padded-margin frames. This is an implementation consistency check,
not human validation of articulation timing.

Initial measurements show only 27.4% of frames labelled. Of 3,210 inspected
phone intervals overlapping chunks, 1,425 (44.4%) contain no video-frame center.
Short CTC intervals are therefore an important limitation of this frame loss.
These intervals may represent narrow acoustic evidence rather than the full
duration of visible articulation; that interpretation remains a hypothesis.
All 24 discrepancies against unrestricted source intervals are missing labels
within the first or last three chunk frames, consistent with the margin mask;
this positional check does not independently reconstruct word membership.
There are zero interior or known-label disagreements. All 16 direct/full-video
crop comparisons match exactly. Stream start offsets have not been verified because ffprobe
is unavailable. Neither a correct frame index nor matching crop tensors proves
that sound and visible movement are synchronized.

After audit checks pass, the diagnostic caches frozen visual features for 16
training chunks. Two independent linear heads start from the same checkpoint:
CTC + 0.1 frame CE, and frame CE alone. Each gets up to 1,000 AdamW updates at
learning rate 0.003, no weight decay, with measurements every 100 updates. The
combined optimization budget is capped at 15 minutes, excluding preparation.
It records in-sample frame accuracy, CE, and greedy PER and verifies the source
model is unchanged. Results belong to the diagnostic artifact, not the table
of held-out accuracy above. Fitting these examples would show optimization
capacity on this sample, not generalization. Failure would not alone prove
incorrect labels: a frozen linear representation may be insufficient.

Both arms completed 1,000 updates in
`checkpoints/phone-alignment-diagnostic-v3-20261013`; the source model remained
tensor-for-tensor unchanged. On the 16 training chunks (380 reference phones,
215 labelled frames), CTC + alignment finishes at 16 phone errors / 380 =
**4.21% in-sample PER**, with 11.63% labelled-frame accuracy. Frame-only training
reaches **100% labelled-frame accuracy**, CE 0.00478, but 351 / 380 = **92.37%
sequence PER**. The head can optimize these fixed features; sparse frame-label
memorization is not sufficient to learn a usable sequence output. These are not
new validation results, and neither diagnostic head is promoted. Narrow timing
and the mismatch between CTC emission positions and audio frame labels deserve
investigation, but their causal contribution to held-out failure is unproven.

## Reproducibility and safe operation

Use the repository GPU launcher with host GPU access; it supplies the required
CUDA library paths. Do not install replacement CUDA packages merely because a
restricted shell cannot see the GPU. Example diagnostic invocation:

```bash
tools/with-vpa-gpu .venv-vpa-gpu/bin/python -m VisualPhoneme.diagnose_phone_alignment \
  --output checkpoints/phone-alignment-diagnostic-v3-20261013 \
  --steps 1000 --max-minutes 15
```

Use a fresh output directory for a new run. Resume only with `--resume` and the
matching saved manifest. The diagnostic checks repository/run `PAUSE` files,
handles termination requests, saves head/optimizer/history state, and uses the
thermal guard. Pause is cooperative, not an instantaneous GPU kill; preparation
and a current operation may finish first. Preserve logs, failed attempts,
checkpoints, and final metrics. A stopped process is not evidence of completed
training or successful evaluation. Never silently promote the last checkpoint.

Phone exports retain an inherited source `epoch` field; use `phone_head_step`
and the run manifest to identify new training progress. Keep validation IDs,
reference hashes, crop policy, decoder normalization, beam settings, and target
inventory alongside reported metrics. Raw dataset media and local checkpoints
should remain outside tracked source artifacts. Do not upload raw recordings
or credentials as part of review or debugging.

## Options and acceptance gates

1. **Retain the older phone baseline.** Lowest cost and immediately reversible;
   its exact PER is better than the current new head, although still poor.
2. **Repair/validate timing supervision, then repeat a controlled adaptation.**
   Recommended next direction if diagnostics support it. Requires confidence
   handling and a defensible interval policy, rather than stretching all labels
   blindly. Compare CTC-only and added supervision on identical training data.
3. **Unfreeze more visual layers or add a larger teacher.** More compute and
   capacity, but greater overfitting/catastrophic-forgetting risk. Defer until
   the simpler diagnostic identifies a plausible bottleneck.

Before larger training: resolve interior label mismatches, inspect a manually
checked audiovisual sample, quantify label coverage, verify finite losses and
resume behavior, and demonstrate meaningful tiny-sample optimization. Before
promotion: beat the exact-phone baseline on matched evaluation, preserve the
family metric where relevant, run the unchanged word decoder to measure actual
WER impact, and confirm on a fresh held-out split. Choose thresholds before
looking at that final split. More data is useful only when its supervision and
distribution are suitable; it is not the current default remedy.

## Comparison boundaries and external review

The local VALLR-style word training is a recipe adaptation, not a reproduction
of a released complete sentence system. The isolated-word diagnostic of the
original visual checkpoint uses supplied word boundaries and a different sample;
it must not be compared numerically with these full-clip results. No conclusion
here establishes superiority over the published original VALLR system.

Independent design review was attempted on 2026-10-13 with:

```bash
.venv-vpa-gpu/bin/python /home/elinzi/.codex/skills/write-design-doc/scripts/review_design.py \
  --design VisualPhoneme/METHODOLOGY.md \
  --output VisualPhoneme/methodology.feedback.md \
  --launcher /Users/haha/github/ai-skills/bin/claude_deepseek.py
```

It failed because the required external launcher is not installed. See the
[review attempt log](methodology.feedback.md.log). No critique was produced.

| Review item | Disposition | Reason |
| --- | --- | --- |
| Independent technical critique | Deferred | Required reviewer launcher unavailable; no feedback to incorporate. |

The recommendation above is based on repository measurements, not an independent
model's endorsement. Code compilation and whitespace checks pass; the targeted
phone-head tests are run with the standard-library unittest runner.
