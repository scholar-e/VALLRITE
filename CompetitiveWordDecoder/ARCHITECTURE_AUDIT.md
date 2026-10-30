# Architecture audit and sequential corrections, 2026-10-29

Scope: the current phone-family-context visual model, frozen mixed Llama adapter,
recent fixed-crop phone exports and source-disjoint decoder continuation. This is
an audit of local code and artifacts, not a comparison against an independently
reproduced original VALLR. Prior system findings remain in
[the visual audit](../VisualPhoneme/SYSTEM_AUDIT.md) and
[matched adaptation results](MATCHED_ADAPTATION.md).

## Actual computation path

1. A grayscale 96×96 mouth crop passes through four stride-two 2D convolution
   blocks (32, 64, 128, 192 channels), each with BatchNorm and SiLU. Spatial
   average pooling reduces each frame to 192 numbers; projection expands to 384.
2. Ten residual temporal blocks each contain two width-384 convolutions with
   kernel three and BatchNorm. Dilation is 1,2,4,8,16 repeated twice. The context
   phone head adds a nine-frame residual correction to a linear classifier.
   The full theoretical receptive field is 257 frames, symmetric/offline.
   The current model is image-only; it does not consume landmark coordinates as
   a parallel modality. Landmarks can influence preprocessing in other runs.
3. The model has 9,477,186 parameters, including 218,192 in the phone head. CTC
   emits blank plus 39 exact phones. Temporal logits repeat twice for alignment
   capacity; repetition supplies no additional visual evidence. Current fixed
   crop exports use batch eight, blank bias -1, no phone LM, beam 64/top-k12.
4. Five collapsed phone strings, rather than frame distributions, enter the
   word stage. Each string is independently decoded into one greedy sentence.
   Phone scores do not enter this language-model prompt; they enter later ranking.
5. The word model is Llama-3.2-3B-Instruct: 28 layers, hidden width 3,072,
   intermediate width 8,192, 24 query heads and eight KV heads. Rank-16 LoRA
   covers attention and feed-forward projections: 24,313,856 adapter elements.
   Saved adapters also include 788,029,440 embedding/output-weight elements due
   to vocabulary resizing; those saved elements are not all trainable LoRA
   parameters and may include tied storage represented twice. Do not call the
   entire deployment a 9.48M model.
6. Ranking adds the retained phone-path score to sentence token log probability,
   aggregates duplicate normalized sentences and selects the largest score.
   This is a restricted candidate score, not calibrated word confidence or full
   marginalization over all text/phone combinations.

Census evidence: `checkpoints/architecture-audit-20261029/architecture.json`.

## Findings, ordered corrections and limits

| Finding | Evidence and impact | Action/status |
| --- | --- | --- |
| Scoring loses the original token path | Evaluation discarded token IDs; ranker re-encoded decoded text, which is not an inverse mapping | Fixed first: preserve IDs through EOS and score those exact IDs; strict validation and explicit legacy mode |
| Continuation judged by loss/final step rather than generated WER | Prior mixed run reduced loss but raised ranked WER from 95.11% to 126.53% | Fixed selection stage: baseline wins validation over both low-rate checkpoints; regression rejected |
| Frozen family loss described as useful auxiliary training | Independent frozen branch has no gradient to phone-head weights | Removed detached term from optimization loss; retained explicit diagnostic; active branch behavior tested |
| Batch padding influences visual logits | Both temporal convolutions and normalization process padded time positions; prior single-clip audit changes PER | Unresolved architecture issue; keep batching fixed now; test masking with retraining separately |
| Early spatial pooling compresses fine mouth information | Each frame becomes 192 pooled features before temporal reasoning | Potential information bottleneck, not a demonstrated isolated cause; compare spatial-temporal frontends only after controls |
| CTC collapse discards timing and most uncertainty | Five strings omit blank durations, frame confidence, alignment and most alternatives | Unresolved interface limitation; future decoder should consume timing/confidence or a compact lattice, with training support |
| One sentence per phone path restricts candidate recovery | Oracle WER remains high and regressed in adaptation | Do not expect ranking alone to recover transcripts absent from the candidate pool |
| Teacher-forced CE does not directly penalize rollout repetition | Clean reconstruction improves while visual generation adds substitutions/insertions | Test smaller updates under generated-WER selection; retention regularization remains a separate future variable |
| Supervision and exposure limit conclusions | CMUdict first-pronunciation references, large PER, historical pretraining not fully audited | Report as pilot validation/evaluation, not independent end-to-end test or manually verified labels |
| CTC blank is not silence | Blank participates in alignment/collapse, not an acoustic rest annotation | Preserve distinction; do not reinterpret blank penalties as silence detection |

The padding issue is real but changing masks on existing weights is not a proven
accuracy fix. Temporal receptive field, crop policy, normalization and training
length distribution interact. Changing all of them at once would hide which
change helped. No visual weights were retrained in this phase. The loss cleanup
changes objective accounting under freezing but preserves the phone gradient;
it does not retroactively improve historical checkpoints.

## Correction 1: exact token-path ranking

`generated_tokens.py` preserves generated IDs, including a real terminating EOS,
strips only post-EOS batch padding, and scores capped prefixes without inventing
EOS. It validates ID ranges, termination and consistency with the saved text.
New reports support `--require-exact-tokens`; old reports explicitly identify
legacy retokenized scoring. `--retokenize-diagnostic` reproduces that method on
identical new candidates. Tests cover non-invertible tokenization, skipped
special tokens, capped prefixes, corrupt text and absent token IDs.

On the 100-clip validation split, 42/500 generated token sequences differ after
text round-trip; five chosen candidate indices change. Exact-token ranked WER
is 93.46% (1,257 errors/1,345 words), legacy 93.31% (1,255 errors). The same
sentences have identical oracle WER, 90.41%. This is a correctness repair,
not an accuracy gain or an explanation for the prior 30-point regression.
The ranking sum still only includes observed text/path pairings; complete
cross-path text scoring would be a separate approximation correction.

Artifacts: `baseline-validation`, `baseline-validation-exact`, and
`baseline-validation-legacy` under `checkpoints/architecture-audit-20261029`.

## Correction 2: validation selection and continued experiment

Predeclared before validation outcomes: rerun mixed adaptation from the same
starting adapter and same 750 training clips, changing only learning rate from
1e-5 to 2e-6. Keep one epoch/375 updates, batch four, same seed, same clean/noisy
mixture, LoRA settings, 15-minute cap, PAUSE/resume and thermal guard. Compare
checkpoints 250 and 375 with the unchanged baseline on all 100 validation clips,
using exact-token ranked WER. Ties favor baseline then the earlier checkpoint.
No evaluation-reference tuning, ranker-weight change, KL penalty or new model
architecture is bundled into this learning-rate comparison.

`select_validation_adapter.py` checks source hash against the explicitly supplied
validation file, complete record identity, word counts, model revision, scoring
mode and rule before choosing. The selector never promotes a checkpoint merely
because training completed. This script adds an explicit selection stage;
it is not yet an in-training early-stopping callback. If baseline wins, do not
consume more evaluation output for these rejected candidates. If a continuation
wins, freeze that choice first and compare it with baseline on the 100-clip pilot
evaluation split, acknowledging that this split has already been inspected.

Run: `checkpoints/decoder-matched-smalllr-20261029`. Evaluation orchestration:
`tools/evaluate-smalllr-decoder`. All 375 updates completed in 175.74 seconds;
optimizer checkpoints at 250 and 375 are saved. Final training loss is 2.2503,
validation teacher-forced loss 2.1139. All generated validation results use the
same 100 clips / 1,345 words and exact-token scoring.

| Candidate | Ranked validation WER | Word errors | Oracle WER@5 |
| --- | ---: | ---: | ---: |
| Unchanged baseline | **93.46%** | **1,257** | 90.41% |
| Low-rate step 250 | 113.09% | 1,521 | 110.63% |
| Low-rate step 375 | 113.46% | 1,526 | 111.08% |

The predeclared selector chooses baseline; its decision and report hashes are
in `checkpoints/architecture-audit-20261029/selection/selection.json`. Neither
continuation proceeds to another evaluation-set pass. The lower rate alone
does not resolve adaptation drift; no claim of an accuracy improvement is made.
This does not isolate a learning-rate benefit versus the previous high-rate run,
whose reported word scores used a different evaluation partition and legacy
token scoring. Compare only matched rows within this table.

A full chunked tensor comparison confirms that saved embedding and output
weights are bit-for-bit identical to the starting adapter. This excludes drift
in those tensors, not every possible serialization/numerical issue. Training
manifest comparison confirms identical data hashes, initialization, seed,
presentation count and optimizer settings except learning rate. Fifty-two
relevant tests pass, including exact-token, selection and family-gradient tests;
whitespace and shell syntax checks pass. No dependencies were added.

## Correction 3: detached family objective

`active_family_objective` adds the independent family-head loss only when that
loss has a gradient graph. With the encoder/family branch frozen it returns
the phone loss unchanged, and the trainer logs `family_active=False`. Gradient
tests show the frozen term does not affect phone updates and the active term
retains the original 0.3-weighted gradient. The separate probability-grouped
phone-head auxiliary objective remains differentiable and unchanged. No visual
retraining or PER gain is claimed for this accounting correction.

## Remaining sequence

Follow-up: [rolling posterior representation and smoothing pilots](../VisualPhoneme/ROLLING_PHONES.md)
now preserve per-frame distributions and timed greedy segments. Plain and
blank-preserving averages over three/five frames both worsen validation PER;
identity remains selected. The current word decoder still does not consume
those extra fields, so the information-loss issue is not resolved by metadata
alone. Do not equate implemented timing exports with a trained timing-aware model.

After the current checkpoint decision, keep changes isolated: first a retention
objective if adaptation still drifts, then a matched padding-mask visual training
ablation, then an information-preserving visual-to-word interface. These are
unresolved experiments, not promises that each will improve accuracy. Do not
increase capacity, train on evaluation examples, or silently replace the retained
pipeline to make this audit appear successful.
