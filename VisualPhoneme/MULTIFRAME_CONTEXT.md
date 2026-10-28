# Multi-frame phone-context experiment

2026-10-13. This is a matched development experiment, not a deployment change.
See [METHODOLOGY.md](METHODOLOGY.md) for data lineage and metric definitions.

2026-10-28 clarification: "full training clips" below means complete prepared
dataset rows, many of which are already short pretraining segments. The selected
training median is 28 frames versus 73 for evaluation. See the
[system audit](SYSTEM_AUDIT.md) for crop, padding and blank-calibration findings.

## Question and predeclared comparison

Can a trainable nonlinear temporal phone head improve exact-phone PER over a
linear head on identical frozen image features? The backbone already contains
ten temporal blocks, each with two kernel-three convolutions, with dilations
`1,2,4,8,16,1,2,4,8,16`. Its theoretical receptive field is 249 input frames;
this experiment does not introduce temporal information into a frame-only model.

The treatment adds a residual correction to existing phone logits: LayerNorm
over 384 features, a 384-to-128 projection, GELU, a kernel-nine 128-channel temporal
convolution, GELU, and a 128-to-40 output projection. The correction output starts
at zero, preserving the initial model predictions exactly. It learns across
nine neighbouring contextual feature frames (approximately 0.36 seconds at
25 fps), increasing the theoretical receptive field to 257 frames. Symmetric
padding uses future frames: this is offline recognition, not causal streaming.

The added capacity and temporal mixing change together. A gain would not by
itself establish that temporal mixing, rather than nonlinear capacity, caused
the improvement; a parameter-matched framewise control would be a later ablation.

## Fixed protocol

- Initialize both arms from `individual-phone-head-20261013/phones-step-0100.pt`.
- Keep encoder and original family head frozen, including normalization buffers.
- Use the same seed-42 subset of 4,000 full LRS3 training clips and the same
  deterministic order, with no flips. Do not use forced-aligned chunks or frame CE.
- Optimize phone CTC with upsampling factor two. The logged auxiliary family CTC
  has weight 0.3 but is constant with respect to the trainable head.
- AdamW head learning rate 3e-4, weight decay 0.01, gradient norm cap five;
  batch eight; 1,000 updates; checkpoint/evaluation every 250 updates.
- Cap each arm at 15 minutes of the training loop. Preparation and initial/final
  evaluation are outside that cap. Preserve thermal guard and cooperative pause.
- Evaluate the same 1,140 development clips and 44,586 reference phones. Primary
  comparison is final exact PER at matched updates. Report the checkpoint curve
  without presenting development-selected best performance as untouched testing.

This full-clip protocol also differs from the preceding aligned-chunk study.
Only the two arms here isolate the context-head change. The 84.67% older hybrid
and 87.75% shared starting head remain reference baselines, not matched controls.

## Artifacts and compatibility

Control: `checkpoints/fullclip-phone-control-20261013`.
Treatment: `checkpoints/fullclip-phone-context-20261013`.
Each stores a manifest, resume checkpoint, step exports, logs, final metrics,
and final predictions. Resume with identical arguments plus `--resume`.

Context exports are explicitly tagged `phone-family-context` and contain both
heads. They must be loaded with `VisualPhoneme.train_phone_head.load_phone_export`;
they are not drop-in `large-image` state dictionaries. Standard control exports
remain compatible with `LargeVisualPhoneme`. Downstream decoder integration is
deferred until the exact-PER comparison justifies it.

Tests cover initial equivalence, gradient flow, temporal-neighbour sensitivity,
export/reload equivalence, the previous standard export, family probability
initialization, and masked frame loss. No additional dependencies are needed.

The local DeepSeek worker was asked for a read-only review (job
`4982b369-6150-4682-89ba-8fb666d984e4`). It exhausted its context repeating its
analysis and did not deliver a substantive final review. Its log's claims that
`frozen_state` is unused and export loading is non-strict are incorrect: final
validation checks the frozen state and the export loader uses strict loading.
Those claims are rejected on code inspection. The learning-rate documentation
suggestion is covered by the fixed protocol above. This is not independent
review approval; the supervising agent verified the implementation and tests.

## Results

Both arms completed all 1,000 updates. Manifest comparison finds exactly one
difference: `context_head`. Source checkpoint hashes, references, training rows,
IDs, batch size and optimization settings match. Both frozen-state checks pass;
family-head PER remains 71.98%. All six unit tests and whitespace checks pass.

| Updates | Linear control exact PER | Context-head exact PER |
| --- | ---: | ---: |
| 0 | 87.7540% | 87.7540% |
| 250 | 87.3705% | 86.1077% |
| 500 | 86.1952% | 84.8854% |
| 750 | 85.0805% | **82.6717%** |
| 1,000 | 85.7825% | **84.6297%** |

At the predeclared final step, context removes 514 edits (37,733 versus 38,247)
over 44,586 reference phones: a **1.153 percentage-point improvement** over the
matched control. At step 750 it removes 1,074 edits relative to the same-step
control, or 2.409 points. Its development-selected best checkpoint is
`fullclip-phone-context-20261013/phones-step-0750.pt` (36,860 errors).

The best checkpoint improves on the older hybrid's approximately 84.67% exact
PER, but is selected on repeatedly inspected development data. The final
84.63% score is only marginally below that older baseline. The regression from
750 to 1,000 steps warrants investigation; it does not alone establish overfitting.
No held-out test or WER evaluation was performed in this experiment. No deployed
model was replaced. The training-loop/final-evaluation timers were approximately
75.5 seconds for control and 73.1 seconds for context, well below either cap;
these are not controlled speed benchmarks.

Next recommended experiment, requiring a separate launch decision: compare a
parameter-matched framewise nonlinear head against this context head, with a
predeclared learning-rate schedule and stopping protocol. Confirm any selected
model on a fresh held-out set and through the unchanged word decoder. Do not
assume that a longer run or a wider context window must improve PER.
