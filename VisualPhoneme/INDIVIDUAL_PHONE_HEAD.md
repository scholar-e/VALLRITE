# Individual-phone head with auxiliary family supervision

The 2026-10-13 pilot starts from
`checkpoints/lrs3-pretrain-433h-large-image-gpu-20260923-v2/best.pt`, the stronger
full-clip family model, not the later REST continuation. A 40-class head predicts
39 ARPAbet phones plus CTC blank from the shared 384-dimensional temporal features.
The existing 18-class family/blank head is retained. Each phone initially copies
its family weight and bias, subtracting log(family size) from the bias. This
preserves summed family probability mass under a uniform within-family prior;
it does not invent observed within-family confidence.

Loss: phone CTC + 0.3 family CTC. Twofold CTC upsampling accommodates repeated
family targets. First 100 steps train only the phone head with the shared model
in evaluation mode. Remaining 300 steps fine-tune jointly. AdamW uses lr 3e-4
for the new head and 1e-5 for shared layers; batch 8, seed 42, weight decay .01,
gradient clipping 5. Nonfinite loss/gradients raise errors. The pilot samples
4,000 training clips deterministically, without augmentation, and performs 400
steps (3,200 clip presentations), capped at 30 minutes with thermal protection.
Targets are transcript/CMUdict phones, not new forced-audio labels.

Evaluation uses the same 1,140 full development-validation clips and 44,586
reference phones as earlier visual tests. IDs and phone labels are checked;
training/validation clip overlap is rejected. These are greedy sequence errors,
not frame accuracy, oracle PER, or word error. The same development split has
been inspected repeatedly, so it is not an untouched final test.

| Checkpoint | Exact-phone PER | Auxiliary family-head error |
| --- | ---: | ---: |
| Uniform-within-family initialization | 92.05% | 71.98% |
| Step 100: phone-head warm-up | 87.75% | 71.98% |
| Step 400: joint fine-tuning | 88.88% | 74.49% |
| Older 158k individual-phone model (prior matched evaluation) | 84.67% | Not applicable |

The warm-up learns something, but joint adaptation regresses from it and damages
family accuracy. Neither checkpoint beats the older model's exact-phone PER.
Do not promote this pilot or claim a WER improvement. No matched no-auxiliary
control was run, so the effect of auxiliary loss itself is not isolated.
The next reasonable diagnostic is longer frozen-encoder head training with
aligned phone supervision, before further joint fine-tuning; not launched here.

Artifacts: `checkpoints/individual-phone-head-20261013/metrics.json`, manifest,
predictions, logs, checkpoints every 100 steps, and resumable optimizer/RNG state.
The exported `phones-step-*.pt` files are compatible with the existing
`large-image` phone evaluator. Their `phone_head_step` identifies adaptation
progress; the inherited `epoch` field refers to the source model, not this run.
The original checkpoint is unchanged. The complete pilot including evaluations
finished in under a minute on the RTX 5090; 30 minutes was a cap, not a target.

```bash
tools/with-vpa-gpu .venv-vpa-gpu/bin/python -m VisualPhoneme.train_phone_head \
  --checkpoint checkpoints/lrs3-pretrain-433h-large-image-gpu-20260923-v2/best.pt \
  --source datasets/decoder-training/lrs3-full-forced-direct-validation/phone-hypotheses.jsonl \
  --output checkpoints/NEW-PHONE-HEAD --steps 400 --warmup-steps 100 \
  --max-minutes 30 --batch-size 8
```

Root or run-local `PAUSE` and SIGINT/SIGTERM stop at a training boundary and
save state. Remove the marker and repeat with `--resume` to continue training.
The time cap applies per invocation. Checkpoint exports and initial probability
mass preservation have dedicated unit tests.

## Frozen encoder with aligned supervision: paired follow-up

Completed 2026-10-13. Both arms start from `phones-step-0100.pt` above and keep
all encoder and family-head parameters and buffers frozen, with dropout and
batch normalization in evaluation mode. Training samples the same 4,000 accepted
forced-aligned chunks from `datasets/lrs3/audio-teacher-full-forced`, with maximum
40 phones per chunk and transcript agreement threshold 0.8. Evaluation remains
the same 1,140 full clips, not the aligned chunks. Source clip IDs are disjoint.
The two manifests' training-row hashes, IDs, checkpoint hashes, and settings
were checked to match except for frame loss weight.

Each arm runs 1,000 steps, batch 8, phone-head lr 3e-4, with evaluation and saved
checkpoints every 250 steps. The control uses phone CTC alone for the trainable
head. The aligned arm adds 0.1 times frame cross-entropy for known phone labels;
unlabelled frames are ignored, not labelled as blank or REST. Family CTC remains
in the logged objective but has no training effect when its entire path is frozen.
Each arm has a 15-minute cap and thermal/pause protection; both completed.

| Checkpoint / condition | Exact-phone PER |
| --- | ---: |
| Shared starting phone head | 87.75% |
| Frozen CTC control, step 1,000 | 88.51% |
| Frozen CTC + aligned frames, step 1,000 | 89.69% |
| Older individual-phone model | 84.67% |

Alignment worsened the final result by 526 edits /44,586 phones, or 1.18
percentage points, relative to the matched control. Neither improves the
starting head. The frozen encoder and family head are verified tensor-for-tensor
unchanged in both arms; family PER remains 71.98%. Three phone-head tests pass,
including masked frame loss and compatible export.

Artifacts: `checkpoints/frozen-phone-control-20261013` and
`checkpoints/frozen-phone-aligned-20261013`. The follow-up does not establish
whether the problem is alignment timing/quality, chunk-to-full-clip mismatch,
the frozen representation, or the chosen loss weight. Do not promote either
checkpoint. Before another training run, inspect those issues and verify the
head can overfit a small aligned sample. That follow-up was subsequently approved
and completed; see the timing-audit and tiny-sample diagnostic section of
[METHODOLOGY.md](METHODOLOGY.md). Its 4.21% in-sample PER is an optimization check,
not an improvement in held-out accuracy.
