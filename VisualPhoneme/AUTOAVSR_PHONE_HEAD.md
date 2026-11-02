# Phone heads on the frozen Auto-AVSR encoder, 2026-11-02

Phase 2 of the [training plan](../FUTURE_TRAINING_PLAN.md), following the
[Auto-AVSR reproduction](AUTOAVSR_REPRODUCTION.md). Approved budget: 2 GPU
hours; about 25 minutes used, including all evaluation.

## Data and crops

`autoavsr_crops.py` reproduces the authors' `video_process.py` crop (landmark
interpolation, 12-frame smoothing, similarity alignment to the 20-words mean
face on eight stable points, 96×96 mouth patch). The mean face is vendored at
`assets/autoavsr_20words_mean_face.npy` (Apache 2.0, mpc001/auto_avsr). The
repository's landmark loader reorders points, so the crop module reads the
original order directly.

- Trainval: 31,957 of 31,982 clips cropped in 87 s (25 have no detected
  landmarks in the supplied files). Manifests: `datasets/lrs3/manifests/autoavsr/`.
- Pretrain (975k chunks) has no supplied landmarks and is not cropped yet.
- Crop check: Auto-AVSR on the 1,200 validation clips (which it trained on)
  reaches 8.99% WER (joint) / 16.57% (greedy), as expected for correct crops.
- The geometry matches the reference face (eyes just above the crop, mouth
  45 px wide). The stored third-party **test** crops look ~15% more zoomed in
  and shifted down, so test inputs differ slightly from training inputs.

## Protocol

Encoder features are cached once (`cache_autoavsr_features.py`, float16,
2.6M training frames, 2 minutes). Heads are trained on 30,757 trainval clips
with CTC over 39 phones + blank, two learned emissions per frame.

Auto-AVSR trained on all of trainval, so the official test set is split in
manifest order: the first 660 clips select checkpoints, the last 661 report
once. Validation is the old 1,140-clip development subset (44,586 phones,
identical to earlier results) but the encoder has seen it, so it is optimistic.

## Results

| System | Test-report greedy PER | Test-report beam PER | Oracle PER@5 | Dev 1,140 beam PER* |
| --- | ---: | ---: | ---: | ---: |
| Previous best visual model (context head, no phone LM) | — | — | — | 70.82% |
| Linear head, lr 1e-3, step 10k | 44.73% | 41.56% | 38.50% | 35.55% |
| Linear head, lr 3e-3, step 9k | 43.46% | 40.34% | 37.45% | 33.95% |
| 2-layer transformer head, lr 1e-3, step 10k | 28.48% | **27.99%** | 25.49% | 17.31% |
| Auto-AVSR words (joint, no LM) → CMUdict phones | — | **18.80%** | — | — |
| Auto-AVSR words (greedy CTC) → CMUdict phones | — | 24.66% | — | — |

\* Seen by the encoder during its training; not a held-out number.
Beam: width 64, token top-k 12, no phone LM. Single seed per configuration.

Artifacts: `checkpoints/autoavsr-phone-head-lr{1e-3,3e-3}-20261102/`,
`checkpoints/autoavsr-phone-context2-20261102/` (config, curves, best/last
heads, N-best phone hypotheses in the word-decoder format).

## Findings

1. **The encoder is the fix the plan predicted.** Even a linear head cuts phone
   error roughly in half relative to every visual model trained in this
   repository.
2. **Context matters for phones.** Two transformer layers on frozen features
   reduce held-out beam PER from 40.3% to 28.0%, and the curve was still
   improving at 10k steps.
3. **Word-first decoding gives better phones than phone-first decoding.**
   Converting Auto-AVSR's own word output to phones gives 18.8% PER on the same
   clips, nine points better than the best phone head. The phoneme bottleneck
   that VALLRITE is built around currently costs accuracy rather than adding it.
4. **Heads trained on encoder-seen clips generalize worse.** The transformer
   head's dev/test gap (17.3% vs 28.0%) is far larger than the linear head's.
   Training features come from clips the encoder memorized, so they are cleaner
   than test features; the head learns to rely on that. Data the encoder never
   saw (for example cropped pretrain clips, or other corpora) would be better
   training material.

## Decision needed before more training

The remaining budget should go to whichever direction the project wants:

- **Keep the phone interface** (VPA thesis): train the context head longer and
  on encoder-unseen data, then rerun the word decoder on its N-best phones.
- **Use Auto-AVSR's word hypotheses directly**, and apply the Llama/Qwen work as
  an N-best reranker or language model over them. This starts at 26.35% WER.
- **Combine**: feed both word hypotheses and phone posteriors to the decoder.
