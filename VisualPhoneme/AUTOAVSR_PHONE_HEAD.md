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

## Larger and masked heads, 2026-11-06

The phone interface was kept. Same cached frozen features, selection on
test-select greedy PER, test-report evaluated once per run. New trainer options:
`--hidden`, `--input-dropout`, and `--time-masks` / `--time-mask-width`, which
zero random spans of encoder frames during training (each span at most a fifth
of the clip), aimed at finding 4 (training features are cleaner than test ones).

| Head | Steps (selected) | test-select greedy | test-report greedy | test-report beam | Oracle@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous: 2 layers × 256 | 10,000 | 29.02% | 28.48% | 27.99% | 25.49% |
| A: same, longer | 18,000 / 20,000 | 28.74% | 28.41% | 28.14% | 25.65% |
| B: 4 layers × 512 | 16,000 / 20,000 | 27.80% | 27.38% | 27.38% | 24.74% |
| **C: B + 2 masks ≤ 10 frames, input dropout 0.3** | 18,000 / 20,000 | **26.90%** | **26.58%** | **26.43%** | **24.03%** |
| D: B + 4 masks ≤ 15 frames, input dropout 0.4, 30k planned | 8,000 (stopped at 9,437) | 29.88% | — | — | — |

Longer training alone does not help; width helps a little and masking most
(−1.56 beam PER points over the previous head). D was behind C's curve and was
stopped early to stay within the session budget; it is incomplete, not a
negative result. C is the selected head
(`checkpoints/autoavsr-phone-C-masked-20261106/best.pt`). Single seed.

End-to-end, C's phones did not lower held-out WER; see "Head C through the
posterior pipeline" in the [decoding record](../CompetitiveWordDecoder/AUTOAVSR_PHONE_DECODING.md).

## Stress-marked vowels, 2026-11-26

`train_autoavsr_phone_head.py --stress` splits each vowel by CMUdict lexical
stress (0/1/2): 24 consonants + 15 × 3 vowels = 69 phones. Collapsed, the
stressed labels equal the cached phone labels on every clip. Head C's exact
recipe (4 × 512, masks, input dropout 0.3, 20k steps, seed 20261102);
`checkpoints/autoavsr-phone-C-stress-20261126/`. (English has no lexical
tone, and pitch is not visible, so only stress was tested.)

| Head | test-select greedy PER, stress ignored | test-report greedy, stress ignored | test-report beam (own classes) |
| --- | ---: | ---: | ---: |
| C (39 phones) | **26.90%** | **26.58%** | **26.43%** |
| C + stress (69 phones) | 27.01% | 26.74% | 27.11% (with stress) |

On test-select vowels whose identity was recognised (5,139), stress is right
98.1% of the time, against 79.5% for each vowel's most common stress. But a
text-only lookup on ±2 neighbouring phones (no video) already gets 93.9%:
stress is mostly implied by the phone sequence (the word), not seen on the
lips. Extra classes cost ~0.1-0.2 PER points and give the decoder little it
cannot infer from the phones themselves, so the decoder was not retrained.

## Phone pieces (word-piece method on phones), 2026-11-26

`--phone-pieces N` learns N byte-pair merges over each word's phones (pieces
never cross words) and trains CTC on the resulting units; outputs expand back
to phones for scoring. 500 merges give 539 units, about 2.2 phones per unit
(`AH_N`, `DH_AH`, `IH_NG`, ... `B_EH_T_ER`); expanded targets equal the cached
phones on every clip. Head C's recipe, 20k steps, 17 min.
`checkpoints/autoavsr-phone-C-pieces500-20261126/` (hypotheses are in piece
units).

| Head | test-select greedy phone PER | test-report greedy phone PER |
| --- | ---: | ---: |
| C (39 phones) | **26.90%** | **26.58%** |
| C, 500 phone pieces | 28.52% | 27.50% |

Pieces are 1-1.6 points worse at every evaluation (step 12k: 29.56% vs
27.47%). Each piece has far fewer training examples than a phone, and the
blank rate rises to 90% (fewer, longer units per clip). Word pieces help
Auto-AVSR through its autoregressive decoder and lexicon, not through a
bigger CTC inventory; a CTC head over pieces does not get that benefit.

## Blank penalty for head C, 2026-11-26

Head C deletes ~3.7 phones per insertion, so `blank_penalty_sweep.py`
subtracts `b` from the blank log probability at decode time (no training).
`checkpoints/blank-penalty-headC-20261126/`.

| b (test-select) | Greedy PER | Substitutions | Deletions | Insertions |
| ---: | ---: | ---: | ---: | ---: |
| -0.50 | 26.96% | 2,468 | 1,868 | 302 |
| 0 | **26.90%** | 2,648 | 1,557 | 423 |
| +0.25 | 27.00% | 2,757 | 1,406 | 483 |
| +1.00 | 27.54% | 2,942 | 1,041 | 756 |
| +2.00 | 30.03% | 3,165 | 624 | 1,378 |

Beam (width 64) on test-select: 26.72% at b=0, 26.80% at 0.25, 26.96% at 0.5,
so b=0 is selected and test-report is unchanged (26.43% beam). PER is flat
within ±0.1 point for b in [-0.25, +0.25]: the head is already at its best
balance. Phones recovered from deletions arrive mostly as substitutions or
insertions, i.e. the skipped phones are ones the head cannot identify, not
ones it is suppressing.
