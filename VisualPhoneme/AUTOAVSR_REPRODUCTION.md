# Auto-AVSR reproduction on the LRS3 test set, 2026-11-02

Phase 1 of the [training plan](../FUTURE_TRAINING_PLAN.md): load the complete
published Auto-AVSR visual model, not only its frontend, and check that it
reproduces its published accuracy on local data before training anything on it.

## What was built

- `autoavsr_encoder.py`: dependency-free port of the Conv3D/ResNet-18 frontend,
  12-layer conformer encoder and 5,049-way CTC head. Loads
  `checkpoints/pretrained/autoavsr/LRS3_V_WER19.1/model.pth` with strict key
  matching (186M parameters). Padded frames are masked, so batched and
  single-clip outputs agree (max difference 6e-4 in log probability).
- `autoavsr_decoding.py`: 6-layer attention decoder and joint CTC/attention
  beam search with CTC prefix scoring, matching ESPnet's decoding rule.
- `evaluate_autoavsr_ctc.py`: test-set evaluator (`--decoding greedy|joint`).
- Unit labels: `unigram5000_units.txt` from the authors' repositories
  (identical in both; sha256 `29001d96…a9122`), stored under
  `checkpoints/pretrained/autoavsr/tokens/`.

Preprocessing follows the authors' test transform: the stored 96×96 grayscale
mouth crops are centre-cropped to 88×88, scaled to [0, 1] and normalized with
mean 0.421, std 0.165. The local test crops (`datasets/lrs3/test-video/*.npy`,
from a third-party parquet) are already mouth-aligned in the authors' style.

One port bug was found and fixed: ESPnet's feed-forward blocks use ReLU even
when the model is configured for swish. With swish everywhere, the model
emitted blank on every frame.

## Results

All 1,321 official LRS3 test clips, 9,863 normalized words, fp32. LRS3 test is
not part of Auto-AVSR's training data, so these are fair held-out numbers.

| Decoding | WER | Sub | Del | Ins | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Greedy CTC head only | 31.72% | 2,089 | 832 | 208 | 6 s |
| Joint CTC/attention, beam 40, CTC weight 0.1, no LM | **26.35%** | 1,923 | 424 | 252 | 60 s |
| Published (adds subword RNN LM, weight 0.3) | 19.1% | — | — | — | — |

For context, the current VALLRITE phoneme pipeline is at 93.29% WER on the
development set.

Artifacts: `checkpoints/autoavsr-ctc-test-20261102/` and
`checkpoints/autoavsr-joint-test-20261102/` (config with hashes, per-clip
hypotheses, metrics, logs).

## Gate assessment

The port is functionally correct: outputs are fluent sentences whose errors are
visually confusable words ("many" → "my"), joint decoding beats CTC alone as
expected, and the CTC prefix scorer matches brute-force path enumeration in
tests. A 7.3-point gap to the published figure remains. Two explanations are
unresolved:

1. **Missing language model.** The published configuration adds a subword RNN
   LM at weight 0.3. This plausibly explains part of the gap, but probably not
   all of it.
2. **Test crop provenance.** The local crops come from a third-party parquet,
   not the authors' preprocessing; raw test videos are not available locally,
   so this cannot be checked directly.

Decoding parameters were not tuned on the test set. The gate is treated as
substantially met for Phase 2's purpose: the encoder's representation is intact
and far stronger than any visual model trained in this repository.

## Implications for Phase 2

- Training crops must match this preprocessing. Training rows currently use the
  repository's own crops (per-frame boxes, letterboxing, no normalization), so
  an Auto-AVSR-style crop pipeline is needed first. LRS3 trainval has supplied
  landmarks; pretrain does not and needs a detector.
- A useful crop check: Auto-AVSR's WER on trainval clips cropped by the new
  pipeline should be very low, since it trained on them. A high WER would
  indicate a crop mismatch.
- Auto-AVSR saw all of trainval and pretrain, so checkpoint selection and final
  evaluation must use disjoint halves of the official test set.
