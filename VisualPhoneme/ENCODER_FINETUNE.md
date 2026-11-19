# Fine-tuning the Auto-AVSR encoder for phones, 2026-11-19

Why: frozen-feature phone heads stop at ~26.9% greedy PER on test-select
(head C), while Auto-AVSR's own words converted to phones reach 18.8% PER, so
the encoder holds phonetic information the heads do not extract. Every
decoder-side change since has tied at ~36.5% WER (see
[SEAMLESS_DATA.md](SEAMLESS_DATA.md)).

## Trainer (`finetune_autoavsr_encoder.py`)

- Starts from `LRS3_V_WER19.1` and head C; trains from mouth crops, not cached
  features.
- Frontend, embedding and the lower `12 - N` conformer blocks frozen (eval
  mode); the top `--trainable-blocks N` blocks and the head train. BatchNorm
  stays in eval mode everywhere (padded batches).
- Loss: phone CTC (head C's feature time masks) + `--distill-weight` × KL from
  a frozen copy of the original top blocks through the published word CTC
  head. The checkpoint has no sentencepiece model, so transcripts cannot be
  re-tokenized for a true word CTC loss; the KL keeps word knowledge instead.
- Augmentation: random 88×88 crop from 96×96 and horizontal flip (Auto-AVSR's
  training transforms); evaluation uses the centre crop.
- Sources `MANIFEST@ROOT`, one per batch, sampled by `--train-weights`.
- Selection: test-select greedy PER every `--eval-every` steps; test-select
  greedy word-CTC WER is logged as a forgetting monitor. Test-report is not
  evaluated by the trainer.
- Checkpoints store only the trained top blocks and head;
  `cache_autoavsr_features.py --finetuned CKPT` re-caches features so the
  existing export and decoding tools apply unchanged.

## Smoke run

4 trainable blocks (69.9M parameters), batch 16, LRS3 trainval + Seamless
batch 0000 at 1:1, 300 steps:

- Step 0 reproduces the frozen system: 26.94% test-select greedy PER (head C:
  26.90%) and 31.63% word-CTC WER (published-encoder greedy CTC: 31.7%).
- Speed: ~13 steps/s (a full LRS3 trainval pass ≈ 2.5 min); test-select
  evaluation 1.4 s. No thermal pauses.
- 300 steps (still in warm-up) move PER to 27.06%: a plumbing check, not a
  result.

Runs R1 (LRS3 only) and R2 (LRS3 + Seamless) have not been started.
