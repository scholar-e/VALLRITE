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

## Payoff check (test-select, 660 clips)

One phone string per clip through the matched-noise decoder
(`decoder-matched-noise-20261102`, `evaluate_llama_visual --visual-top-one`).
Word-derived phones are Auto-AVSR's joint-decoded words (no LM) mapped through
CMUdict. Artifacts: `checkpoints/payoff-word-phones-20261119/`.

| Phone input | PER | Decoder WER |
| --- | ---: | ---: |
| Reference phones | 0% | 2.64% |
| Auto-AVSR words → CMUdict phones | 20.42% | 28.62% |
| Head C best path | 26.72% | 40.47% |
| (Auto-AVSR words directly, same clips) | — | 27.14% |

About 1.9 WER points per PER point between the two visual inputs. At ~20% PER
a single phone string comes within 1.5 points of Auto-AVSR's words, and
20-path rescoring has added ~3.7 points over top-1 for head C. So parity
needs roughly 20% PER, and beating the words needs less. Optimistic: the
word-derived errors form real words, which is easier for the decoder than
head errors.

## R1: LRS3 trainval only

Defaults above (4 blocks, encoder lr 3e-5, head lr 1e-4, distill 0.3,
batch 16), 8,000 steps (~4 passes), 885 s with frequent GPU thermal pauses.
`checkpoints/encoder-finetune-R1-lrs3-20261119/`.

| Step | Test-select greedy PER | Word-CTC WER |
| ---: | ---: | ---: |
| 0 | 26.94% | 31.63% |
| 3,000 (best) | 26.69% | 32.21% |
| 8,000 | 26.87% | 32.49% |

No gain: the best value is −0.25 points, selected on the same clips, and the
curve is flat within ±0.15 after step 500. Training phone CTC started at 0.60
and ended at 0.57 (with Seamless mixed in, the smoke run started at 1.3): the
encoder already fits LRS3 trainval, so it gives almost no learning signal.
Word-CTC WER drifted up 0.6–0.9 points despite the distillation term. This
supports training on encoder-unseen clips (R2) rather than LRS3.

## R2: LRS3 + Seamless (batches 0000 and 0001), 1:1

Same settings as R1; sources weighted 2:1:1 (LRS3 : Seamless 0000 : 0001, so
half the batches are encoder-unseen clips). 8,000 steps, 810 s.
`checkpoints/encoder-finetune-R2-mixed-20261119/`.

| Step | Test-select greedy PER | Word-CTC WER |
| ---: | ---: | ---: |
| 0 | 26.94% | 31.63% |
| 4,000 (best) | 26.58% | 32.29% |
| 8,000 | 26.72% | 32.17% |

Training phone CTC fell from 1.24 to 1.02 (mixed batches), so the Seamless
clips did give a learning signal, but it did not transfer to LRS3 test:
−0.36 points at the selected step, −0.22 at the end, within the ±0.2
step-to-step noise. Word-CTC WER again rose ~0.6 points.

## Reading

Neither run moved phone error by more than ~0.3 points; parity with Auto-AVSR's
words needs ~7. The 18.8%/20.4% word-derived PER is not a CTC number:
Auto-AVSR's own greedy CTC words give 24.66% PER on test-report, close to head
C (26.43% beam). The gap comes from the autoregressive attention decoder
(context over the whole sentence), not from information a CTC head fails to
read from the encoder. R3 (more blocks) was conditional on a clear gain and is
not run.
