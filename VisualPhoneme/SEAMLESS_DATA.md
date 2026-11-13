# Encoder-unseen data: Seamless Interaction, 2026-11-08

The [posterior decoder](../CompetitiveWordDecoder/POSTERIOR_DECODER.md) was
limited by training inputs: every LRS3 clip outside test was seen by the
Auto-AVSR encoder. The checkpoint (`LRS3_V_WER19.1`) is the 3,448-hour model,
trained on LRW, LRS2, LRS3, VoxCeleb2 and AVSpeech, so none of those qualify.

## Candidates considered

| Corpus | Size | Transcripts | Access | Use |
| --- | --- | --- | --- | --- |
| Seamless Interaction (Meta) | 4,000+ h, 4,000+ people | automatic, word-timed with scores | Hugging Face, CC-BY-NC 4.0, ungated | chosen |
| Candor-LR (CANDOR corpus) | ~700 h video calls | word-aligned (commercial ASR) | licence from CANDOR authors | later, needs paperwork |
| TCD-TIMIT | ~8 h, 62 speakers | read TIMIT sentences, phone labels | research registration | small clean check |
| LRS-VoxMM | ~23 h (unconfirmed) | human | evaluation only | second test set |

## Format

Each `naturalistic/train/<batch>/<n>.tar` (~1 GB, 40 per batch) holds per
participant and interaction: a 1080x1920 30 fps H.264 face video, a WAV, a JSON
with WhisperX-style segments (words with start, end, score) and an NPZ with
COCO-WholeBody keypoints (133 points; 23-90 are the 68 iBUG face points, mouth
confidence ~0.97) and a valid-box flag. Faces are ~155 px wide.

`prepare_seamless.py` reads tars directly, splits transcripts at pauses > 0.5 s
or 6 s, keeps utterances of 3+ alphabetic CMUdict words with alignment score
>= 0.5 and confident mouth keypoints, resamples to 25 fps (nearest frame) and
crops with the `autoavsr_crops` alignment. Crops look like the LRS3 ones.

## First tar (probe)

`naturalistic/train/0000/0000.tar`: 12 participant files, 125 utterances kept
(5.5 min, 1,015 words). Rejections: 180 too short, 152 low word score, 21 bad
length, 9 non-alphabetic.

| Greedy, same models | LRS3 test-select (660) | Seamless probe (125) |
| --- | ---: | ---: |
| Head C phone error rate | 26.9% | 41.1% |
| Head C blank rate | 75.0% | 79.2% |
| Auto-AVSR CTC word error rate | 31.8% | 46.6% |

Real unseen-video errors, but more of them than on LRS3 test (conversational
speech, a new camera setup, and automatic transcripts add label noise). The
synthetic noise level used before matched 26.9%; these clips are harder, so
training on them may need filtering (e.g. by word score) or mixing with the
calibrated LRS3 inputs. Expected yield: ~125 utterances per GB, so one batch
(~41 GB) gives ~5,000 utterances, ~40k words, ~3.7 h.

Download speed on this machine is ~2-3 MB/s (all servers, not only Hugging
Face; the parallel `hf download` Xet client stalled), so a batch takes ~4 h.

## Two full batches, 2026-11-12

| Batch | Participant files | Utterances | Hours | Words | Head C greedy PER | Mean max prob (emitted) | Entropy (nats/step) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| naturalistic/train/0000 | 500 | 4,846 | 3.07 | 33,891 | 37.6% | 0.880 | 0.141 |
| naturalistic/train/0001 | 481 | 5,070 | 3.19 | 35,648 | 38.2% | 0.880 | 0.140 |
| LRS3 test-select (target) | — | 660 | — | — | 26.9% | 0.903 | 0.130 |
| LRS3 train, synthetic noise 1.3 | — | 30,757 | — | — | 25.6% | 0.892 | 0.180 |

Main rejections: utterances under three words (~11k) and words with alignment
score < 0.5 (~11k). Confidence and entropy match test better than the
synthetic inputs; phone error is higher, partly from automatic-transcript
label noise. Artifacts (not in git): `datasets/seamless-interaction/prepared/
batch000{0,1}/{manifest.jsonl,crops,features,posteriors-headC}`.

## Posterior decoder trained on real unseen posteriors

Same recipe as run 2 of the [posterior decoder](../CompetitiveWordDecoder/POSTERIOR_DECODER.md)
(posterior encoder only, LoRA frozen, lr 1e-3, test-select greedy WER every
300 steps, patience 4); `posterior_decoder.py --train` now takes several sets
with `--train-weights` (one source per batch). Scorer: the same pooled
rescoring (`posterior_decoder_rescoring.py`, final checkpoint, weights fitted
on test-select). Gain vs control = paired difference on test-report against
the zeroed-soft-token control (36.59%), 10,000 bootstrap resamples.

| Training posteriors | Best select loss (zeroed 1.686) | Greedy select WER (baseline 40.49%) | Scorer select | Scorer report | Gain vs control, 95% CI |
| --- | ---: | --- | ---: | ---: | --- |
| LRS3 synthetic (run 2, 2026-11-07) | 1.487 | 42.6-43.9% | **35.69%** | 36.39% | +0.21 [-0.27, 0.68] |
| A: Seamless batch 0000 | 1.550 | 42.9-44.1% | 35.71% | 36.47% | +0.12 [-0.39, 0.63] |
| B: Seamless 0000 + LRS3 synthetic, 1:1 | 1.517 | 43.0-44.4% | 35.89% | 36.14% | +0.45 [0.02, 0.90] |
| B2: Seamless 0000+0001 + LRS3 synthetic, 1:1 | **1.486** | 43.8-46.9% | 35.85% | 36.41% | not computed |

Real unseen posteriors do not change the picture. Greedy generation still gets
worse with posteriors (so the earlier failure was not caused by synthetic
inputs), and as scorers all four encoders are within ~0.2 points on
test-select. B's report number is the best, but it is the worst of the three
on test-select and doubling the real data (B2) removes it, so it is treated as
noise. Text-domain mismatch (conversational Seamless vs TED) likely limits the
Seamless-only run: its training loss stays near 2.2 against ~1.4 on LRS3.

Conclusion: the soft-token posterior encoder is at its ceiling on this test set
whatever the training posteriors; more Seamless data for this route is not
recommended.

## Phone-string decoder trained on real phone errors, 2026-11-13

`CompetitiveWordDecoder/seamless_decoder_rows.py` appends one row per Seamless
clip (prompt = head C greedy phone string, ~38% PER; target = transcript) to the
matched-noise rows: 12,000 + 9,891 = 21,891 rows. Training repeated the
matched-noise recipe exactly (from `vallr-eos-20261010`, 1 epoch, lr 2e-5,
batch 4): 5,473 steps, 63 min, `checkpoints/decoder-matched-seamless-20261113`.
Evaluation reused head C's 20 phone paths (`...-20261113-pipeline`).

| Head C phones, WER | Matched-noise decoder | + Seamless real errors |
| --- | ---: | ---: |
| test-select, top-1 of best path | 40.47% | 41.19% |
| test-select, 5 paths ranked | 38.51% | 38.45% |
| test-select, 20 paths rescored (fitted) | 36.81% | 36.47% |
| test-report, 20 paths rescored (select weights) | **36.63%** | 36.78% |

Test-report paired difference: −0.14 points, 95% CI −0.80 to +0.53 (63 clips
better, 64 worse). Real phone errors from unseen video do not improve the
decoder on LRS3. The new decoder's top-1 report WER (46.31% vs 41.13%) is
inflated by one runaway 214-word generation, which rescoring never selects.

Also fixed: `evaluate_llama_visual.py`, `evaluate_probability_ranker.py`,
`decoder_tolerance.py` and `train_timed_phone_adapter.py` loaded Llama with
mean resizing, which drove the CPU to 95 °C and a ThermalGuard kill once
(the adapters store full `embed_tokens`/`lm_head`, so results are unchanged).
