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
