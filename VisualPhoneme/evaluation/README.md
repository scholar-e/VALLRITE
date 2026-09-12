# Visual-phoneme validation experiments

Run date: 2026-09-11. This is a validation result, not a final test result.

The 123,096-parameter grayscale model trained on GRID speakers 1–8 and used
speakers 9–10 for early stopping. Two inputs were compared without changing
the model:

| Input | Best epoch | Validation PER | Training PER at best epoch |
|---|---:|---:|---:|
| Full face cutout | 5 | 73.01% | 73.65% |
| Mouth region | 10 | **71.55%** | 72.73% |
| Mouth region + 41 VPA coordinates | 20 | **62.92%** | 35.05% |

The mouth-only run improved validation PER by 1.46 absolute percentage points.
The fusion run improved another 8.63 points, or 10.09 points over the full-face
baseline. It stopped after 28 epochs in 863 seconds under a 30-minute wall-clock
budget. The selected fusion checkpoint has 139,480 parameters. No language
model, lexicon, audio, transcript, or test-speaker data entered inference; its
landmarks were extracted from the silent video.

The loader retained 7,895 training clips and all 2,000 validation clips. It
excluded 105 speaker-8 videos whose decoded frame counts were too short for
their CTC targets. Decoder warnings indicate that a small number of retained
MPEG files also contain damaged blocks, though they still produced enough
frames. Speakers 11–12 remain locked for this image model.

This result proves the new path can train and emit variable-length phoneme
sequences and that explicit geometry contributes useful validation information.
It is not yet accurate. Fusion also opened a 27.87-point train/validation PER
gap at the selected epoch, consistent with substantial speaker or corpus
overfitting. The next validation-only comparison should regularize or normalize
the coordinate path and include a coordinate-only ablation. Do not evaluate the
locked speakers while selecting that change.

The mouth-only run used horizontal reflection, while fusion disabled it because
the corresponding left/right landmark permutation is not implemented. The first
fusion gain is therefore exploratory rather than a controlled final ablation.

## Controlled modality and representation ablation

The first controlled suite used the same mouth crop, speaker split, seed,
optimizer, batch size, temporal stack, and no horizontal reflection. The
coordinate models have 52,456 parameters; the image and fusion models have
123,096 and 139,480 respectively.

| Condition | Best epoch | Validation PER | Training PER at best epoch |
|---|---:|---:|---:|
| Constant coordinates (duration/grammar control) | 13 | 71.98% | 73.06% |
| Image only, no reflection | 10 | 71.59% | 71.95% |
| Eye-normalized coordinates only | 19 | 56.88% | 25.88% |
| Eye-normalized image + coordinates | 20 | 62.92% | 35.05% |
| Clip-centered image + coordinates | 20 | 52.59% | 18.65% |
| **Clip-centered coordinates only** | **20** | **44.74%** | **19.07%** |

Clip-centering subtracts the median observed 41-point geometry independently
for each clip. It removes static face shape while retaining coordinate motion.
It improved coordinates-only PER by 12.14 absolute points. Constant coordinates
performed similarly to the image baseline, showing the fixed GRID grammar and
clip duration are a material baseline but do not account for the centered
coordinate result.

The image branch is not yet additive: centered fusion trails centered
coordinates by 7.85 points and has a larger train/validation gap. The next
development comparison should regularize the fusion path and test whether the
image branch can improve centered coordinates, rather than increasing encoder
size. All numbers are single-seed validation results; speakers 11–12 remain
locked.

Repeated MPEG decoding was the main GPU input bottleneck. A persistent uint8
frame cache reduced complete fusion epochs after cache population from roughly
30 seconds to 11–12 seconds without changing model inputs. The cache occupied
6.4 GiB for the train and validation clips. Increasing loader workers from 6
to 16 did not improve training throughput in a 1,024-clip benchmark.

## Top-5 sequence selection

The best coordinate condition was retrained with the same CTC objective but
validated using a Top-5 pruned prefix beam (beam width 16, eight token
expansions per frame). Checkpoints prioritized exact reference inclusion in the
five candidates, with oracle PER@5 as the tie-breaker.

| Selected epoch | Greedy PER | Oracle PER@5 | Exact reference in Top-5 |
|---:|---:|---:|---:|
| 18 | 44.83% | **39.96%** | 3/2,000 (**0.15%**) |

The candidate set reduces oracle PER by 4.86 absolute points compared with the
selected checkpoint's greedy output. Whole-sequence recall remains low because
a GRID reference contains roughly 18 phonemes and every token must be correct
for an exact hit. Epoch 20 had the lowest greedy PER, 44.74%, but slightly worse
oracle PER@5, 40.08%, at the same 0.15% exact recall; Top-5 selection therefore
kept epoch 18. These remain single-seed validation results, and test speakers
11–12 remain locked.

## Oracle improvement stages

The next suite implemented five planned changes: a full oracle curve, explicit
multi-lag coordinate dynamics, stronger coordinate regularization, gated
residual image fusion, and a tiny training-only phoneme-bigram prior for beam
search. All values below use the same 2,000 validation clips, beam width 64,
eight token expansions per frame, and no test speakers.

| Condition | Parameters | Greedy PER | Oracle PER@1 | @3 | @5 | @10 | @20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Centered position, no LM | 52,456 | 44.83% | 44.30% | 41.31% | 39.93% | 38.12% | 36.42% |
| Centered position, bigram 0.25 | 52,456 | 44.83% | 43.17% | 39.43% | 37.68% | 35.44% | 33.28% |
| Regularized motion, bigram 0.25 | 60,808 | 43.74% | 41.83% | 38.32% | 36.73% | 34.67% | 32.57% |
| **Gated image + motion, bigram 0.25** | **149,209** | **43.18%** | **41.40%** | **37.76%** | **36.20%** | **34.00%** | **32.06%** |

Against the original centered-position beam without a language prior, the final
condition improves oracle PER by 3.73 points at N=5 and 4.35 points at N=20.
The stricter comparison against the same 0.25 bigram prior improves PER by 1.48
points at N=5 and 1.21 points at N=20. Exact reference inclusion rises from
0.35% to 0.75% at N=5 and from 1.00% to 1.75% at N=20 in that matched comparison.

The motion-coordinate run selected epoch 32 after reaching 36.82% oracle PER@5
with the narrower training-time beam. The gated run initialized from that
checkpoint and selected epoch 2 at 36.35% with the same narrow beam. Its learned
image gate remained approximately 0.00247, so the measured gain is a small
pixel-based correction rather than a takeover by the image branch. A weight of
0.25 beat 0.50 across the final curve. These are development results from one
seed; exact whole-sequence recall is still low, and the locked speaker test must
wait until model and decoder choices are frozen.

Complete local artifacts are ignored by Git at
`checkpoints/visual-phoneme-fusion/`: `best.pt`, `metrics.json`, and `train.log`.
The selected fusion checkpoint was saved at epoch 20. Earlier checkpoints remain
available in `checkpoints/visual-phoneme-mouth/` and
`checkpoints/visual-phoneme-small/` for the eventual controlled comparison.
