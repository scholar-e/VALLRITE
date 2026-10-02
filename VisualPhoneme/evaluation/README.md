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

## Image pretraining and inner-mouth ablation

Two matched seed-20260912 runs initialized the coordinate path from the selected
motion checkpoint and the image encoders from the no-reflection image baseline.
The coordinate path was frozen for three warm-up epochs. Both runs used batch
64, the same regularization, bigram weight 0.25, and the same 2,000 validation
clips. Test speakers remained locked.

| Condition | Parameters | Selected epoch | Greedy PER | Oracle PER@5 | Exact Top-5 | Learned gates |
|---|---:|---:|---:|---:|---:|---|
| Prior coordinate-dominant gated run | 149,209 | 2 | 43.18% | **36.35%** | 0.75% | image 0.25% |
| Image-pretrained staged fusion | 149,209 | 9 | 43.43% | 36.70% | 0.65% | image 4.88% |
| Aperture-gated inner-mouth fusion | 237,610 | 9 | 44.31% | 37.41% | 0.70% | image 4.91%, inner mouth 4.86% |

The interventions prevented image-gate collapse but did not improve held-out
speaker accuracy. The inner-mouth condition is therefore retained as a negative
ablation and not promoted as the default. Its crop can consume visible tongue
pixels only when lip aperture makes the oral cavity observable; it does not
estimate hidden tongue position. A future tongue-specific loss requires
separately reviewed visibility, mask, or keypoint labels.

## Preliminary LRS3 scaling and staged fusion

The first LRS3 scaling run matched the original GRID training volume with 7,895
training clips and used all 1,140 usable validation clips (44,586 reference
phones). This comparison must not be described as a reproduction of 10% WER:
the earlier GRID result was a 10.09-point *PER improvement*, ending at 62.92%
PER, and GRID's fixed six-word grammar is much easier than unconstrained LRS3.

LRS3 has a median 2.16 frames per target phone, versus roughly 3.4 in the GRID
pilot. A checkpointed `--ctc-upsample-factor 2` therefore repeats temporal
emissions before CTC loss and decoding to provide more alignment states. The
coordinate path was trained first; gated fusion then initialized from its best
checkpoint and froze that path for three image warm-up epochs. Both models used
the same clip-centered motion features, regularization, bigram weight 0.25, and
held-out validation set.

| Condition | Greedy PER | Oracle PER@1 | @3 | @5 | @10 | @20 |
|---|---:|---:|---:|---:|---:|---:|
| Coordinates, 2x CTC, bigram 0.25 | 91.19% | 83.52% | 82.69% | **82.33%** | 81.82% | 81.27% |
| Staged gated fusion, 2x CTC, bigram 0.25 | 94.67% | 84.00% | 83.11% | **82.64%** | 82.06% | 81.51% |

The image residual worsened oracle PER@5 by 0.31 points. Both runs selected
epoch 1 and later drifted toward blank-heavy CTC output even as loss fell. The
2x alignment change improved the early coordinate result but did not solve
collapse.

## Quarter-LRS3 teacher-aligned curriculum

The follow-up froze a quarter-sized source subset: 7,895 LRS3 trainval clips.
Whisper `large-v3-turbo` supplied word timestamps for 7,557 accepted or cached
clips. Its transcript was checked against the official LRS3 transcript, with a
minimum normalized word-sequence agreement of 0.8. Phones still come from the
official transcript and CMUdict; each word's phones are divided uniformly over
the teacher word interval. This is useful weak timing supervision, not a claim
that Whisper produced phone-level ground truth.

After transcript, pronunciation, alignment, and CTC-validity filtering, 6,607
source videos produced 21,106 consecutive word chunks of at most 16 phones
(except when one word itself exceeds the cap). Training used timestamped
nonblank frame cross-entropy as an auxiliary objective. The coordinate model
received five frame-only warm-up epochs, then CTC at weight 0.1 with 2x emission
upsampling. The gated hybrid initialized that checkpoint, froze coordinates for
three image-residual warm-up epochs, and then used the same low CTC weight.
Validation remained the unchanged 1,140-clip AV-HuBERT LRS3 validation set;
teacher timings were used only to construct training examples.

| Quarter-LRS3 condition | Parameters | Selected epoch | Greedy PER | Oracle PER@5 | Image gate |
|---|---:|---:|---:|---:|---:|
| Teacher-aligned coordinates | 69,448 | 15 | 91.48% | 81.16% | n/a |
| Teacher-aligned gated hybrid | 157,849 | 9 | 87.48% | **80.36%** | 6.08% |

The hybrid improves oracle PER@5 by 0.80 points over its coordinate initializer.
Unlike the full-strength CTC attempts, neither low-weight run collapsed to blank
predictions. The global image gate scales an additive image feature correction;
it is not a per-frame confidence, modality percentage, or tongue detector.

### Forced-boundary repeat

The same frozen 7,895-source subset was labelled again after adding the local
transcript-constrained Wav2Vec2 CTC/Viterbi aligner. All 7,895 clips received
acoustic phone intervals; 7,557 passed label quality and 338 remained rejected
for pronunciation coverage. After the unchanged transcript-agreement and
CTC-validity filters, training contained 21,102 chunks. The validation split,
model sizes, curriculum, LM weight, and PER@5 checkpoint selection remained
unchanged.

| Forced-boundary condition | Parameters | Selected epoch | Greedy PER | Oracle PER@5 | Image gate |
|---|---:|---:|---:|---:|---:|
| Coordinates | 69,448 | 12 | 83.73% | 81.28% | n/a |
| Gated hybrid | 157,849 | 24 | 84.49% | **78.81%** | 6.49% |

The hybrid ran for the full 30-minute budget and saved 33 validated epoch
checkpoints; the coordinate stage saved 22 before early stopping. The selected
hybrid improves oracle PER@5 by 2.47 points over its forced-boundary coordinate
initializer. Greedy and oracle trends diverged, so the selected checkpoint's
84.49% greedy PER must not be replaced by the lower greedy PER from a different
epoch. Periodic weights and `metrics.json` retain that complete curve for later
decoder and calibration analysis.

### Full-LRS3 forced-alignment scaling

The official-transcript CTC/Viterbi aligner was applied to the remaining
training manifest without rerunning Whisper. It reused the quarter-set labels,
added 22,117 accepted alignments, rejected 1,079 new clips at the existing
pronunciation-coverage gate, and had zero alignment failures. The resulting
loader retained 28,739 source videos and produced 88,949 chunks after CTC
validity filtering, versus 21,102 chunks in the quarter run.

A plain full-utterance CTC fine-tune was stopped after one epoch because it
collapsed toward blank output (99.41% validation PER and 94.27% oracle PER@5).
The aligned auxiliary objective remained stable. Fine-tuning the selected
quarter hybrid at learning rate 3e-5 stopped after 11 epochs and selected epoch
3: greedy PER 84.67% and narrow-beam oracle PER@5 78.69%, improving the prior
78.81% oracle by 0.12 points. Later epochs reached 83.65% greedy PER but had
worse Top-5 oracle results and were not selected.

A frozen wider decoder sweep (beam 64, twelve token expansions per frame) found
that phone-bigram weight 0.5 gives 78.14% oracle PER@5 and 76.86% oracle PER@20.
These are decoder-assisted oracle values, not deployable top-1 accuracy; the
same setting gives 79.63% oracle PER@1. Alignment coverage alone is therefore
not the remaining primary bottleneck.

## First 10M-model run

The compact GRID and LRS3 architecture names were left unchanged. A separate
`large-gated-fusion` model uses 9,405,321 parameters, a 384-channel fused
representation, and ten full residual temporal blocks. It trained for the same
30-minute ceiling on all 88,949 forced-aligned chunks. Two frame-only warm-up
epochs preceded the aligned-frame plus 0.1-weight CTC objective. Symmetric 10%
modality dropout and a 10% initial image gate kept the expanded image branch
active; the selected epoch's image gate reached 19.42%.

Epoch 5 was selected on strict 39-phone oracle PER@5. No viseme grouping entered
training, beam scoring, or checkpoint selection. Under the training-time beam
(width 16, phone-bigram weight 0.25), strict greedy PER is 79.48% and strict
oracle PER@5 is 75.81%. A frozen beam-64 sweep at weight 0.5 improves strict
oracle PER to 74.80% at N=5 and 73.40% at N=20.

The separately labelled visual-group metric maps both references and hypotheses
through the exhaustive phone-family partition after decoding. It is 70.76%
greedy group PER, 65.33% oracle group PER@5, and 64.00% oracle group PER@20.
These lower group numbers measure recovery up to visual equivalence and must not
be reported as ordinary PER. The selected checkpoint is 36 MB and all ten
validated epoch checkpoints were retained.

### Direct visual-group objective

As a controlled ablation, the same 9.4M architecture and 88,949 chunks were
trained to predict the 17 visual groups directly instead of the 39 phones. The
run stopped after 12 epochs and retained every epoch checkpoint. With the same
frozen beam-64 decoder and language-model weight 0.5, the selected checkpoint
reaches 65.92% group PER at N=1, 64.11% oracle group PER at N=5, and 62.68% at
N=20. This improves the post-hoc grouping of the strict-phone model by 1.22
points at N=5 and 1.32 points at N=20.

These results are **group PER only**. A group prediction deliberately preserves
all member-phone alternatives, so it cannot be converted into ordinary
39-phone PER without a separate disambiguating decoder. The modest gain shows
that direct group supervision helps the intended probabilistic handoff, while
the large remaining error and widening train/validation gap point to visual
representation and generalization as the next bottlenecks.

Evaluation-time modality masking on that selected checkpoint keeps all weights,
examples, and decoder settings fixed. Removing mouth pixels raises oracle group
PER@5 from 64.11% to 70.39%; removing coordinates raises it to 65.88%. At N=20,
the corresponding values are 62.68%, 69.03%, and 64.42%. Both modalities are
therefore complementary, but the larger image-removal penalty shows that pixels
carry most of the discriminative visual evidence despite the learned 23% image
gate. The gate scales feature tensors and is not an attribution percentage.
This masking test measures dependence of the trained hybrid; separately trained
unimodal models remain a different, more expensive ablation.

### Lightweight patch-transformer ablation

The CNN frame encoder was replaced by a three-layer spatial transformer using
16-by-16 patches (36 tokens for each 96-by-96 mouth frame). Landmarks, the
384-channel temporal decoder, group targets, optimizer, data, regularization,
and 30-minute training ceiling were unchanged. The resulting model has
10,403,795 parameters and selected epoch 8 from 11 validated epochs.

Under the matched beam-64 evaluation, the transformer reaches 66.12% group PER
at N=1, 64.22% at N=5, and 62.71% at N=20. The CNN reaches 65.92%, 64.11%, and
62.68%, respectively. The scratch-trained transformer is therefore effectively
tied but consistently worse; it is not promoted over the CNN. A future
transformer experiment should test relevant visual-speech pretraining rather
than adding more randomly initialized attention layers.

### Auto-AVSR visual-speech pretraining

The published Auto-AVSR LRS3 visual-only checkpoint was downloaded from the
official model zoo and all 120 Conv3D/ResNet-18 frontend tensors were loaded
without importing its legacy ESPnet stack. The resulting hybrid has 20,318,419
parameters. After one frozen-frontend warm-up and a batch-32 resumed fine-tune,
epoch 3 reaches 60.97% group PER at N=1, 59.18% at N=5, and 57.69% at N=20 with
the matched beam-64 evaluation. This improves the scratch CNN by about five PER
points and confirms that relevant visual-speech pretraining matters more than
replacing the frontend with a randomly initialized transformer.

A separate group-to-word evaluation mapped every CMUdict pronunciation into
the 17-group inventory and used the bounded pronunciation-only lexical decoder
(beam 16, lexical beam 32). Across all 1,140 development-validation clips it
produces 178.85% top-1 WER and 178.45% oracle WER@5. Insertions allow WER to
exceed 100%. This is the VALLRITE group checkpoint plus its simple lexicon; it
is not the upstream Auto-AVSR word decoder or its published 19.1% LRS3 test WER.
The result shows that visual grouping needs a strong contextual word decoder or
parallel strict-phone evidence rather than direct pronunciation-only lookup.

Complete local artifacts are ignored by Git at
`checkpoints/visual-phoneme-fusion/`: `best.pt`, `metrics.json`, and `train.log`.
The selected fusion checkpoint was saved at epoch 20. Earlier checkpoints remain
available in `checkpoints/visual-phoneme-mouth/` and
`checkpoints/visual-phoneme-small/` for the eventual controlled comparison.
