# VALLRITE Technical Program

This is the detailed implementation and research plan for engineers and AI
agents working in this repository. The root README is the short introduction;
the VPA directory contains the narrower visual-alphabet specification.

Technical audit: 2026-09-05. Progress update: 2026-09-09. This revision
distinguishes verified implementation, published precedents, and proposed
experiments. It supersedes stronger claims in earlier conversation and the
initial VPA draft where they conflict. Sources and comparable systems are linked
below; no head-to-head accuracy comparison has yet established the best
architecture for this repository.

The audit's central prediction — that the current 16-frame/eight-step
configuration cannot emit sentence-length phoneme sequences — has since been
measured rather than merely inferred; see section 3.

## 1. Objective

Build an accurate, interpretable, visual-only speech recognizer that extracts
visible speech evidence, preserves ambiguity, converts it into calibrated
phonetic and lexical alternatives, uses linguistic context without discarding
the visual signal, and ultimately runs offline on a phone.

The central hypothesis is that VALLR's two-stage phoneme-centric architecture
can be improved by replacing its single-sequence handoff with a probabilistic,
temporally structured representation.

## 2. Baseline and terminology

The published VALLR design is not purely GPT-2. Its primary reported system is:

```text
16 video frames
  -> face/lip preprocessing
  -> ViT-style visual encoder
  -> temporal convolution/pooling adapter
  -> CTC phoneme head and beam search
  -> one selected phoneme sequence
  -> LoRA-tuned Llama 3.2-3B
  -> sentence
```

GPT-2 Small is an ablation. CMUdict derives pronunciation labels for language
training; it is not described as inference-time dictionary search. The paper
says CTC beam search is used but does not describe passing the beam scores or
lattice into the LLM.

These are descriptions of the paper, not a verified reproduction. The
[ICCV paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Thomas_VALLR_Visual_ASR_Language_Model_for_Lip_Reading_ICCV_2025_paper.pdf)
reports 18.7% LRS3 WER for Llama 3.2-3B; its GPT-2 result is internally
inconsistent (33.9% in Table 5 versus 23.9% in the surrounding text).
The local upstream Git version of `Models/Llama.py` selects
`meta-llama/Llama-3.2-3B-Instruct` and WikiText-2. Pin the paper version and
resolve preprocessing, chunking, vocabulary, and language-adapter availability
before claiming those reported numbers are reproducible here. The paper's
small labeled-video budget does not count all language-model pretraining.

Keep these separate:

- **CTC search:** finds label sequences through frame posteriors.
- **Pronunciation search:** maps phoneme sequences to possible words.
- **Language search:** scores word sequences in context.
- **Generative correction:** rewrites candidates, with hallucination risk.

## 3. Current repository state

Implemented:

- published VALLR visual checkpoint loading;
- compatibility migration for older VideoMAE attention-bias names;
- offline visual, text, and pipeline modes in `vallrite.py`;
- per-step top-five phoneme probabilities;
- local Qwen3.5-2B integration;
- Qwen-compatible LoRA targets;
- CMUdict single-word LoRA training;
- AVSpeech manifests and incremental clip materialization;
- a real audiovisual CPU smoke test; and
- the initial VPA specification.

Added since the 2026-09-05 audit:

- a dependency-light VPA extractor (`VisualPhoneticAlphabet/core.py`,
  `__main__.py`) emitting schema-validated `vpa-0.1` JSONL: eight geometry and
  motion features, provisional gesture intervals, quality/provenance fields, and
  41 normalized landmarks (20 outer lip, 20 inner lip, chin) plus their
  original-pixel coordinates;
- automatic face/neck isolation (`crop.py`): OpenCV cascade detection,
  conservative single-track association with explicit gaps, letterboxed crop
  video, and a timestamped sidecar; usable inline (`--face-neck`) or standalone;
- a frozen phoneme-accuracy harness (`evaluate.py`) with Levenshtein PER,
  alignment records, checkpoint/media hashes, and retained raw posteriors;
- the first measured accuracy result (below);
- a no-build browser inspector (`web/`) for VPA records, including video-position
  mouth overlay and seven exploratory shape measurements in `shape.js`;
- a verified GRID pilot corpus: 12 speakers, 12,000 clips, ~17 GB, 36 archives
  hash-recorded, fixed speaker-disjoint split (1–8 train, 9–10 validation,
  11–12 test) with a `clips.jsonl` media/alignment index; and
- 16 deterministic tests (11 dependency-free, 5 requiring OpenCV) plus a Node
  test for the derived shape measurements.

### First measured result

Two official GRID demonstration clips, greedy CTC, no language model:
**90.625% transcript-derived PER (29 edits / 32 reference phones)**, identical
before and after automatic face/neck cropping; roughly 87.9% using the best
listed pronunciation alternative per clip. Errors are dominated by deletions,
not substitutions. This is a two-clip diagnostic against dictionary-derived
reference phones — not a held-out benchmark, not a reproduction of official
VALLR evaluation, and not a measure of general accuracy. Details, per-clip
predictions, and reproduction commands are in
[VisualPhoneticAlphabet/evaluation/README.md](VisualPhoneticAlphabet/evaluation/README.md).

Two audit predictions were confirmed by this run: the eight-step output cannot
carry 16-phone sentences even with perfect labels, and the cropper is not yet
reliable (37.33% of frames retained on one clip versus 100% on the other, with
missing frames passed to the model as black images and no missing-frame mask).

Known limitations:

- runnable inference currently uses greedy CTC collapse;
- frame top-k values are not a sequence lattice;
- Qwen receives probabilities as text, not learned numerical features;
- Qwen has not trained on real VALLR errors or lattices;
- the dictionary adapter covers only an initial 1,024-entry subset;
- face tracking exists but is a lightweight cascade tracker with measured
  failures, no active-speaker selection, and no missing-frame signal to the model;
- VPA emits no learned phoneme predictions: `phoneme_hypotheses` and
  `teacher_labels` remain empty reserved interfaces, and gesture thresholds are
  untuned;
- GRID has word alignments only; phoneme timings have not been generated, and
  no VALLRITE training run has used the pilot corpus yet;
- the GRID `clips.jsonl` index and `verification.json` were produced outside
  version control and need a checked-in script before they count as reproducible;
- official held-out PER/WER has not been reproduced; and
- no mobile runtime has been selected or benchmarked.

No smoke-test transcript should be presented as an accuracy result, and the
two-clip PER above is a diagnostic, not a baseline number to improve against.

### Audit findings that affect the implementation order

- **Sequence length:** `vallrite.py` samples 16 frames across the entire clip.
  With default VideoMAE settings, `Models/VALLR.py` downsamples 1,568 flattened
  spatial/temporal patch tokens to eight CTC positions. A meta-tensor shape
  check confirmed this on 2026-09-05. One forward pass therefore cannot emit
  more than eight nonblank phonemes. Adjacent repeated labels need extra CTC
  positions. Longer sentences need verified chunking/aggregation or a revised
  temporal architecture; increasing beam width cannot fix this bound.
  **Confirmed end to end on 2026-09-09:** the GRID diagnostic returned two to
  six phonemes against 16-phone references, with every clip's error budget
  dominated by deletions. This is the single highest-priority defect; no
  decoding, lexical, or reranking work can compensate for it.
- **Time semantics:** downsampling a flattened patch axis does not establish
  a frame-to-output timing map. Do not stamp these eight logits with evenly
  spaced frame times. Verify token layout and preserve a temporal axis before
  attempting timed lattices or framewise teacher distillation.
- **Input correctness:** the CLI resizes full RGB frames and supplies float
  values on the 0–255 scale. Check this against the checkpoint's actual training
  preprocessing; neither standard ImageNet normalization nor raw pixels should
  be assumed correct without evidence. Verify face cropping and sampling too.
- **Qwen identity:** both the downloaded model card and the
  [official card](https://huggingface.co/Qwen/Qwen3.5-2B) describe a post-trained
  model. Previous references to an untouched base model were incorrect. It lacks
  VALLR task adaptation, but already has instruction post-training. Current
  inference bypasses its chat template. Test the documented template and
  non-thinking mode before attributing repetitive output solely to missing
  phoneme training. The existing word adapter needs its matching training prompt.
- **Adapter evidence:** the 1,024-entry run is a training smoke test. It has no
  held-out evaluation, and two example outputs do not measure generalization.
  Its attention-only LoRA targets differ from the broader legacy trainer. It
  saves adapter weights, but no resumable optimizer checkpoint. Report complete
  epochs, grouped train/eval splits, and before/after metrics before expanding it.
- **Media timing:** the downloader uses FFmpeg stream-copy cuts and checks file
  existence when skipping downloads. Audit presentation timestamps, audio/video
  offsets, cut boundaries, and file integrity before using these clips as
  alignment supervision. A playable 4.64-second file alone does not prove exact
  alignment with a 4.62-second annotation.
- **Evaluation data (2026-09-09):** the GRID pilot supersedes AVSpeech as the
  near-term evaluation corpus. It ships transcripts and word alignments, has a
  fixed speaker-disjoint split, and needs no YouTube availability. It is also
  small-vocabulary, fixed-grammar, studio-recorded, and 360×288, so it bounds
  what a GRID result can claim about general lip reading. Keep AVSpeech for
  scaling work; do not report GRID numbers as LRS2/LRS3-comparable.

## 4. Target architecture

```text
                              TRAINING ONLY
 transcript -> pronunciation alternatives -----+
 audio -> phoneme teacher -> alignment ---------+----+
                                                       |
                                                       v
video -> tracking -> visual encoder -> VPA features/phoneme posterior
                                      |
                                      v
                             CTC prefix beam/lattice
                                      |
                                      v
                     dictionary + pronunciation WFST
                                      |
                                      v
                           compact word lattice/n-best
                                      |
                         +------------+-------------+
                         |                          |
                  small neural LM             Qwen baseline
                         +------------+-------------+
                                      |
                         calibrated constrained choice
                                      |
                                  transcript
```

This diagram is an experimental candidate. Start with a corrected video/CTC
baseline and add branches independently. It is not necessary to implement every
branch to obtain a useful system. Audio is used only in training for this
product; final inference remains video-only.

## 5. Visual Phonetic Alphabet

VPA is a visual-articulation interchange layer, not merely another static
viseme list. It synthesizes prior work on visemes, sequential visual units,
facial articulatory features, landmark motion, learned hidden speech units, and
confusion-derived mappings.

Its intended combination is:

```text
continuous camera-observable measurements
  + dynamic gesture intervals
  + learned visual-unit embeddings
  + pose/quality/visibility confidence
  + ranked IPA and ARPAbet candidates
  + explicit unobservable properties
  + a timed probabilistic lattice
```

The draft schema and gesture inventory live in
[VisualPhoneticAlphabet/README.md](VisualPhoneticAlphabet/README.md).
They need the corrections in this audit before becoming a frozen specification.
There is no evidence yet that this combination is unique or improves WER.

### VPA implementation rule

Do not replace learned visual representations with a few handcrafted values.
Use a spatiotemporal video encoder as the primary signal and VPA measurements
as auxiliary targets, structured outputs, and diagnostics.

The starting feature families are:

- lip aperture, width, aspect ratio, curvature, and displacement;
- closure, compression, labiodental contact, teeth and cavity visibility;
- jaw, lip-corner, optical-flow, opening, closing, hold, and release motion;
- sparse tongue visibility; and
- head pose, tracking confidence, blur, occlusion, and speech activity.

Keep a feature only if it improves speaker-held-out phoneme recall,
calibration, or WER at acceptable cost. Remove features that mainly encode
identity, expression, pose, or recording conditions.

No optimal feature list has been established. Begin with image features plus
normalized lip contours and their trajectories. Compare explicit optical flow
against motion already learned by the video encoder; extra streams may add
cost without information. Frontal 2D width/height is only a proxy for rounding;
protrusion requires depth assumptions. Facial landmarks generally do not locate
teeth or an occluded tongue. Contacts need dedicated image annotations or
segmentation. Darkness is sensitive to lighting and camera exposure, not simply
articulation. Keep measurement units, normalization, confidence, and missingness
separate; low visibility must not be encoded as zero articulation.

Build a small manually reviewed video-only feature set with repeat annotations
before generating mass pseudo-labels. Measure annotator agreement and tracking
error. Audio phoneme labels do not prove a visible gesture occurred. Learned
clusters also need nuisance/identity controls and comparison with phonetic
decoding, not just an appealing cluster visualization.

## 6. Phonetic representation

Maintain three connected vocabularies:

1. ARPAbet for existing-checkpoint and CMUdict compatibility.
2. IPA as a documented rendering/interchange mapping, initially optional.
3. VPA gestures for camera-observable evidence.

Keep the checkpoint's 39 nonblank ARPAbet phones plus one CTC blank intact.
Relabeling them with IPA adds no new evidence or pronunciation coverage.
Accent-specific allophones, stress, length, and syllabification require an
explicit inventory and data source; broad ARPAbet-to-IPA conversion cannot
recover distinctions the source omitted. Preserve Unicode combining marks and
define phone tokens independently of Unicode characters and LLM subwords.

A bilabial closure is direct evidence; voicing/nasality are inferred properties.
However, phonemes within a conventional viseme group are not necessarily
visually identical over a moving utterance. Human experiments found above-chance
discrimination within visemes. Avoid hard masks that forbid the model from
learning subtle temporal cues. Store direct observation and inference separately,
and assess visibility per observation rather than declaring a phoneme property
universally invisible. [Files et al., 2015](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2015.00878/full)

Preserve CMUdict stress in transcript/audio supervision, but do not require the
visual model to infer stress directly in the first experiment. Keep both
stress-preserving and stress-stripped training views and test them. Lexical
stress and realized prosody differ. CMUdict supplies lexical pronunciations,
not pronunciation frequencies or a universal accent inventory.
[CMUdict](https://github.com/cmusphinx/cmudict)

Version all vocabularies and conversion tables. CTC blank means no label emitted
at that step; it is not silence, padding, or occlusion. The existing `<pad>` ID
serves as blank, but metadata must distinguish these concepts without casually
adding untrained output classes. Gesture probabilities are multi-label, while
phone softmax classes are mutually exclusive; they need different score fields.

## 7. Video preprocessing

Build a reproducible pipeline that:

1. detects and tracks the active speaker;
2. emits stable mouth-only and lower/full-face crops;
3. normalizes rigid head motion without deleting speech deformation;
4. preserves timestamps and frame rate;
5. extracts dense landmarks and local motion;
6. measures pose, blur, occlusion, and crop/tracking failure; and
7. records provenance and preprocessing versions.

Compare mouth-only and larger facial regions. Cheeks and jaw may help but can
leak identity or expression. Never silently interpolate long tracking failures;
mark them missing so filtering and models can respond explicitly.

Status: `VisualPhoneticAlphabet/crop.py` covers (1) weakly, (2) as a rectangular
face/neck context region rather than a mouth-only crop, (4), and part of (6) —
detection count, clipping, track segment, and explicit missing frames. It does
not stabilize rigid motion, select an active speaker, or measure pose, and its
measured retention on GRID was 37% on one of two clips. Landmarks and local
motion (5) come from the separate VPA extractor.

Use fixed-rate chronological windows with overlap where needed; document how
repeated phonemes and boundary words merge across windows. Uniformly resampling
an entire long clip to 16 frames cannot preserve rapid speech events. Start with
a known speaker crop before adding speaker selection. Silent visual activity
does not guarantee that the visible person is the source of recorded audio.

## 8. Supervision and visual training

### Transcript path

- Preserve original text alongside normalized text.
- Generate every permitted pronunciation rather than taking only the first.
- Retain stress and word boundaries.
- Represent out-of-vocabulary terms explicitly.

### Training-only audio path

- Run a strong audio ASR/phoneme teacher.
- Use a forced aligner for candidate audio boundaries; retain video time and
  measured synchronization offsets separately.
- Preserve posterior probabilities rather than only hard labels.
- Record teacher/transcript disagreement and synchronization quality.
- Reject or down-weight unreliable examples.

An ASR teacher, a phone recognizer, and a forced aligner are different tools.
Word/subword ASR output is not a phone posterior. An aligner conditions on the
supplied transcript, so agreement with that transcript is not independent
verification. Map teacher and student inventories and time resolutions before
KL distillation; never compare unrelated softmax classes directly. Acoustic
boundaries are approximate supervision for visual gestures: coarticulation can
begin before and persist after the associated sound. Use soft windows or learned
lag, and validate on video annotations. Start with sequence supervision if
reliable temporal correspondence is unavailable.

Split by speaker before teacher generation and tuning. Generate out-of-fold
visual predictions for downstream decoder training so the decoder learns
realistic errors rather than memorized, artificially clean outputs.

Out-of-fold generation is one option; a fixed disjoint training partition for
decoder examples is cheaper initially. Keep final test data untouched and freeze
teachers/configuration before evaluation. Running a frozen teacher is not itself
training leakage, but fitting filters or selecting teachers using test outcomes is.

### Multi-task model

Train a shared encoder with phoneme CTC, VPA feature, gesture event, boundary,
audio-teacher distillation, and temporal-consistency objectives. Optionally use
identity-adversarial or mutual-information regularization. Select loss weights
on end-to-end validation WER and calibration, not auxiliary accuracy alone.

Treat this loss list as a menu. Add one supported objective at a time, mask
missing labels, and check gradient conflicts. Audio teaching cannot create
visual evidence for acoustically distinctive events without visual correlates.
Use finite distillation weights and compare hard pseudo-labels, soft phone targets,
and latent audio features. AV-HuBERT and RAVEn supply strong comparable learned
representations; their targets are not a human-readable, visual-only alphabet.
[AV-HuBERT](https://arxiv.org/abs/2201.02184), [RAVEn](https://arxiv.org/abs/2212.06246)

## 9. CTC uncertainty

Per-frame top-k tokens are useful diagnostics but are not a proper sequence
lattice: they omit blank/repetition rules, timing, and cumulative sequence
probability.

Implement prefix beam search that emits:

- complete phoneme hypotheses;
- cumulative log probabilities;
- blank/non-blank prefix scores;
- token timing estimates;
- pruning metadata.

CTC sums the probabilities of all alignments that collapse to the same label
sequence; taking only the best alignment is not equivalent. Prefix search needs
separate blank/nonblank bookkeeping and correct repeated-label handling. Check
it against exhaustive enumeration on tiny examples. Beam pruning approximates
that sum. [Original CTC formulation](https://www.cs.toronto.edu/~graves/icml_2006.pdf)

Start with n-best sequences. An n-best list is not a time-aligned lattice or
confusion network: those require retained graph arcs/alignment information and
explicit construction. Label token times as estimates, and only expose them
after resolving the output-axis problem in section 3. Preserve raw log scores,
score provenance, pruning settings, and any normalization separately.

Calibrate on held-out speakers. Define the event being calibrated (e.g., word
correctness) before reporting ECE or Brier score; unaligned frame labels cannot
provide frame-level calibration truth. Temperature scaling is a baseline, not
a guarantee for CTC sequences or unseen domains.
[Calibration baseline](https://arxiv.org/abs/1706.04599)

Report oracle n-best PER/WER at fixed candidate and compute budgets. Larger
nested candidate sets automatically cannot worsen oracle error; that alone is
not better modeling. Search coverage, ranking accuracy, and calibration are
different properties. Scores normalized within a pruned beam are not global
posterior probabilities.

## 10. Lexical and sentence decoding

Qwen is one benchmark, not the whole decoder.

### First-pass weighted search

Test a pronunciation lexicon and n-gram word language model in weighted search. The
lexicon should contain CMUdict alternatives, stress-preserving and agnostic
paths, trained pronunciation probabilities, reductions, documented accent
variants, proper names where permitted, and an unknown-word fallback.

Compare post-hoc search on phone n-best with lexical constraints applied during
beam expansion: a small phone beam can already have discarded the correct word.
Keep an unrestricted/OOV route so the lexicon does not force a familiar but
incorrect sentence. Do not silently expand the checkpoint's phone vocabulary
when adding stress-preserving dictionary entries; use an explicit projection.

Use an explicit, validation-tuned log-linear score:

```text
S = log P_visual
  + alpha * log P_pronunciation
  + beta * log P_ngram
  + gamma * word_count
```

These terms are features, not automatically independent Bayesian factors.
Phone posteriors already contain training-set priors. Avoid double-counting the
same LM or correlated VPA evidence, and document any prior correction rather
than assuming it helps. Pronunciation weights need a defined normalization:
MFA's lexicon weights, for example, normalize the most likely pronunciation to
one, not the sum to one.
[MFA pronunciation weights](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/implementations/lexicon_probabilities.html)

WFST is a search representation, not an alternative intelligence model. HMMs
model state sequences; n-grams supply finite-context language priors; neural
models can also score paths. Test a simple beam+lexicon implementation before
committing to full graph composition. Measure graph size and latency: weighted
search is auditable but not automatically small or fast.

### Second-pass comparison

Compare:

1. greedy CTC, prefix beam, and lexicon/n-gram search without neural reranking;
2. small neural-LM candidate scoring or a discriminative n-best reranker;
3. a compact lattice-aware Transformer if graph coverage warrants it; and
4. Qwen3.5-2B candidate scoring, candidate selection, and task-trained generation.

Serialized scores are a valid low-cost baseline, not inherently unusable.
Compare candidates alone, candidates with textual scores, explicit score fusion,
and learned numerical/phonetic embeddings. Lattice attention and confidence-aware
LLM rescoring have ASR precedents, but neither proves a gain here.
[Lattice attention](https://arxiv.org/abs/2111.10157),
[ProGRes](https://arxiv.org/abs/2409.00217)

Train on clean phonemes only as a warm start, then synthetic errors and held-out
visual predictions. Synthesize insertions, deletions, repetitions, timing and
calibration errors as well as within-viseme substitutions. Label artificial
scores as synthetic; match their distribution to real model outputs. Avoid
training only with the true answer inserted into every candidate set.

Evaluate constrained candidate selection separately from free rewriting.
Prefer conservative selection unless generation shows a reliable WER gain
without more unsupported substitutions.

Selection cannot beat the candidate set's oracle WER; generation can recover
missing words but can also overcorrect. Track both. The current Qwen CLI is a
conditional generator, not yet a trained reranker. In experiments, distinguish
unconditional text likelihood from phoneme-conditioned likelihood and avoid
adding the visual score twice through overlapping scoring terms.

Qwen3.5-2B is a plausible baseline/teacher, not an established best choice or
proven excessive size. Its checkpoint includes a vision encoder unused by this
text path. Benchmark task accuracy, memory, and latency against smaller models;
general-language benchmark scores do not establish VSR quality.

Also include a modern direct word/subword visual recognizer as an external
baseline. An obligatory phoneme bottleneck can discard useful information;
VPA may work better as auxiliary supervision than as the decoding interface.

## 11. Data program

Use only documented, permitted data. Track source, license, revisions, hashes,
speaker identity, language/accent metadata, filtering, usable hours, and split.

For every corpus:

1. deduplicate videos and near-duplicate segments;
2. create speaker-disjoint splits before model work;
3. validate synchronization and transcripts;
4. filter unstable faces and weak visibility;
5. retain quality metadata for hard examples;
6. balance speakers, poses, accents, and recording conditions; and
7. report usable rather than advertised hours.

AVSpeech is distributed as manifests referencing public YouTube content.
Preparation must be resumable and log unavailable entries. Do not bypass
private, deleted, geographical, age, or authentication restrictions.

The local AVSpeech download is manifests plus one smoke-test clip, not a
materialized large training corpus. AVSpeech's five CSV fields specify video ID, start/end,
and face location; they do not supply transcripts or explicit speaker IDs.
Preserve its official train/test separation, document additional split auditing,
filter language for this English checkpoint, and create reviewed references or
clearly labeled ASR pseudo-transcripts. Do not treat teacher-generated test text
as human ground truth. Validate cuts and synchronization before alignment.
[AVSpeech distribution](https://looking-to-listen.github.io/avspeech/download.html)

The GRID pilot is materialized locally: 12 speakers, 12,000 clips, ~17 GB of
360×288 video, 50 kHz audio, and word alignments, with archive hashes and a
fixed 8/2/2 speaker split recorded in
`datasets/grid-pilot/`. Treat it as a controlled development corpus, not a
general benchmark: fixed six-word grammar, 51-word vocabulary, studio lighting,
frontal pose, and no accent or condition diversity. Word alignments are not
phoneme supervision, and the split is a local pilot convention rather than a
published protocol. Media stay git-ignored.

ASR pseudo-labeling for scaling VSR is established by Auto-AVSR. Its precedent
supports testing the strategy, not assuming every available video improves
accuracy. Match duration, domain, supervision, and pretrained-data accounting
when comparing data efficiency; the earlier “1%” claim is not a controlled ratio.
[Auto-AVSR](https://arxiv.org/abs/2303.14307)

Run controlled data-scaling experiments so data gains are not mistaken for
architectural gains.

## 12. Evaluation

Reproduce upstream VALLR before claiming improvement. Pin commit, checkpoint
hash, preprocessing, split, seeds, dependencies, and decoder parameters.

| Component | Required metrics |
|---|---|
| Face pipeline | failure rate, usable frames, landmark error |
| VPA | annotation agreement, feature error by speaker/pose/quality, incremental WER |
| Visual model | PER, oracle n-best PER, search coverage, defined-event calibration |
| Lexical search | oracle WER, search error, OOV rate |
| Reranker | WER/CER, correction and miscorrection rates |
| End to end | WER by speaker, accent, pose, and quality |
| Deployment | size, RAM, latency, real-time factor, energy |

Required ablations:

- greedy versus beam versus lattice;
- frame top-k versus n-best versus lattice;
- probabilities removed versus retained;
- identical-inventory ARPAbet/IPA rendering as a control, then genuinely expanded targets;
- learned pixels versus each VPA feature family;
- mouth-only versus lower/full face;
- transcript versus audio-teacher versus combined labels;
- clean versus synthetic-error versus real-lattice decoder training;
- WFST versus compact Transformer versus Qwen;
- constrained selection versus rewriting;
- each data-volume increment; and
- full precision versus each compressed candidate.

Use confidence intervals or matched significance tests for WER changes. Publish
negative ablations; more symbols, parameters, or data are not automatically
better.

Use identical normalization and splits for every decoder; resample confidence
intervals by speaker/recording, not independent frames. Evaluate abstention with
risk-versus-coverage curves so lower error from refusing more clips is visible.
Run ablations in stages; this is not a requirement for a full combinatorial grid.

For dictionary adaptation, split by normalized word before expanding alternate
pronunciations, deduplicate records, and report held-out exact-word accuracy.
A phone string can have several valid homophones: single-word training cannot
determine the intended spelling without context. Test valid-answer sets and
sentence disambiguation separately. Loss reduction on training batches is not
evidence of better held-out lip reading.

## 13. Mobile program

Choose a reference phone and define budgets for package size, peak RAM, first
partial transcript, real-time factor, sustained thermals, energy per minute,
and acceptable WER regression.

Investigate smaller video encoders, adaptive frames, compact crops, streaming,
INT8/INT4 quantization, structured pruning, low-rank factorization,
teacher/student distillation, runtime operator support, and WFST plus a small
reranker instead of a 2B on-device LLM.

Face detection, tracking, preprocessing, lattice search, and decoding all count
toward resource measurements—not only neural forward time.

Compression is conditional on a useful baseline, not a promised final step.
Quantized weights do not describe peak RAM: activations, caches, runtime buffers,
and search graphs matter. Verify supported operators and sustained on-device
behavior. MobiVSR is a mobile-oriented word-level precedent, not proof that this
continuous-sentence pipeline will run in real time.
[MobiVSR](https://arxiv.org/abs/1905.03968)

## 14. Delivery sequence

### Phase A: reproduce VALLR — in progress

- Resolve 16-frame/eight-position capacity, temporal layout, and checkpoint
  preprocessing. *Capacity bound confirmed empirically; not yet fixed. Temporal
  layout and preprocessing still unverified.*
- Recover upstream chunking/evaluation or document an explicitly modified baseline.
- Obtain authorized evaluation data or document an alternative. *Done: GRID
  pilot, 12 speakers, speaker-disjoint split, hashes recorded.*
- Measure PER and WER; distinguish independent baseline results from paper
  reproduction. *PER tooling and a frozen two-clip diagnostic exist; a
  speaker-disjoint GRID test-split run and WER tooling do not.*
- Verify Qwen chat formatting and held-out dictionary behavior.
- Add deterministic tests and experiment tracking. *Tests exist for VPA geometry,
  cropping, and error metrics; no experiment tracking yet.*

Exit: a trusted, repeatable baseline.

### Phase B: uncertainty interface

- Implement and test CTC prefix beam search.
- Define versioned n-best/lattice records.
- Add timing and calibration.
- Measure top-1 and oracle performance.

Exit: tested approximate CTC sequence scores, with explicit pruning and time semantics.

### Phase C: VPA MVP — partially started ahead of B

- Implement stabilization, landmarks, motion, and quality features. *Landmarks,
  geometry, motion derivatives, and quality/provenance fields are implemented;
  crop stabilization, optical flow, contacts, yaw/pitch, tracking confidence,
  and speech activity are not.*
- Emit schema-valid VPA JSONL. *Done, plus a browser inspector for it.*
- Add auxiliary heads and speaker-disjoint ablations. *Not started; VPA is
  currently a diagnostic stream with no learned head and no fusion with VALLR.*

Do not expand the feature inventory further until the Phase A capacity defect is
fixed; more measurements cannot be evaluated against a decoder that cannot emit
sentence-length output.

Decision: retain VPA features only with held-out benefit at a fixed resource
budget; otherwise keep them optional diagnostics or drop them.

### Phase D: supervision and scale

- Add audio teacher and forced alignment.
- Generate out-of-fold labels and lattices.
- Materialize permitted data and run scaling studies.

Decision: retain beneficial supervision, with data and architecture effects
separated; a null result should not block the rest of the product.

### Phase E: grounded decoding (start cheap baselines immediately after B)

- Build pronunciation and n-gram search.
- Integrate WFST/lattice decoding.
- Train compact neural and Qwen rerankers.
- Measure correction versus hallucination.

Decision: choose the best validated accuracy/resource tradeoff, including the
simpler baseline if probability-aware neural decoding does not improve it.

### Phase F: deployment

- Select the smallest decoder inside the WER budget.
- Distill, quantize, export, and test the complete phone pipeline.

Exit: offline real-time operation within documented accuracy and resource
limits.

## 15. Immediate backlog

Revised 2026-09-09. Items 1–3 of the previous list are partly discharged:
held-out speaker-disjoint data exists, PER tooling exists, and the temporal
capacity question is answered. The remaining order is:

1. **Fix output capacity.** Establish the checkpoint's real token/time layout,
   then either window the input with documented overlap and merge rules, or
   revise the temporal adapter. Verify against reference length before any other
   accuracy work.
2. Verify checkpoint input preprocessing (crop geometry, pixel scale,
   normalization, frame sampling) against the published training recipe; the
   current 0–255 full-frame path remains unjustified and confounds every
   measured error.
3. Run a real speaker-disjoint GRID baseline on speakers 11–12 with frozen
   configuration, and add WER tooling alongside the existing PER tooling.
4. Check into version control the script that produced the GRID `clips.jsonl`
   index and `verification.json`, and generate phoneme alignments from the
   50 kHz audio and word alignments.
5. Make missing frames explicit to the model instead of black images, and
   measure crop failure rate across the pilot corpus rather than two clips.
6. Test checkpoint migration, CTC collapse, and exhaustive tiny-sequence beam
   correctness; define n-best records and score provenance. Defer timed graphs
   until timing is valid.
7. Check Qwen prompt/template compatibility and held-out dictionary adaptation.
8. Benchmark beam+lexicon/n-gram and cheap neural scoring before more
   large-model training.
9. Collect a small manually reviewed VPA feature set; add auxiliary heads only
   once items 1–3 give an evaluable baseline.
10. Compare audio teachers/aligners, then expand permitted media with quality checks.
11. Train selected decoder candidates on realistic visual errors and measure ablations.
12. Evaluate compression only after an accuracy baseline and device budget exist.

## 16. Guardrails

- Do not describe execution success as recognition accuracy.
- Do not train or tune on test speakers.
- Do not report oracle results as deployable results.
- Do not call frame top-k output a sequence lattice.
- Do not infer invisible articulation without uncertainty labels.
- Do not let an LLM silently discard visual likelihoods.
- Do not bypass data access or source-video restrictions.
- Do not claim VPA is wholly unprecedented; cite relevant prior art.
- Do not optimize for mobile using parameter count alone.

## 17. Completion criteria

Product success requires:

1. an auditable VALLR baseline;
2. a selected architecture with speaker-disjoint accuracy and calibrated uncertainty;
3. documented positive or negative evidence for the major proposed additions;
4. conservative behavior on uncertain and out-of-domain inputs; and
5. offline operation on a named phone within fixed resource budgets.

VPA, audio distillation, lattices, and Qwen are hypotheses, not mandatory winners.
Do not keep an ineffective module merely to satisfy the original design.

Until then, VALLRITE is a research prototype rather than a validated assistive
transcription product.

## 18. Comparable work and evidence boundaries

This is a targeted technical comparison, checked 2026-09-05, not an exhaustive
novelty review. A comparable component does not validate our complete pipeline.
ASR results below concern audio recognition unless explicitly marked VSR.

| Proposed component | Closest checked precedent | What it establishes / remaining gap |
|---|---|---|
| Updated phoneme-to-text LLM | [VALLR, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Thomas_VALLR_Visual_ASR_Language_Model_for_Lip_Reading_ICCV_2025_paper.pdf); [Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B) | VALLR already compares GPT-2 and Llama. Qwen is post-trained, but no checked head-to-head VALLR evaluation establishes a Qwen advantage. |
| Ranked CTC hypotheses and noise-aware LLM retraining | [reggosong/lip-reader](https://github.com/reggosong/lip-reader) | Direct VALLR extension with prefix beam, confidence-formatted hypotheses, synthetic viseme errors, and multi-format LoRA. Reports decoder eval loss, not its own LRS2/LRS3 end-to-end WER; reuse requires code/license review and independent tests. |
| Visual alphabet instead of coarse visemes | [Phonemes versus visemes](https://arxiv.org/abs/1805.02924); [intermediate visual units](https://arxiv.org/abs/1909.07147) | Existing VSR studies test these unit choices. Phonemes beat visemes at word accuracy in the former; intermediate units are dataset/speaker dependent. More visually homogeneous labels need not decode words better. |
| Dynamic/sequential visual units | [Sequential viseme-driven VSR, 2026](https://www.sciencedirect.com/science/article/pii/S0893608026000286); [associated repository](https://github.com/clayh24/lipreading-based-on-sequential-viseme) | Strong overlap: sequential viseme annotation and a dual-stream viseme/character recognizer evaluated on VSR corpora. Publisher preview and linked repository checked; full annotation reproducibility remains to audit. Do not claim VPA is the first dynamic visual alphabet. |
| Geometry, motion, and speaker robustness | [ADFAC](https://www.sciencedirect.com/science/article/pii/S2215016120302260); [cross-speaker landmark learning](https://aclanthology.org/2024.lrec-main.876/) | Articulatory distance/velocity/acceleration measurements and landmark-based VSR already exist. ADFAC is an analysis method, not evidence that our scalar feature list improves large-vocabulary VSR. |
| Learned hidden units and audio-guided representations | [AV-HuBERT](https://arxiv.org/abs/2201.02184); [RAVEn](https://arxiv.org/abs/2212.06246) | Established audiovisual self-supervision and cross-modal prediction. AV-HuBERT's clustering is audio-initialized, not a purely visual alphabet; neither supplies our interpretable gesture inventory. |
| Training-only audio teacher | [ASR is all you need](https://arxiv.org/abs/1911.12747) | Cross-modal ASR-to-lip-reading distillation is established. Our particular phone-posterior supervision still needs inventory mapping, synchronization, and an ablation against latent-feature teaching. |
| Transcript-derived pronunciations and audio alignment | [CMUdict](https://github.com/cmusphinx/cmudict); [MFA](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/implementations/lexicon_probabilities.html) | Lexica, alternative pronunciations, and alignment are existing tools, not a new alphabet. Neither dictionary conversion nor forced alignment independently verifies the spoken transcript or visible articulation. |
| CTC search, dictionary, HMM/n-gram/WFST alternative | [CTC](https://www.cs.toronto.edu/~graves/icml_2006.pdf); [DNN-HMM/WFST lip reading](https://arxiv.org/abs/1805.02924) | Probabilistic sequence search and lexical decoding already apply to VSR. A plain Markov chain is not a substitute for a visual encoder; compare language priors with the same emissions and search budget. |
| Lattice-aware and confidence-aware neural decoding | [Lattice attention](https://arxiv.org/abs/2111.10157); [ProGRes](https://arxiv.org/abs/2409.00217) | ASR precedents for weighted lattice representations and confidence-aware LLM rescoring/generation. Learned numerical embeddings are one experiment, not a prerequisite for using uncertainty. |
| Conservative correction | [Preventing ASR overcorrection](https://aclanthology.org/2024.emnlp-industry.20/) | Published ASR work explicitly addresses harmful LLM edits. Candidate constraints limit output space but cannot guarantee fidelity; measure miscorrections and abstention. |
| More audiovisual data with automatic labels | [Auto-AVSR](https://arxiv.org/abs/2303.14307); [AVSpeech](https://looking-to-listen.github.io/avspeech/download.html) | Scaled pseudo-labeled VSR is established. Downloadable manifests are not downloaded media or gold transcripts; teacher quality and domain remain confounders. |
| Probability calibration | [Guo et al.](https://arxiv.org/abs/1706.04599) | A classification-calibration baseline, not proof that CTC beam scores are calibrated sentence probabilities. Define and evaluate the actual prediction event. |
| Smaller specialized language decoder | [Discriminative rescoring distillation](https://arxiv.org/abs/2306.09452) | ASR work distills task-trained rerankers using recognition-oriented objectives. Test ranking/sequence distillation against generic text imitation; no VALLRITE gain is established. |
| Mobile visual recognition | [MobiVSR](https://arxiv.org/abs/1905.03968) | Mobile-oriented word-level VSR exists. Continuous sentences, full preprocessing, and this checkpoint's runtime need separate device measurements. |

### Research decisions after this audit

- **Keep:** uncertainty-preserving decoding, realistic-error training, audio-guided
  supervision, controlled data scaling, and eventual compression as experiments.
- **Change:** establish correct temporal emissions first; begin with n-best and
  simple lexical scoring; make VPA features optional; benchmark Qwen rather than
  assuming it wins.
- **Reject as established facts:** universally indistinguishable within-viseme
  phones, inherently superior IPA spelling, optimal handcrafted measurements,
  mandatory confidence embeddings, automatic WFST efficiency, and guaranteed
  phone deployment.
- **Possible contribution:** a reproducible, quality-aware visual-articulation
  representation connected to uncertainty-preserving decoding and tested on
  held-out speakers. Its value and originality must be demonstrated against
  these comparables, not inferred from the name “VPA.”
