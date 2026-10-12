# Competitive Word Decoder

This is the dedicated workspace for the **phoneme-to-word model**: training,
teacher experiments, distillation, and word-error evaluation.

## Entry points

### Original Llama clean-phone baseline

See [VALLR_CLEAN.md](VALLR_CLEAN.md) for the upstream Llama-3.2-3B/WikiText-2
reproduction, clean held-out evaluation, checkpoints, pause/resume, and known
upstream quirks. This experiment deliberately excludes noisy visual inputs.

### Local DeepSeek worker

Run a bounded task through the repository wrapper:

```bash
bash tools/local-deepseek CompetitiveWordDecoder/deepseek-training-review.txt
```

The default mode is `read-only`; an explicit second argument `workspace-write`
allows edits. The wrapper uses the installed local employee client and LM Studio
server; it does not download a model or fall back to a cloud model. Reports are
stored under `/home/elinzi/.lmstudio/employee/jobs/`.

The 2026-10-09 review completed successfully. Its resume-setting and hypothesis
validation findings were independently checked and implemented. Its proposal
for built-in validation was deferred because generation WER is already measured
with the separate evaluator. The proposed resume patch required correction:
hyperparameters live under the manifest's `arguments` field.

### Training continuation

The 2026-10-09 continuation uses the prior noisy-direct 2B adapter, learning rate
1e-5, 1,000 additional steps, and a 10-minute limit. Outputs are stored in
`checkpoints/competitive-word-decoder-lrs3-noisy-direct-20261009/`. This is a new
adapter continuation with a fresh optimizer, not an optimizer-state resume.
Teacher training now accepts `--max-minutes`; reaching that limit saves a
resumable checkpoint just like `PAUSE`. The stronger clean-phone baseline is
retained until the continuation's validation WER is measured.

| Task | Module |
| --- | --- |
| Train on noisy visual phoneme hypotheses | `CompetitiveWordDecoder.train_noisy_teacher` |
| Evaluate the combined N-best teacher | `CompetitiveWordDecoder.evaluate_noisy_teacher` |
| Distill candidate rankings into a smaller model | `CompetitiveWordDecoder.train_distilled_reranker` |
| Evaluate the VALLR prompt baseline | `CompetitiveWordDecoder.evaluate_vallr_qwen` |
| Evaluate individual-hypothesis generation | `PhonemeDecoder.evaluate_qwen_direct` |

`PhonemeDecoder/` contains the existing lexical decoder and shared Qwen
utilities. `VisualPhoneme/` produces the visual phoneme evidence. New word-model
experiments belong here; datasets stay under `datasets/decoder-training/` and
weights/results under `checkpoints/competitive-word-decoder-*/`.

The current LRS3 noisy training export is
`datasets/decoder-training/lrs3-full-forced-direct-train-5k/phone-hypotheses.jsonl`.
Its matched validation export is
`datasets/decoder-training/lrs3-full-forced-direct-validation/phone-hypotheses.jsonl`.

## LRS3 noisy teacher pilot results (2026-10-02)

Exported 5,000 unique training clips using the frozen gated-fusion epoch-3
checkpoint, beam 64, five hypotheses, and phone bigram weight 0.5. Validation
overlap was zero. Training oracle PER@5 was 77.98%. These are predictions from
a visual model that previously trained on this corpus; they are not out-of-fold
training examples. Evaluation uses the existing 1,140 held-out validation clips.

| Decoder | Top-1 WER | Oracle WER@5 |
| --- | ---: | ---: |
| Previous clean-phone 2B adapter, individual hypotheses | 104.58% | 96.94% |
| Noisy individual-hypothesis 2B continuation, 1,000 steps | 120.19% | 105.96% |
| Combined five-hypothesis teacher, best tested step 800 | 114.88% | Not measured |

The individual continuation covered 4,000 of the 25,000 expanded training
pairs (0.16 epoch). It regressed on held-out validation, so its predictions
should not be used to distill a student. The combined prompt also failed to
beat the prior individual decoder. The previous adapter remains the baseline.
The new training export and checkpoints are retained for further experiments.

Teacher training supports `--pause-file PAUSE`: a request saves a resumable
checkpoint and stops after the current step. Remove the file and use `--resume`
to continue. Combined training prompts had a maximum actual length of 427
tokens, within the configured 512-token limit.

This folder isolates phoneme/viseme-to-word decoding experiments from both the
original `VALLR` subtree and the visual encoder. Its goal is a fair decoder
leaderboard on fixed visual hypotheses, with top-1 WER, oracle WER@N, latency,
and model size reported together.

## What the original VALLR repository provides

The unmodified upstream `Models/Llama.py` at commit
`db932ca83ff8b2b858eb6a4c78d02b5e3805c299` is a **training recipe**, not a runnable
released word decoder. It creates clean ARPAbet-to-text examples from WikiText-2
and fine-tunes **Llama-3.2-3B-Instruct** using LoRA. Our local
`VALLR/Models/Llama.py` was modified to use Qwen3.5-2B; that is not the original
VALLR model. The recipe does not include the trained LoRA adapter, load VALLR
visual predictions, perform beam search, or report WER. Consequently, running
the local Qwen base without an adapter is only a zero-shot diagnostic; it is not
an evaluation of a released VALLR word decoder.

The Qwen evaluator here reproduces the local script's tag-based input format and
records this provenance in every result:

```bash
tools/with-vpa-gpu .venv-qwen/bin/python \
  -m CompetitiveWordDecoder.evaluate_vallr_qwen \
  --hypotheses datasets/decoder-training/lrs3-direct-smoke/phone-hypotheses.jsonl \
  --model checkpoints/Qwen3.5-2B \
  --output evaluation/competitive-word-decoder/vallr-base-smoke.json \
  --limit 20 --batch-size 16 --require-cuda
```

Use `.venv-qwen` for Qwen3.5: it contains Transformers 5.16+, which recognizes
the `qwen3_5` architecture. The general `.venv-vpa-gpu` environment currently
pins Transformers 4.57 for the visual/audio pipeline and cannot load this model.

If a compatible VALLR-trained adapter becomes available, add
`--adapter /path/to/adapter`. Do not use the existing direct-LRS3 adapter with
this prompt: it was trained with a different instruction template.

### Zero-shot diagnostic result

On 2026-10-01, the unadapted 2B model was tested on 20 LRS3 records (100 visual
phone hypotheses). It produced closing markup repeatedly rather than useful
sentences: top-1 WER was 136.74%, oracle WER@5 was 131.06%, and exact sentence
accuracy was 0%. Generation took 4.49 seconds on the RTX 5090. This confirms
that the missing task adapter is essential; it does not measure a trained VALLR
word decoder. The machine-readable report is
`evaluation/competitive-word-decoder/vallr-unadapted-base-smoke20.json`.

## Existing local reference points

- The 0.8B GRID candidate reranker reached 45.83% WER, versus 48.48% for its
  fixed candidate baseline (2,000 validation records). This is the strongest
  currently measured decoder result, but it is GRID-domain and candidate-bound.
- The direct 2B LRS3 adapter reached 104.58% top-1 WER and 96.94% oracle WER@5
  on forced-aligned visual hypotheses (1,140 records). It is not competitive.

Those numbers are not directly comparable because their datasets and decoding
interfaces differ. Future experiments in this folder should share one frozen
LRS3 manifest and one frozen set of visual N-best hypotheses.

## Planned competitive track

1. Freeze train/development/test manifests and visual N-best inputs.
2. Establish lexicon plus n-gram and neural-LM baselines.
3. Train a small error-aware reranker on noisy visual hypotheses, including
   silence/rest timing, family ambiguity, sequence score, and duration.
4. Distill the best 2B teacher into a mobile-sized reranker.
5. Report top-1 WER, oracle WER@5, real-time factor, peak memory, and parameter
   count for every checkpoint.

## One-round distillation

`train_distilled_reranker.py` transfers the adapted 2B teacher's complete soft
distribution over each bounded candidate set into the 0.8B adapter. Its loss is
temperature-scaled KL plus reference-candidate ranking and reference token NLL.
It never trains on validation records. Training is resumable and checks the
repository-root `PAUSE` file between optimizer steps.

```bash
tools/with-vpa-gpu .venv-qwen/bin/python \
  -m CompetitiveWordDecoder.train_distilled_reranker \
  --examples datasets/decoder-training/grid-train-gated-seed-20260912-b64/examples.jsonl \
  --teacher-scores checkpoints/competitive-word-decoder-distill-grid-20261001/teacher-train.scores.jsonl \
  --model checkpoints/Qwen3.5-0.8B \
  --initial-adapter checkpoints/qwen-decoder-0.8b-grid-20260913/adapter \
  --output checkpoints/competitive-word-decoder-distill-grid-20261001/student \
  --max-steps 500 --save-steps 100 --require-cuda
```

Create `PAUSE` to release the GPU after the current optimizer step; remove it to
continue. Repeat the command with `--resume` after an interruption. The first
pilot deliberately uses GRID because its training hypotheses and held-out
validation hypotheses already existed when this pilot ran. LRS3 now also has a
separate 5,000-clip training export; validation hypotheses remain evaluation-only.

### Pilot result (2026-10-01)

The adapted 2B teacher scored 38,621 candidate sequences from 7,895 GRID
training clips. Its in-sample WER was 8.87% at reranker weight 50, versus 13.22%
for the fixed lexical order. A 500-step distillation pilot then continued the
existing 0.8B adapter using 5,685 shuffled records, of which 4,117 contained the
reference in the retained candidate set.

On the untouched 2,000-clip GRID validation split, the distilled student's best
dense-sweep WER was **46.01%** (5,521 errors at weight 100). The undistilled
0.8B adapter remains better at **45.83%** (5,500 errors at weight 50). At weight
1 the new model was effectively tied: 46.30% versus 46.31%. This pilot therefore
does not justify longer training with the same teacher and loss. The adapted 2B
teacher itself is only 0.03 percentage points better than the old 0.8B model on
validation, leaving almost no useful teacher headroom. The next meaningful
distillation run needs LRS3 **training** hypotheses and a teacher that materially
beats the student on a frozen held-out set.
