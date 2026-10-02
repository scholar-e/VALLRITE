# Competitive Word Decoder

This folder isolates phoneme/viseme-to-word decoding experiments from both the
original `VALLR` subtree and the visual encoder. Its goal is a fair decoder
leaderboard on fixed visual hypotheses, with top-1 WER, oracle WER@N, latency,
and model size reported together.

## What the original VALLR repository provides

`VALLR/Models/Llama.py` is a **training recipe**, not a runnable released word
decoder. It creates clean ARPAbet-to-text examples from WikiText-2 and fine-tunes
Qwen3.5-2B using LoRA. It does not include the trained LoRA adapter, load VALLR
visual predictions, perform beam search, or report WER. Consequently, running
the local Qwen base without an adapter is only a zero-shot diagnostic; it is not
an evaluation of a released VALLR word decoder.

The evaluator here exactly reproduces that script's tag-based input format and
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
validation hypotheses already exist. LRS3 currently has validation hypotheses
only, and those must not be reused for decoder training.

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
