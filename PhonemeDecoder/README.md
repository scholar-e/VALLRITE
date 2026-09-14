# Phoneme probability → words decoder

A framework for decoding **complete CTC phoneme distributions into ranked word
sequences**, using the visual models, CMUdict work, and optional Qwen adapter
already in VALLRITE.

**Status: initial runnable Python reference plus design contracts.** The runtime
accepts the current VisualPhoneme models' `[T,B,40]` logits through an adapter,
or exported full probability rows. It incrementally constrains CTC prefixes with
a pronunciation trie and segments complete hypotheses into ranked words.

## Run it

From the repository root, no third-party decoder dependencies are needed:

```bash
python -m PhonemeDecoder PhonemeDecoder/examples/request.json \
  --lexicon PhonemeDecoder/examples/lexicon.json \
  --output /tmp/words.json
```

Use the existing visual predictor's Python environment for real video:

```bash
python -m VisualPhoneme.predict clip.mp4 --checkpoint path/to/best.pt \
  --emissions-output /tmp/emissions.json --word-lexicon path/to/lexicon.json
```

Coordinate/fusion models still need their existing `--landmarks-npz` argument.
`--word-lexicon` adds `word_decoding` to the existing predictor result;
`--emissions-output` is independently optional. Use a real lexicon matching your
application: the supplied example only knows pat, bat, and mat.

Generate realistic candidate-ranking examples from a frozen visual checkpoint
and measure the lexical decoder on GRID:

```bash
.venv-vpa-gpu/bin/python -m PhonemeDecoder.evaluate_grid \
  --checkpoint path/to/best.pt --split train \
  --output-dir datasets/decoder-training/grid-train
```

This writes `examples.jsonl`, the generated runtime `lexicon.json`, a summary,
and a dual-output log. Each example contains the reference words and phones,
greedy visual phones, ranked lexical candidates, and Top-1/oracle errors. A run
on `train` is explicitly marked as an in-sample diagnostic because the visual
producer has already seen those speakers. Use out-of-fold visual checkpoints
before treating these records as unbiased decoder-training data. The command
does not expose locked test speakers.

```python
from PhonemeDecoder import Lexicon, StreamingDecoder, probabilities_from_logits

decoder = StreamingDecoder(Lexicon(lexicon_document), beam_width=16)
rows = probabilities_from_logits(logits, checkpoint["phones"], valid_steps)
decoder.accept(rows)  # Or accept successive, non-overlapping chunks.
result = decoder.result(nbest=5)
decoder.reset()  # New utterance; state must not cross utterance boundaries.
```

The runtime is deliberately narrower than the proposed schemas: it emits
`phoneme-decoder-runtime-0.1`, with ranked word IDs, text, and unnormalized log
scores. It does not claim conformance to the full design response envelope.
It uses pronunciation priors but has no trained word LM or Qwen dependency.
Input rows must be full normalized probabilities, not the predictor's top-N
phone candidates. The direct adapter checks the checkpoint inventory and valid
length. The CLI also accepts the design's simple probability fixture, but does
not implement resource hashes, calibration, or all design request semantics.

Phone and lexical beams, maximum steps, phones, and words bound work per
utterance. Phone/step overflow raises an error; lexical/word caps prune paths.
A raised update may have consumed earlier rows in that chunk; reset before retry.
Results are provisional complete dictionary paths and can change with more input.
No endpoint detector or committed-word interface is implemented. The current
visual models still run whole clips; streaming the decoder does not make those
noncausal models streaming. Python speed on a phone, energy use, and vocabulary
scaling remain unmeasured. Native implementation and device profiling are next.

## Recommended starting point

### Experimental Qwen training

#### Direct phonemes to words

`train_qwen_direct.py` trains a separate LoRA whose input is one ARPAbet phone
sequence and whose target is the transcript. It does not construct, expose, or
score a lexicon candidate list. `evaluate_qwen_direct.py` applies that model to
each sequence in a visual CTC N-best list and reports both top-1 WER and oracle
WER@N over the directly generated sentences.

```bash
tools/with-vpa-gpu .venv-qwen/bin/python -m PhonemeDecoder.train_qwen_direct \
  --manifest datasets/lrs3/manifests/train.jsonl \
  --model checkpoints/Qwen3.5-2B \
  --output checkpoints/qwen-direct-lrs3-20260913 \
  --max-steps 1000 --max-entries 7895 --batch-size 4 \
  --save-steps 100 --require-cuda

tools/with-vpa-gpu .venv-qwen/bin/python -m PhonemeDecoder.evaluate_qwen_direct \
  --hypotheses datasets/decoder-training/lrs3-direct-validation/phone-hypotheses.jsonl \
  --model checkpoints/Qwen3.5-2B \
  --adapter checkpoints/qwen-direct-lrs3-20260913/adapter \
  --output checkpoints/qwen-direct-lrs3-20260913/eval-validation.json \
  --batch-size 32 --require-cuda
```

The first held-out LRS3 development result covers 1,140 clips and 12,675
reference words. The frozen hybrid visual checkpoint's phone beam has 78.83%
oracle PER@5. Direct Qwen decoding gives 116.50% top-1 WER and 102.87% oracle
WER@5; WER can exceed 100% when insertions are numerous. The 2B base plus this
22 MB adapter therefore does not yet replace the lexical decoder. Its training
input used clean transcript-derived phones, while evaluation used very noisy
visual phones. The next direct experiment should train on out-of-fold visual
phone hypotheses paired with transcripts (and optionally calibrated synthetic
phone corruption), without reintroducing word candidates. Do not train the
decoder on predictions from a visual checkpoint that already fitted the same
clips and describe the result as held-out.

After scaling forced alignment to 28,739 LRS3 training videos and selecting the
full-data hybrid's epoch 3, a beam-64/phone-bigram-0.5 export reached 78.14%
oracle PER@5. Holding the direct Qwen adapter fixed and changing only those
visual hypotheses improved top-1 WER from 116.50% to **104.58%** and oracle
WER@5 from 102.87% to **96.94%** on the same 1,140 clips. This demonstrates
that improving the phone producer helps the word stage, although the remaining
error still requires noisy-phone decoder training and a stronger visual model.

`train_qwen.py` provides a supervised LoRA warm-up using the local Qwen3.5-2B
checkpoint and exported GRID candidates. It conditions on observed greedy phones
and candidate text, masks prompt tokens from the loss, and learns the reference
candidate plus one EOS. Reference phones and oracle annotations never enter the
prompt. Records with references absent from the beam are skipped; only training
speakers 1–8 are accepted. The manifest records the input hash and selected clips.

```bash
.venv-qwen/bin/python -m PhonemeDecoder.train_qwen \
  --examples datasets/decoder-training/grid-train-gated-seed-20260912-b64/examples.jsonl \
  --output checkpoints/qwen-decoder-warmup-new \
  --max-steps 10 --max-entries 128
```

Use a new output directory per run. `--prepare-only` validates and writes the
manifest without loading Qwen. Progress goes to `train.log`; resumable Trainer
checkpoints are saved every five steps and the final adapter to `adapter/`.
To resume, repeat the original command with `--resume`. Add `--require-cuda`
for GPU runs to reject accidental CPU fallback. GPU access may require running
outside the isolated sandbox, with CUDA-enabled PyTorch in `.venv-qwen`.
This initial dataset is an **in-sample diagnostic**, with 5,685 eligible records
and 2,210 absent references. A small warm-up is not evidence of WER improvement.
Candidate-only inference, visual-score filtering, and held-out validation are
still required before enabling this adapter in the runtime. Future scoring must
reuse the prompt and mean target-token log likelihood including EOS; generated
text must not introduce candidates. The existing runtime is not changed by training.

The final study retains three comparable paths: lexical decoding alone,
Qwen3.5-2B as a research proof of concept, and a two-layer 256-wide word-level
GRU intended for int8 phone deployment. Train both rerankers before comparing
WER on identical frozen emissions. The phone evaluation must also report model
size, peak memory, latency, and energy; WER alone does not select the mobile model.

### Reusable candidate-model WER evaluation

### Connect video predictions to a trained adapter

Run `VisualPhoneme.predict` with `--word-lexicon` and `--output video.json` first.
Fusion readers also require `--landmarks-npz`. Then use the Qwen environment:

```bash
tools/with-vpa-gpu .venv-qwen/bin/python -m PhonemeDecoder.rerank video.json \
  --model checkpoints/Qwen3.5-0.8B \
  --adapter checkpoints/qwen-decoder-0.8b-grid-20260913/adapter \
  --output video-words.json --weight 1 --device cuda
```

The final `adapter/` exists after training completes; use a preserved intermediate
checkpoint for preliminary testing. This two-command bridge lets visual inference
and Qwen use their existing separate Python environments. The output preserves
`word_decoding` and adds `word_decoding_reranked`, with raw model scores, combined
scores, original ranks, and fallback diagnostics. No generated text becomes a new
candidate. Model-load or scoring failure retains the lexical ranking.

The default weight 1 is an experimental setting, not a mobile deployment choice.
The filter currently uses the runtime's aggregated lexical `log_score`, not pure
per-path CTC likelihood; it is explicitly recorded as a **base-score window**.
The earlier evaluation option named `--visual-score-window` uses this same
aggregate score and must not be interpreted as the full design's visual window.

Completed Qwen3.5-0.8B validation result (2,000 GRID clips, speakers 9–10,
12,000 reference words): baseline 48.48% WER; weight 1 gives 46.31%, weight 3
gives 45.90%, and weight 50 gives 45.83% (5,500 errors). The completed 2B adapter
at weight 50 gives 45.80% (5,496 errors). Weight 50 was previously selected on
this validation set for 2B; this is development comparison, not untouched test
performance. Full results and reusable scores are in
`checkpoints/qwen-decoder-0.8b-grid-20260913/eval-validation-final.json` and
`eval-validation-final.scores.jsonl` in the same directory.

The bridge was checked on real clip `s9:bgbt4n` with the gated-fusion reader,
its landmark cache, and the completed 0.8B adapter. The output
`checkpoints/qwen-decoder-0.8b-grid-20260913/video-smoke-reranked.json`
records successful candidate scoring. This verifies the connection, not clip
accuracy or phone deployment performance.

The additional mid-sized experiment uses `Qwen/Qwen3.5-0.8B`, downloaded to
`checkpoints/Qwen3.5-0.8B`. It uses the same training examples, seed, rank-8 LoRA,
and 5,685-step budget as the 2B run. Its separate output directory is
`checkpoints/qwen-decoder-0.8b-grid-20260913`. This adds a model-size comparison;
phone performance remains unmeasured and the compact GRU remains proposed.

```bash
tools/with-vpa-gpu .venv-qwen/bin/python -m PhonemeDecoder.train_qwen \
  --model checkpoints/Qwen3.5-0.8B \
  --examples datasets/decoder-training/grid-train-gated-seed-20260912-b64/examples.jsonl \
  --output checkpoints/qwen-decoder-0.8b-grid-20260913 \
  --max-steps 5685 --max-entries 5685 --require-cuda
```

`evaluate_candidate_scores.py` separates model inference from WER calculation.
Any future phoneme-to-word model can emit one JSON object per line containing
`clip_id`, `candidate_index`, `candidate_text`, and a finite higher-is-better
`score`. The evaluator validates candidate identity, applies the configured
visual-score window, combines model and decoder scores over a weight sweep, and
reports baseline, reranked, and oracle N-best WER. Saved score files make new
weight sweeps CPU-only and deterministic.

`evaluate_qwen.py` is the Qwen adapter for this format. For example:

```bash
tools/with-vpa-gpu .venv-qwen/bin/python -m PhonemeDecoder.evaluate_qwen \
  --examples path/to/examples.jsonl --adapter path/to/adapter \
  --output path/to/evaluation.json --batch-size 8 --require-cuda

.venv-qwen/bin/python -m PhonemeDecoder.evaluate_candidate_scores \
  --examples path/to/examples.jsonl \
  --scores path/to/evaluation.scores.jsonl --output path/to/weight-sweep.json
```

Use the compact `VisualPhoneme` model as the first producer. Export its full
blank-plus-39-phone distribution at each valid output step. Apply CTC prefix
beam search, segment the retained phoneme sequences through a pronunciation
lexicon, and rank word candidates with an explicit language score. Retain
alternatives and component scores. Add Qwen only as a bounded candidate reranker
once the lexical baseline is measured.

```mermaid
flowchart LR
    V[Visual CTC logits] --> A[Export and validate full distributions]
    A --> B[CTC phoneme prefix beam]
    B --> L[Pronunciation lexicon and word segmentation]
    L --> N[Word candidates with component scores]
    N --> O[Ranked words or no lexical path]
    N -. optional .-> Q[Qwen candidate scoring]
    Q --> O
```

The first reference decoder deliberately uses two stages so CTC correctness and
lexical segmentation can be tested separately. This may prune a useful phone
sequence before the lexicon sees it. Measure that loss explicitly; move to a
joint lexicon/CTC search or WFST when the reference exposes a search bottleneck.
An N-best list is not a complete lattice.

## What the existing project contributes

| Existing work | Reuse | Required change or limitation |
|---|---|---|
| [`VisualPhoneme/model.py`](../VisualPhoneme/model.py) | Image-only and image/landmark fusion CTC models; one output step per frame | Export time-major logits `[T,B,40]` with valid lengths, model vocabulary, and source timing |
| [`VisualPhoneme/predict.py`](../VisualPhoneme/predict.py) | Checkpoint loading, crop selection, inference | Currently saves greedy phones and a blank summary, not the full distribution; fusion truncation must be recorded |
| [`vallrite.py`](../vallrite.py) | Older VALLR logits, CTC blank convention, greedy baseline, Qwen path | Logits are `[B,T,40]`; rounded top-5 output cannot reconstruct missing probability mass |
| [`train_phoneme_dictionary.py`](../tools/train_phoneme_dictionary.py) | CMUdict normalization and local LoRA training machinery | Single-word deterministic examples do not train a probability-aware sentence decoder; retain all pronunciations and homophones |
| [`VisualPhoneticAlphabet`](../VisualPhoneticAlphabet/README.md) | Observations, missing-data metadata, experimental evidence | Gesture compatibility and geometric proxies are not phoneme emission probabilities |
| [`VisualPhoneticAlphabet/evaluate.py`](../VisualPhoneticAlphabet/evaluate.py) | Token edit-distance scoring pattern | Add word-level scoring, oracle candidate recall, and search/resource diagnostics |

The old VALLR smoke test emitted only eight CTC steps for 16-phone references;
no decoder can recover a complete 16-phone CTC path from eight steps. The newer
compact recognizer avoids that structural limit, but its current results remain
validation results with substantial error. See the respective
[old-model diagnostic](../VisualPhoneticAlphabet/evaluation/README.md) and
[compact-model results](../VisualPhoneme/evaluation/README.md). This framework
makes no new accuracy claim.

## Artifacts in this folder

- [DESIGN.md](DESIGN.md): algorithms, scores, alternatives, failure behavior,
  integration, and evaluation gates.
- [schemas/request.schema.json](schemas/request.schema.json): full CTC
  probability input and bounded search configuration.
- [schemas/response.schema.json](schemas/response.schema.json): N-best word
  output, explicit scores, diagnostics, and failure states.
- [schemas/lexicon.schema.json](schemas/lexicon.schema.json): versioned
  pronunciation alternatives, including homophones.
- [examples](examples/README.md): synthetic ambiguous `P/B/M AE T` request and
  tiny lexicon. These are format fixtures, not measured model predictions.

JSON Schema checks structural constraints. The decoder must additionally check
normalization, vocabulary agreement, monotonic timestamps, hash integrity,
score identities, and status/candidate consistency as specified in DESIGN.md.

## Planned Python boundary

These original design signatures remain **proposed**; use the runtime API above:

```python
# Producer adapter: no dependency on a particular visual architecture.
export_ctc(logits, *, layout, valid_length, vocabulary, step_times_ms,
           model_fingerprint, preprocessing_fingerprint) -> EmissionRecord

# Offline resource preparation; preserve pronunciations and lexical identity.
build_lexicon(entries, *, phone_inventory, source_revision) -> Lexicon

# Reference decoder stages, run in log space.
ctc_prefix_beam(emissions, *, beam_width) -> list[PhoneHypothesis]
segment_words(phone_hypotheses, lexicon, lm, *, lexical_beam) -> list[WordCandidate]
aggregate_and_rank(candidates, config) -> DecodeResult

# Optional second pass: score only existing complete candidates.
rerank(candidates, scorer, *, visual_score_window) -> DecodeResult
```

## First implementation checklist

1. Implement request validation and full-posterior exporters for the two CTC
   tensor layouts. Preserve existing greedy outputs as comparison baselines.
2. Implement a dependency-light log-space CTC prefix beam and prove it against
   exhaustive path enumeration on tiny inputs.
3. Add a trie lexicon with exact multiword segmentation, pronunciation priors,
   explicit homophones, and a no-LM mode.
4. Add a smoothed word LM or optional GRID grammar. Name and evaluate each mode
   separately; do not present a GRID-constrained result as unrestricted speech.
5. Freeze the visual checkpoint and use validation speakers for decoder tuning.
   Publish WER, oracle N-best WER, OOV rate, search failures, and latency.
6. Consider a joint search implementation and then bounded Qwen reranking only
   when each improves the declared baseline.

Keep the existing CLIs intact during development. A future `--decoder lexical`
flag should be opt-in; rollback is selecting the existing greedy path. No model
or data download is triggered by this folder.

## Check the contracts

With `jsonschema` installed (see [requirements-dev.txt](requirements-dev.txt)),
run from the repository root:

```bash
python -m unittest discover -s PhonemeDecoder/tests -v
```

The eight checks cover schema validity, full versus sparse/non-CTC inputs,
failure-state structure, configuration consistency, pre-search errors, fixture hashes and normalization, CTC repeats, and the
worked example's exhaustive scores. Runtime tests additionally exercise chunk
equivalence, lexical constraints, homophones, repeats, limits, and the torch
adapter (skipped if torch is unavailable). This is not an accuracy benchmark.
