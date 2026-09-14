# Probability-preserving phoneme-to-word decoding

Status: proposed full framework; initial narrower runtime is documented in README.md. Date: 2026-09-11. Audience: implementers of the
VALLRITE visual recognizer, lexical decoder, and evaluation pipeline.

## Decision and scope

Start with a transparent, offline **CTC prefix beam → pronunciation lexicon →
word scorer** reference decoder. Its input is the full time-by-class CTC
probability matrix; its output is an N-best list of complete word sequences,
including the empty sequence when supported. Use the current compact visual
recognizer as the primary producer. Keep neural reranking optional.

The purpose is to resolve ambiguous visible phonemes using lexical evidence and
bounded language context while preserving alternatives. It is not to synthesize
a plausible sentence despite absent visual evidence. This is a design package;
the initial runtime implements incremental trie-constrained phone search and bounded
lexical segmentation, but not the full scoring and resource contracts below.

Goals:

- Correct CTC blank/repeat handling, with probability mass summed across paths.
- Word-boundary discovery without assuming a word-separator phoneme.
- Homophones and pronunciation alternatives retained until ranked in context.
- Reproducible, decomposable scores and explicit search failures.
- The same decoder contract for old and compact visual models.

Non-goals for the first implementation: streaming, unconstrained LLM generation,
new visual-model training, invented phone timing, phone insertions/deletions
through an error channel, and production mobile latency guarantees.

## Alternatives

| Approach | Advantages | Costs and failure modes | Decision |
|---|---|---|---|
| Greedy phones → Qwen, as today | Existing code; broad textual vocabulary | Loses path mass; unconstrained corrections; single-word adapter is not calibrated for lattices | Keep as a baseline |
| Phone prefix beam → lexicon → word LM | Small inspectable stages; no neural decoder required; straightforward correctness oracles | Phone-only pruning can remove the best lexical answer; large lexicons need bounded segmentation | Implement first |
| Joint lexicon/CTC beam or WFST | Applies lexical constraints during search; efficient mature implementations exist | More state/semiring complexity, build dependencies, and tuning | Next if oracle experiments show material pruning loss |
| Learned sequence-to-sequence decoder | Can learn pronunciation variation and sentence context | Needs paired noisy emissions and text; weaker provenance; hallucination risk and training cost | Later controlled comparison |

[Flashlight Text](https://github.com/flashlight/text) is a concrete candidate for
the joint-search stage. Benchmark the available implementation and its current
CTC/lexicon interfaces before adopting it; no dependency is pinned by this design.

## Input contract and model adapters

See [request.schema.json](schemas/request.schema.json). The sole initial input
kind is `ctc_probabilities`, with exactly 40 columns in this canonical order:

```text
<blank> AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M
N NG OW OY P R S SH T TH UH UW V W Y Z ZH
```

Blank is class zero. `<pad>` in the old CLI maps to `<blank>` at this interface;
it does not become silence. Stress digits belong to dictionary provenance, not
to the canonical inventory. Reorder by explicit label mapping, never by assuming
matching array positions. Reject duplicates, extra/missing labels, mismatched
checkpoint inventories, or non-CTC frame-classifier scores. In particular, the
landmark discrimination probes do not become CTC producers merely by inserting
a blank column. Geometry confidence is not a CTC probability.

Producer adapters:

| Producer | Raw layout | Export operation |
|---|---|---|
| Compact visual model | `[T,B,C]` | Select batch item, truncate to valid output length, softmax over `C` |
| Original VALLR | `[B,T,C]` | Select batch item, truncate to valid output length, softmax over `C` |

Export from logits, not from greedy strings or rounded top-5 lists. Store
`softmax(logits / temperature)` with the temperature and its validation-artifact
fingerprint. Default temperature is 1, with status `raw_softmax`; it is not a
claim of calibration. A fitted temperature is still not proof of calibrated word
confidence. Fitting must use CTC sequence likelihood or another declared valid
CTC objective, not pretend that phone-interval labels are CTC spike alignments.

Semantic validation beyond JSON Schema:

- `producer.valid_steps == T`, where `T` is the number of rows. JSON Schema
  cannot enforce this equality, timestamp monotonicity, or parallel-array lengths;
  semantic validation must reject each mismatch before search.
- Each row sums to 1 within absolute tolerance `1e-5`; values are finite and in
  `[0,1]`. Reject malformed rows instead of silently normalizing corrupted data.
- Zero is allowed and becomes negative infinity internally. Use log-sum-exp;
  never multiply long products in floating point. JSON output never contains
  NaN or Infinity. Omit impossible candidates.
- Empty input (`T=0`) is legal and returns `no_evidence`, no candidates.
- `step_times_ms` is null or has exactly `T` strictly increasing nonnegative
  entries. Use source PTS and a declared adapter mapping. A null array is required
  when timing is unknown; search still works without timestamps.
- Do not invent VALLR time positions from its eight flattened/downsampled steps.
  The compact model preserves step count, but its temporal receptive field means
  the step time is an anchor, not a proven phoneme onset.
- `observed_steps` is null or `T` booleans and is diagnostic. Do not drop missing
  frames or replace them with blanks. The producer's trained missing-data policy
  determines emissions. A known all-missing clip returns `no_evidence`.
- The request's lexicon ID/hash and the supplied lexicon must agree. IDs and
  hashes select already-loaded resources; they are not URLs to fetch or paths
  to execute.

Record valid length after any crop/cache truncation and the preprocessing hash.
The existing fusion predictor truncates video to coordinate-cache length; its
future exporter must report that fact rather than imply full-clip coverage.

The compact producer uses CTC training with implicit blank class zero.
`raw_softmax` requires temperature 1 and no calibration artifact;
`temperature_scaled` requires a fitted-artifact fingerprint. A null LM or reranker
requires its corresponding weight to be zero; contradictions are `invalid_input`.

## Lexicon representation

See [lexicon.schema.json](schemas/lexicon.schema.json). Build a trie over canonical
phone IDs. Every terminal stores one or more word/pronunciation IDs. The same
phone sequence may produce several word IDs; never overwrite homophones.

Reuse the normalization idea from
[`train_phoneme_dictionary.py`](../tools/train_phoneme_dictionary.py), but avoid
its shuffled single-target representation. Import **all** CMUdict alternatives,
strip stress for compatibility, and deduplicate identical `(word, phones)`
entries after normalization. Preserve original pronunciations and source version
in resource provenance. Default pronunciation prior is uniform over the unique
pronunciations of each word, not over every dictionary row. Check each word's
prior sum against 1. CMUdict membership is not a word-frequency estimate.

Require nonempty words and phone sequences, unique word IDs, unique pronunciation
IDs within a word, and finite positive priors. Empty pronunciations are forbidden
because they create zero-length word loops. Cap pronunciation length and lexicon
size in the resource loader. Retain upstream license/attribution with any
redistributed lexicon; see [CMUdict source and license](https://github.com/cmusphinx/cmudict).

GRID's letter names need explicit entries, especially `a → EY`; unrestricted
English must also allow article pronunciations for `a`. These are separate
lexicon configurations with distinct IDs and content hashes. Do not inject a held-out clip's transcript into its
lexicon. An optional GRID grammar must be published as a constrained condition.

## CTC search semantics

For emissions `p_t(c)` and collapsed phone sequence `q`:

```text
L_ctc(q) = log sum over paths pi with collapse(pi)=q of product_t p_t(pi_t)
```

CTC collapse first merges adjacent repeats, then removes blank. Consequently:
`P P → P`, `P blank P → P P`, and `blank blank → []`. A word boundary does not
reset the CTC repeat state. Repeated phones across adjacent words still require
a separating blank in a path. See the original
[CTC paper](https://www.cs.toronto.edu/~graves/icml_2006.pdf).

Maintain separate blank-ending and nonblank-ending log masses for each phone
prefix. Initialize the empty prefix's blank mass to 0 and all other masses to
negative infinity. For each step, accumulate all contributions into the next
prefix table **before pruning**:

1. Blank adds the previous prefix's total mass to its next blank-ending mass.
2. A nonblank unlike the last phone extends from the previous total mass.
3. A nonblank equal to the last phone stays at the same prefix from its previous
   nonblank mass; it extends the repeated phone only from its previous blank mass.
4. Merge identical prefixes by log-sum-exp. Sort deterministically by total log
   mass, with lexicographic phone IDs as a tie-breaker; retain the configured beam.

Start with all 40 classes per step: token top-k pruning is not needed at this
inventory size. Prefix-beam pruning still makes retained sequence masses lower
bounds on exact CTC masses because contributing paths may have been pruned.
Name the diagnostic `retained_prefix_log_mass`, not `posterior_coverage`; do not
claim to know lost mass without an appropriate exhaustive/reference computation.
Retain the empty prefix as an explicit diagnostic, even if it falls outside the
candidate beam. All-blank deterministic input should produce an empty transcript
candidate, not a word. Empty transcript does not prove acoustic silence.

Do not assign phone or word times from the greedy path. Later, constrained CTC
alignment may supply candidate-specific timing estimates; first-release output
has null word times.

Define `retained_prefix_log_mass` as log-sum-exp of the final retained prefix
masses, including empty exactly once if retained in the beam. An empty prefix
kept only for diagnostics does not contribute. It is not lexicon coverage.
For each previous prefix q, let b(q), n(q) be its blank/nonblank log masses,
a(q)=logaddexp(b(q),n(q)), and ell(c)=log p_t(c). Accumulate into fresh tables:

```text
b_next(q) +=log a(q) + ell(blank)
c != last(q): n_next(q+c) +=log a(q) + ell(c)
c == last(q): n_next(q) +=log n(q) + ell(c)
              n_next(q+c) +=log b(q) + ell(c)
```

Here `+=log` means log-add-exp; every right-hand side uses the previous time
step, and `last(empty)` matches no phone.

## From phone hypotheses to word candidates

For each retained phone sequence, run dynamic programming over phone offsets and
lexicon trie transitions. At a terminal, emit a word and start a new trie walk at
the next phone offset. A terminal that also has children must allow both a word
boundary and continued traversal. Each emitted word consumes at least one phone.
Stop at sequence end; an unfinished trie prefix is not a complete candidate.

The lexical search state includes `(phone_offset, trie_node, LM_state, words/pronunciation
backpointer)`. Apply the lexical beam per offset after accumulating equivalent
states. Keep distinct word histories when their output or LM state differs.
An LM score is charged exactly once per emitted word and once for end-of-sentence.
Empty output pays its own BOS→EOS transition once. A no-LM implementation returns
zero for both transitions. No transition score is charged during blank/repeat
steps in the first-stage phone search.

Do not silently discard unmatched phones, invent an OOV pronunciation, or force
a partial lexical path to complete. When there are no nonempty lexical candidates, allow an empty-only result
only if the highest-scoring retained phone prefix is itself empty. Otherwise
return `no_lexical_path`, even if the all-blank path has nonzero mass. When
nonempty lexical candidates exist, the empty sequence may compete normally.
This prevents a tiny blank-path probability from disguising an OOV/search failure.
Retained phone hypotheses remain available in internal diagnostics. OOV spelling,
phoneme edit channels, and partial-word display need separate contracts and
validation. Homophones with identical priors and no LM remain tied alternatives.

## Scores and aggregation

Use natural logarithms. Fix the visual coefficient at 1 for identifiability:

```text
path_score(q, W, r) = L_ctc(q)
                   + pronunciation_weight * sum_j log P(r_j | w_j)
                   + lm_weight * [sum_j log P(w_j | history_j) + log P(EOS | W)]
                   + word_insertion_bonus * number_of_words(W)

base_score(W) = logsumexp over UNIQUE (q, r) yielding W of path_score(q, W, r)
final_score(W) = base_score(W) + reranker_weight * reranker_score(W)
```

The LM is a word LM; a spelling tokenizer's token score must not be substituted
without defining its EOS handling and length convention. `word_insertion_bonus`
is a signed score term, not a probability. Positive values favor more words;
negative values penalize them. Pronunciation and LM weights are nonnegative.
Zero weight contributes zero even when the corresponding component is negative
infinity; implement this explicitly rather than evaluating `0 * -inf`.

Deduplicate lexical paths before aggregating. The CTC prefix score already sums
its retained frame paths; attaching it once per unique lexical analysis avoids
counting every acoustic alignment a second time. Paths with the same word IDs
but genuinely different pronunciations can contribute separately. Alias display
spellings must not create duplicate hidden paths. This is a weighted decoding
objective, not a normalized generative word posterior.

Output includes `base_score`, `final_score`, raw reranker score, and a
`representative_path` with its component scores. That path is the best individual
path for explanation; its `weighted_total` is **not** generally `base_score`,
which can aggregate several paths. Non-representative analyses are aggregated
but not individually exported. Require pronunciation IDs in word order, exactly
one per word, and `word_count == len(words) == len(pronunciation_ids)`. Candidate
IDs must be unique within a response; stable word-ID sequences break score ties. Check `base_score >= representative total`
within numerical tolerance and the final-score identity. A returned N-best list
is sorted by final score, then stable word IDs. Keep homophone alternatives.

Do not expose `confidence: 0.9` by applying softmax to an arbitrary beam. The
first response contract deliberately has no word-confidence field. If candidate
renormalization is added later, label it explicitly as conditional on the
retained set and evaluate calibration separately.

## Optional Qwen reranking

The existing Qwen path generates unrestricted text from a greedy sequence and
prose top-k values. Its CMUdict LoRA adapter was trained on individual dictionary
words, not uncertain sentence lattices. Reuse checkpoint-loading machinery for
an experiment, not its output contract as a production decoder.

Qwen3.5-2B is the research proof-of-concept reranker, not the intended mobile
model. The mobile learned reranker is a two-layer word-level GRU with tied input
and output embeddings, trained on the same candidate-ranking records and
exported as int8. Its vocabulary comes from the immutable decoder lexicon and
must include explicit BOS, EOS, and unknown tokens. Start with hidden and
embedding width 256; record its serialized size, peak memory, latency, and energy
on the target phone. It uses the same candidate-only boundary, EOS convention,
visual-score window, and failure fallback as Qwen.

Keep and compare three final systems on identical frozen emissions: lexical
decoder without a learned reranker, Qwen3.5-2B, and the compact GRU. Tune weights
on selection speakers, then report WER and resource measurements together after
both learned models are trained. Do not use intermediate training checkpoints to
choose a system from test-set WER.

Proposed integration scores only enumerated complete word candidates. Use a
fixed prompt and teacher-forced candidate-token log likelihood, including one
EOS and a versioned token-length normalization convention. Treat that as a
reranker score, not an acoustic likelihood. Do not parse generated free text into
new candidates, and do not treat instruction-following as an enforcement boundary.

Compare each candidate's best retained `L_ctc` with the maximum among complete
lexical candidates. Build the reranking set from candidates within `visual_score_window` log units,
plus the base winner as an explicit exception so fallback is always possible.
Score and rank only this set when reranking succeeds; filtered candidates cannot
win through an unscored default of zero. Record the number filtered in diagnostics.
If no eligible alternative beats the base winner under the weighted objective,
return it. This window limits evidence
loss but cannot prove a correction is right. Tune it and reranker weight on
validation WER. On scorer failure, return the unchanged base ranking with a
warning. `reranker=none` requires weight 0 and never loads Qwen.

## Failure behavior, resources, and observability

- Invalid structure, nonfinite values, bad normalization, resource mismatch:
  `invalid_input` or `resource_mismatch`, with a machine-readable diagnostic.
- `T=0` or all known-unobserved steps: `no_evidence`, no word candidates.
- Valid observed all-blank input: `ok` with an empty-word candidate.
- No nonempty dictionary path and a nonempty best phone prefix:
  `no_lexical_path`; do not substitute the low-probability empty path.
- Hard memory/state/input budget exceeded: `resource_limit`, no candidates.
  Regular configured beam pruning is expected approximate search, not this error.
- Optional reranker failure: `ok`, original base ranking, explicit warning.

Suggested starting caps, not measured performance claims: 10,000 output steps,
512 phone prefixes, 128 lexical states per offset, 20 returned word candidates,
128 words per candidate, 64 phones per pronunciation. Tune smaller operational
limits after profiling. A candidate cap must never be mislabeled as exhaustive
search. Resource loading and hash verification happen once per immutable resource
bundle; per-request state is isolated and cannot leak context across users.

The probability matrix is about `T * 40 * 4` bytes as float32 (1.6 MB at 10,000
steps, excluding JSON overhead and search state). Reference phone search is
approximately `O(T * 40 * beam_width)` plus prefix bookkeeping and lexical search.
CPU search is a reasonable initial implementation; GPU visual inference and
optional Qwen are separate costs. No latency or memory target has been measured.

Log request ID, producer/checkpoint hash, resource/config hashes, valid length,
beam sizes, state counts, stage latencies, blank statistics, failure reason, and
whether reranking changed the winner. Keep transcripts and dense emissions out
of routine logs; explicit research exports may retain them locally. No network
calls, corpus downloads, shell execution, or model downloads during decoding.
An identical immutable input/resource/config bundle must reproduce scores within
a declared numeric tolerance and the same stable ordering. Cache by the full
bundle, never only by a clip name. A failed optional scorer must not poison cached
base results.

`no_lexical_path` means no complete nonempty path survived the retained phone
and lexical beams under the empty-output rule above. It does not establish OOV
or exhaustive dictionary failure. Set `approximate_search: true` for beam search;
measure lexicon coverage separately. Diagnostics include `error_code`,
`empty_prefix_log_mass` (null if impossible), and `reranker_filtered_count`.
For failures before search, use `search_mode: not_applicable`. Unparseable or
structurally invalid requests use the same envelope with nullable request ID
and hash; echo a valid ID when available. The hash is SHA-256 of exact received
UTF-8 request bytes, or null if bytes are unavailable. This intentionally identifies
an artifact, not a canonical semantic object. Whitespace changes may miss cache
hits; they must not alter scores. Cache keys include this byte hash plus all
resource/config/implementation hashes; request ID participates via those bytes.
No interoperable semantic-deduplication guarantee is made.

All finite score weights have magnitude at most 100 in this contract; this is
an operational guard, not a tuned recommendation. Default pronunciation weight
is 1; any tuning uses validation data. Reject nonfinite accumulated scores as
`resource_limit` with `error_code: numeric_overflow`. Null-only word timing is
intentional; introducing estimates requires a new response schema version.

## Training and evaluation

Freeze the producer checkpoint, crop, vocabulary, lexicon, and dataset manifests
before decoder comparisons. Use speakers 1–8 for model/LM training, 9–10 for
selection, and 11–12 only after choices are fixed. Record prior exposure: if a
split was inspected in earlier experiments, disclose that history. This document
does not establish that any local split is newly untouched.

Use published word transcripts as word references. MFA phone alignments can
support diagnostics but are generated supervision, not human phonetic truth.
CMUdict is a pronunciation resource, not a sentence language-model corpus. Train
a smoothed LM only from permitted training text or a separately declared external
corpus. Preserve homophones and group duplicate/alternate dictionary entries
when splitting any dictionary-based learning task.

Evaluate the same emission cache with:

1. Existing greedy phones → existing Qwen baseline, where applicable.
2. CTC beam → exact lexicon, no LM.
3. Same search + word LM.
4. Separately labeled GRID grammar constraint.
5. Joint search only if needed, then optional candidate-only Qwen reranking.

Report corpus WER, per-speaker WER, phone PER, N-best oracle WER, fraction of
references available in N-best, OOV rate under the declared lexicon, no-path rate,
empty-output rate, and p50/p95 latency/peak memory on a named device. Publish
beam-width sweeps and lexicon coverage so search loss is separable from visual
model error. Confidence intervals should resample utterances within speakers;
with only two test speakers, those intervals do not estimate population-wide
speaker uncertainty. A letter-name normalization policy and punctuation/case
rules must be frozen for reference and hypothesis alike.

Acceptance gates:

- Tiny exhaustive CTC tests agree with unpruned beam masses, including repeats,
  blank-separated repeats, zero probabilities, and all-blank input.
- Lexicon tests cover ambiguous segmentation, shared prefixes, repeated phones
  across words, homophones, alternate pronunciations, duplicate suppression,
  unknown words, and empty pronunciations rejected.
- Equivalent `[T,B,C]` and `[B,T,C]` producer fixtures yield identical emissions;
  padding never enters the decoder. Sparse top-k and non-CTC probes are rejected.
- Empty/no-evidence and all-blank statuses are distinct; malformed output scores
  and impossible timing arrays fail semantic validation.
- All supplied example JSON files pass their structural schemas. This is a
  contract check, not a beam-search correctness or accuracy result.
- Keep a new LM/reranker stage only with measured validation WER improvement;
  freeze it before final evaluation. Report ties/regressions rather than silently
  increasing language weight until test text looks right.

## Rollout and open decisions

First add a full-emission export option to the compact predictor without changing
its default result. Add the reference decoder as a separate package/command.
Compare cached emissions offline, then expose an opt-in integration flag. Existing
greedy behavior remains the rollback path. Any future streaming design needs
chunk IDs, overlap reconciliation, prefix state persistence, endpointing, and
retraction rules; the current noncausal compact model also needs a declared
look-ahead policy. None is implied by this offline contract.

Open implementation decisions: resource-loader format for a full CMUdict trie;
word LM training corpus and smoothing; beam sizes; calibration procedure; whether
joint search justifies its dependency cost; and the candidate-token scoring
convention for Qwen. Defaults in schemas are starting experiment settings, not
optimized parameters. No remaining decision blocks using this write-up as the
implementation framework.

## External design feedback

The bundled write-design-doc reviewer ran on 2026-09-11 using the local
DeepSeek-backed launcher. Full original feedback: [design.feedback.md](design.feedback.md).
It reviewed the design and request/response schemas, not the lexicon or tests.

| Finding | Disposition |
|---|---|
| C1: beam failure is not exhaustive lexical absence | Accepted: defined pruning-conditional status and separated coverage measurement. |
| C2: pre-search errors cannot be represented | Accepted: added not-applicable search mode and nullable identifiers. |
| C3: unspecified canonical hashing | Modified: exact-byte artifact hashing is explicit; semantic cache deduplication is not promised. |
| M1, M4: resource weights and calibration contradictions | Accepted: schema conditionals reject inconsistent combinations. |
| M2: missing trie state | Accepted: trie node is explicit. |
| M3, M5: cross-field and pronunciation correspondence | Accepted: semantic invariants and implementation acceptance cases specified. |
| Minor: recurrence, mass, diagnostics, IDs, bounds, timing | Accepted: recurrence and diagnostic definitions added, weights bounded, timing versioning clarified. |
| Questions: CTC producer, lexicon variants, pronunciation tuning | Clarified: blank-trained producer, distinct lexicon hashes, validation-only tuning. |

These corrections tighten the selected architecture's contract without changing
its staged-search decision. Executable checks cover structural conditionals and
fixture arithmetic; a future decoder must implement the listed semantic rejection
and search acceptance tests.
