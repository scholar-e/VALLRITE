# Design review: Probability-preserving phoneme-to-word decoding

Scope: only `DESIGN.md`, `request.schema.json`, and `response.schema.json` were reviewed. `lexicon.schema.json` was not supplied. Tags: **[V]** verified contradiction, **[U]** under-specified/uncertain.

## Critical

### C1. `no_lexical_path` claims an exact condition that the beam search cannot establish **[V]**

The design first states that phone-beam pruning "can remove the best lexical answer," then defines `no_lexical_path` as "No complete dictionary path." Those are in tension: a pruned phone prefix may be exactly the one that the lexicon could have completed.

- **Impact:** failure status will conflate genuine lexicon/OOV failure with beam-pruning loss. An evaluation counting `no_lexical_path` as "reference unavailable in lexicon" would be wrong, which undermines the stated goal of explicit search failures.
- **Correction:** define the status as `no_lexical_path_in_beam` (or "no lexical path among retained prefixes"). Reserve unqualified `no_lexical_path` for an exhaustive/unpruned search mode, and otherwise set `approximate_search: true` plus a warning that the absence is pruning-conditional. Include the retained empty prefix in that diagnostic.

### C2. Error and non-search responses are not representable by `response.schema.json` **[V]**

`diagnostics.search_mode` is `required` with enum values only `reference_two_stage` and `exhaustive_fixture`. For `no_evidence`, `invalid_input`, `resource_mismatch`, and `resource_limit`, no search may have run, so no valid value exists. Similarly, if a request is malformed before `request_id` is known, the response cannot satisfy the required `request_id` echo.

- **Impact:** the decoder contract has no valid response for several of its own declared statuses. Implementers must either lie in `search_mode` or violate the schema.
- **Correction:** make `diagnostics` conditional per status, or add a `search_mode` value such as `not_applicable`/nullable. Define a separate error envelope for requests that fail JSON Schema or cannot be hashed/identified, with `request_id` and `request_sha256` optional there.

### C3. `request_sha256`, bundle caching, and reproducibility require a canonical serialization that is not specified **[U]**

The design promises that "an identical immutable input/resource/config bundle must reproduce scores" and to "cache by the full bundle," and the response echoes `request_sha256`. But no canonical JSON encoding is defined (key ordering, number formatting, whitespace, `-0`, escaping). The same semantic request can produce different hashes and cache keys across serializers.

- **Impact:** the reproducibility and cache-deduplication guarantees are not implementable interoperably; hashes become accidental artifacts of the client serializer.
- **Correction:** specify canonical JSON bytes for hashing (sorted keys, fixed number representation, normalized Unicode/escaping, no trailing zeros), state that `request_sha256` is the SHA-256 of that encoding, and document whether `request_id` participates in the cache key.

## Major

### M1. Null resources with nonzero weights are not a declared validation condition **[V/U]**

The design says `reranker=none` requires weight 0, but the schema does not enforce it (`reranker_weight` may be >0 while `resources.reranker` is null). There is no equivalent rule at all for `language_model: null` with `lm_weight > 0`.

- **Impact:** contradictory configurations can be accepted silently, either ignoring resources or attempting to load null resources.
- **Correction:** add explicit semantic validation (and ideally JSON Schema `if/then`) requiring `resources.reranker != null` whenever `reranker_weight > 0`, and define whether `lm_weight > 0` with null LM is `resource_mismatch` or treated as a no-LM implementation.

### M2. Lexical search state omits the current trie node **[U]**

The state is defined as `(phone_offset, LM_state, words/pronunciation backpointer)`. For a trie terminal that also has children, deciding between a word boundary and continued traversal requires knowing the current trie position/partial-word prefix, which is not in the tuple.

- **Impact:** ambiguity in the core DP; two implementers can build different (and incompatible) state machines.
- **Correction:** include `(phone_offset, trie_node, LM_state, backpointer)` explicitly, or state precisely how `trie_node` is derived from `phone_offset` and the emitted words.

### M3. Cross-field equality/length checks are under-specified for semantic validation **[U]**

`producer.valid_steps` should equal the number of probability rows, `step_times_ms` (when non-null) must be exactly `T` strictly increasing values, and `observed_steps` (when non-null) must be exactly `T` booleans. The prose mentions most of this but does not list `valid_steps == T` as a validation rule, and JSON Schema alone cannot express equality or strict monotonicity.

- **Impact:** malformed fixtures can pass the structural schema and reach search with mismatched timing/valid-length metadata.
- **Correction:** enumerate these as first-class semantic validation rules and add acceptance tests for each mismatch.

### M4. `calibration.status` semantics are internally ambiguous **[V/U]**

The design says to store "softmax(logits / temperature) with the temperature and its validation-artifact fingerprint," and the schema requires both `temperature` and `artifact` regardless of status, yet the enumeration allows `raw_softmax` with a non-1 temperature and `temperature_scaled` with a null artifact.

- **Impact:** the provenance of scores is ambiguous; `raw_softmax` with `temperature=2.0` is both raw and scaled.
- **Correction:** constrain `raw_softmax` to `temperature == 1` (artifact may be null or producer/preprocessing fingerprint), and require non-null artifact for `temperature_scaled`. Make these schema-level `if/then` constraints.

### M5. Candidate pronunciation mapping is not sufficient for decomposability **[U]**

`representative_path.pronunciation_ids` is not tied to `word_count`, and candidate `words` carry no per-word pronunciation id. For words with multiple pronunciations, the response does not show which pronunciation was used for each emitted word.

- **Impact:** weakens the stated "decomposable scores" goal and makes acceptance/debugging of homophone alternatives harder.
- **Correction:** enforce `pronunciation_ids.length == word_count` for the representative path (or add `pronunciation_id` on each word object), and document that non-representative pronunciation contributions are hidden in `base_score`.

## Minor

- The CTC recurrence prose "extends the repeated phone only from its previous blank mass" is ambiguous about whether "previous" means the same prefix or the shorter prefix. Give the exact log-space recurrence.
- `retained_prefix_log_mass` is never defined precisely (logsumexp over retained final prefixes? including/excluding the empty prefix?). It will be misread as coverage anyway.
- "Machine-readable diagnostic" is promised, but the response provides only `failure_reason` and string `warnings`; add a structured error-code field.
- `step_times_ms` strict monotonicity and `observed_steps` length are not expressible in JSON Schema; this should be explicit in the semantic-validation text.
- Candidate `id` has no uniqueness format; scoring sort only mentions "stable word IDs," not candidate IDs.
- `word_insertion_bonus`, `pronunciation_weight`, `lm_weight`, and `reranker_weight` are unbounded; an upper sanity bound would prevent overflow/underflow from effectively changing the ranking.
- `words[].start_ms/end_ms` are typed as null-only, so future timing support requires a schema version bump; acceptable but worth a comment.

## Questions

- Is the compact visual model actually CTC-trained with a blank class? The design rejects fake CTC probes but does not state that the primary producer satisfies the CTC emission assumption.
- How is `request_sha256` computed when the request body cannot be parsed as JSON or is missing `request_id`?
- Do the lexicon ID/hash distinguish the GRID letter-name configuration (`a → EY`) from unrestricted English with article pronunciations? If not, a single hash could load the wrong lexical interpretation.
- Is `pronunciation_weight` intended as a tuned parameter or fixed at 1? Identifiability is only claimed for the visual coefficient.
- Should `no_lexical_path` also distinguish "no path survives the lexical beam" from "no path in any retained phone sequence"? Lexical-beam pruning can independently remove completions.
- `lexicon.schema.json` was not provided, so constraints on empty pronunciations, prior sums, and caps could not be checked against the prose.

## Positive observations

- The CTC semantics are carefully separated from producer geometry: blank/repeat collapse, cross-word repeat requiring a blank, and rejection of fake CTC columns are correctly stated.
- The empty-vs-all-blank distinction (`no_evidence` vs `ok` with empty candidate), zero-probability handling, log-sum-exp, and refusal to expose calibrated confidence are all sound.
- The weighted-objective transparency (`base_score`, `final_score`, representative path, no softmax-faked confidence) is honest about not being a normalized posterior.
- Resource immutability, no network/shell side effects, cache poisoning protection for failed rerankers, and per-request isolation are appropriate operations posture.
- The evaluation plan correctly separates visual error from search loss, freezes splits/resources, and requires kept LM/reranker stages to earn their cost on validation WER.
