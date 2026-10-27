# Decoder contract audit and error-tolerance diagnostic

Completed 2026-10-26. No model weights changed. This completes the contract
audit and tolerance-measurement stages; current-model error training and a
multi-candidate-input training comparison have not yet been run.

## Upstream compatibility

Inspected the saved original commit
`db932ca83ff8b2b858eb6a4c78d02b5e3805c299`, rather than treating the modified local
VALLR files as originals. Its `Models/Llama.py` uses CMUdict first pronunciations,
stress removal, a space-separated phone line and the four tags used locally.
Although `text_to_phoneme_line` describes pipes between words in its docstring,
its implementation returns a space join. Four directly executed upstream prompt
fixtures, including an unknown word, match our clean training prompt exactly.
The local evaluator uses the same tags, no chat template and no added BOS.

The inspected upstream dataset helpers also remove stress and choose the first
pronunciation. Unknown-word behavior differs across upstream helpers: the Llama
recipe preserves `UNK(word)` spelling, whereas some visual-data helpers skip
unknown words. That is a reference/coverage caveat, not evidence that adding
word separators would repair our decoder. Our visual exports use explicit phone
strings rather than handing the word decoder raw numeric class IDs.

Important intentional differences remain: our continuation explicitly supervises
EOS and excludes overlong complete pairs instead of retaining upstream truncation.
The current clean model is a local recipe adaptation, not a released original
full-system checkpoint. This audit does not validate the published end-to-end
VALLR evaluation or establish its decoder's performance on our visual errors.

## Experiment

Select 100 of the existing development clips with seed20261026, without filtering
by results. Keep these same sentences for every condition and both adapters.
The reference count is1,057 words. Clean phones come from the saved reference;
actual input is the context model's rank-one beam candidate, not greedy output.

Estimate substitution/deletion/insertion frequencies and per-phone substitution
destinations using minimum-edit alignment of the older model's5,000 training
predictions only. Assert all source IDs belong to the training manifest and are
disjoint from development. Alignment ties prefer diagonal, then deletion.
Sample edit locations uniformly, operation types from training counts,
substitution destinations conditioned on the source phone when observed, and
inserted phones from training insertion counts. No development predictions or
references are used to estimate these error distributions.

Nominal edit budgets are0/20/40/60/80%; report actual minimum-edit PER because
multiple operations can admit a shorter alignment. This model does not simulate
correlated bursts, coarticulation, phone-specific deletion probability, timing,
or the new context model's entire error distribution. It is a controlled stress
test, not a replacement for current-model error data. It uses one corruption
seed and100 repeatedly inspected development clips, with no significance claim.

Both decoders use the same saved base-model revision, FP16 inference, tagged
prompt, greedy generation, batch16, supervised EOS ID, maximum256 new tokens,
and unchanged word normalization. Prompt truncation is forbidden. Every condition
contains the same100 clips; no length exclusions occurred. Thermal protection
and cooperative pause were active. Inputs, training-derived counts, source and
adapter hashes, per-sentence predictions and completed metrics are saved under
`checkpoints/decoder-tolerance-20261026`.

## Results

| Input | Realized exact PER | Clean EOS decoder WER | Actual-error-adapted decoder WER |
| --- | ---: | ---: | ---: |
| Clean reference phones | 0.00% | 12.30% | 6.81% |
| Nominal20% edits | 19.92% | 69.91% | 110.50% |
| Nominal40% edits | 39.29% | 91.01% | 100.66% |
| Nominal60% edits | 58.05% | 98.20% | 141.25% |
| Nominal80% edits | 73.32% | 99.24% | 140.78% |
| Actual context-model beam input | 83.35% | 96.31% | 98.58% |

WER above100% is possible because of insertions; it is not a percentage of
sentences classified incorrectly. At nominal20% both decoders terminate without
hitting the generation cap, so that regression is not merely a cap artifact.
The non-monotonic adapted curve does not establish an optimal error rate.

The adapted model's improved clean reconstruction has not translated into
robustness. The clean decoder also deteriorates sharply at moderate corruption.
Therefore the poor pipeline cannot be dismissed as only an upstream problem;
decoder robustness is worth addressing. Conversely, these experiments do not
show that a decoder can recover information lost at approximately83% phone PER.

## Decision and next training comparison

Keep the clean EOS adapter as the robustness baseline. Do not extend the latest
actual-error adapter by default. Prepare current-context-model training errors
with explicit exposure provenance, then compare clean-only domain adaptation
against a matched clean/noisy curriculum from the same initial adapter, with
equal transcript pools and update budgets. Include moderate error levels rather
than training only on severely corrupted examples. Validate both real visual
WER and clean-phone retention; the tolerance grid is a diagnostic, not a new
benchmark to tune repeatedly.

After obtaining that matched baseline, test multiple candidates and their scores
as an explicit separately trained input representation. Simply concatenating
five alternatives into the existing single-sequence prompt would introduce an
untrained format and would not be a fair uncertainty-aware comparison. Keep
fixed ranking as a lower-cost reference. No new training or representation
experiment was launched in this diagnostic stage.

Implementation: `decoder_tolerance.py`. Minimum-edit alignment counts passed
100 random comparisons against the existing edit-distance function. Upstream
prompt fixtures and repository whitespace checks pass. Partial condition files
are retained on pause; this diagnostic does not currently resume generation.
