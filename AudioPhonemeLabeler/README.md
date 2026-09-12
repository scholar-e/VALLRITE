# Audio teacher phoneme labeller

This package creates audibly supervised ARPAbet labels for silent-video model
training. Its hybrid mode runs one or more large Whisper teachers for text,
Montreal Forced Aligner (MFA) for phone boundaries, and a direct Wav2Vec2 phone
recognizer as an independent acoustic check. Output uses the same
`tiers.phones.entries` alignment contract as the current GRID pipeline.

This is a machine-label pipeline, not ground truth. If MFA is unavailable or
fails, the fallback subdivides each Whisper word interval uniformly among its
CMUdict phonemes and records that approximation. The visual CTC trainer currently
consumes only phoneme order, so provisional internal boundaries do not enter its
loss.

The direct model emits multilingual IPA-like labels. A conservative checked-in
mapping converts common English phones to the visual model's 39-phone ARPAbet
inventory. Unmapped tokens reduce coverage and therefore confidence; they are
never guessed. Direct-phone agreement is computed against the transcript-derived
sequence before MFA changes timing.

## Quality controls

Every label includes the source SHA-256, teacher configurations and transcripts,
mean word confidence, language confidence, transcript agreement, CMUdict coverage,
direct-phone output and agreement, timing method, unknown words, and rejection
reasons. Defaults are quality 0.65, pronunciation coverage 0.95, transcript
agreement 0.75, and acoustic-phone agreement 0.60. Rejected clips remain
available for review but must not enter training.

Teacher consensus estimates transcription stability; it does not prove phonetic
correctness. Speaker-disjoint train, validation, and test assignment must happen
from source speaker identity before these files are admitted to a dataset.

## Install and run

Install the isolated optional dependencies:

```bash
.venv-vpa-gpu/bin/python -m pip install \
  -r AudioPhonemeLabeler/requirements-labeler.txt
```

A practical single-teacher pass uses Whisper large-v3-turbo:

```bash
.venv-vpa-gpu/bin/python -m AudioPhonemeLabeler \
  path/to/media --teacher large-v3-turbo --device cuda \
  --compute-type float16 --output-dir datasets/audio-teacher-labels
```

Hybrid stages default to `preferred`: failures retain an auditable fallback.
For dataset production, require all three paths:

```bash
.venv-vpa-gpu/bin/python -m AudioPhonemeLabeler \
  path/to/media --teacher large-v3-turbo --device cuda \
  --compute-type float16 --phone-teacher-mode required --mfa-mode required \
  --mfa-dictionary english_us_arpa --mfa-acoustic-model english_us_arpa \
  --output-dir datasets/audio-teacher-labels
```

For a consensus-filtered offline pass, run two teacher variants:

```bash
.venv-vpa-gpu/bin/python -m AudioPhonemeLabeler \
  path/to/media --teacher large-v3 --teacher distil-large-v3 \
  --device cuda --compute-type float16 \
  --output-dir datasets/audio-teacher-labels
```

Model weights are downloaded by `faster-whisper` on first use. Add
`--local-files-only` in controlled or air-gapped runs. The command is resumable:
existing output labels are reported as cached unless `--overwrite` is supplied.
`labels.jsonl`, `labels.json`, and `labeler.log` provide batch-level audit trails.
Teacher variants can share training lineage, so agreement is a useful rejection
signal rather than an independent correctness guarantee.

Teacher weights share `OUTPUT_DIR/.model-cache` by default. Override this with
`--model-cache-dir` when several labelled datasets should use one writable cache.

MFA is intentionally an external dependency because its supported installation
is separate from the Python inference environment. Install MFA, download the
`english_us_arpa` acoustic model and dictionary, and confirm `mfa` is on `PATH`.
Use `--mfa-mode off` for sequence-only experiments. Similarly,
`--phone-teacher-mode off` disables the direct acoustic check.

## Admission policy

Use `required` modes for new production data. The more permissive defaults are for
bootstrapping, diagnostics, and machines where MFA is not installed. Always audit
a speaker-stratified sample and keep source speakers isolated across data splits.

## Integration smoke result

The complete Whisper-plus-Wav2Vec2 CLI was exercised on GRID clip `sgwp9s` using
CPU INT8 inference. Whisper incorrectly produced “set cream of p soon”; the direct
teacher mapped 94.1% of its IPA output and agreed with that candidate at only
58.8%. The default 60% acoustic threshold rejected the label, as intended. With
cached weights the repeat run took about seven seconds. The local teacher cache is
3.9 GiB. MFA was not installed on this host, so the result explicitly records
`forced_alignment: fallback`.

The 60% threshold is provisional and was not estimated from a representative
corpus. Calibrate it on a separate, speaker-disjoint set with trusted transcripts
before admitting a large external dataset.
