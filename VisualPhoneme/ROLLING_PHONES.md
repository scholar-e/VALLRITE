# Rolling phone probabilities and timed segments

2026-10-29. User requested a rolling average phone string to reduce loss of timing
and confidence. Implemented `rolling_phones.py` and a cached-logit pilot in
`evaluate_rolling_phones.py`. No new dependencies or visual training.

## Representation

Average distributions over blank plus 39 phones, not phone IDs or strings.
A trailing window uses only observed frames and a shorter denominator at clip
startup. `RollingPhoneAverage` preserves its history across chunks; construct a
new instance for a new clip. Five unit tests cover chunk invariance, causality,
normalization, exact identity, invalid cached repeats, blank-separated repeated
phones, time conversion and blank-preserving smoothing.

The current model repeats each original frame's logits twice for CTC. The pilot
verifies and removes those exact repeats before smoothing, then restores the two
alignment states for the unchanged beam search. A three-frame window at 25 fps
covers 120 ms, with a 40 ms mean historical lag once full; five frames cover
200 ms with 80 ms mean lag. This is a trailing filter, not additional lookahead.
The upstream visual network still has symmetric context and the beam export
still decodes whole clips: this is not an end-to-end streaming recognizer.

Each export contains:

- Five ordinary beam phone candidates, compatible with the existing word decoder.
- A separate `greedy_rolling_string` and `rolling_greedy_segments`, including
  start/end frames and seconds, selected-phone posterior mass and the complete
  mean distribution for that segment. Blank runs are retained as `ctc_blank`.
- `rolling-posteriors.pt` containing all per-original-frame distributions, FPS,
  clip IDs and the exact label order. This preserves alternatives for later use.

The segment boundaries are frame-argmax runs, **not alignments of the beam
strings** and not manually verified articulatory or word boundaries. CTC blank
is not silence. Posterior mass after a blank bias is not calibrated confidence
that the phone is correct. These distinctions are recorded in each exported row.
The current word decoder consumes only the five phone strings; attaching metadata
alone does not teach it to use timing or distributions. A future encoder/interface
and matched training are required for that benefit.

## Pilot protocol and first result

Use the existing constant-head/fixed-crop caches, batch policy unchanged. Compare
windows 1 (identity), 3 and 5; keep blank bias -1, beam64/top-k12/top5, no phone
LM. Choose by rank-one PER on the existing 100-clip validation split, ties favor
smaller windows. Freeze before evaluating the selected setting on the separate
100-clip pilot evaluation split. These are previously inspected pilot splits,
not untouched official test data. Hash all sources and save selection separately.
No word references or phone targets enter the smoothing operation.

| Plain probability average | Validation PER | Deletions | Substitutions | Insertions |
| --- | ---: | ---: | ---: | ---: |
| Window 1 | 70.60% | 1,180 | 1,972 | 166 |
| Window 3 | 73.43% | 2,195 | 1,225 | 31 |
| Window 5 | 76.66% | 2,620 | 965 | 18 |

All use 4,700 reference phones. Window one exactly reproduces every saved phone
candidate and score. It wins selection; evaluation remains 70.87% PER. The
longer windows trade substitutions/insertions for many more deletions. Smoother
posteriors are not necessarily better recognition: CTC relies on brief emissions.

## Targeted diagnostic extension

Declared after observing the deletion increase, before its results: keep the
instantaneous blank probability, and smooth only relative nonblank identity.
For each time t, compute the trailing average q. Replace its blank with p_t(blank)
and renormalize q's nonblank coordinates to the current nonblank mass. This is
an emission-mass-weighted average of conditional phone identity across the window.
It preserves the blank/emission timing signal while testing phone smoothing.
Again compare windows 1/3/5 on validation and retain identity if it wins. No
additional parameter tuning or training is introduced. This is exploratory,
not an independent confirmation of the first pilot.

Plain artifacts: `checkpoints/rolling-phones-20261029`.
Extension artifacts: `checkpoints/rolling-phones-blank-preserved-20261029`.
The extension completed:

| Preserve instantaneous blank mass | Validation PER | Deletions | Substitutions | Insertions |
| --- | ---: | ---: | ---: | ---: |
| Window 1 | 70.60% | 1,180 | 1,972 | 166 |
| Window 3 | 72.85% | 1,712 | 1,646 | 66 |
| Window 5 | 74.85% | 1,834 | 1,626 | 58 |

Preserving emission mass reduces the damage relative to plain averaging, but
both nontrivial windows still lose to identity. Identity is retained, with
70.87% PER / 69.96% oracle PER@5 on the 100-clip pilot evaluation split.
No smoothed setting was selected for a downstream word experiment. The selected
beam candidates, scores and constructed word-decoder prompts were verified
identical to the existing export, so another GPU word evaluation would repeat
the same inputs rather than test a changed condition. No new WER is claimed.

All posterior rows sum to one over 40 labels, and segment intervals cover each
original-frame timeline continuously with the manifest FPS. Fifty-seven relevant
tests pass, including five rolling-specific tests. No production checkpoint or
decoder default changed. Both CPU pilots completed within their 15-minute caps;
they use root/run PAUSE checks and reusable partial output. No training occurred.

The implemented representation is useful independently of smoothing: retain
the window-one timed posteriors and alternative phones as the information-rich
input for a future decoder. Uniform temporal averaging is not an accuracy fix
for this model. Training a decoder to consume those distributions is a separate
next step; the legacy word prompt still sees only hard phone strings.

Example commands:

```bash
.venv-qwen/bin/python -m unittest VisualPhoneme.tests.test_rolling_phones -q
.venv-qwen/bin/python -m VisualPhoneme.evaluate_rolling_phones \
  --output checkpoints/NEW-ROLLING-RUN --preserve-blank --max-minutes 15
```
