# Phoneme smoke test — 2026-09-05

The current system is not producing usable sentence phonemes on this test.
Greedy phoneme error rate (PER) was **90.625% (29 edits / 32 reference phones)**
both before and after automatic face/neck cropping. This is a two-clip diagnostic,
not a held-out benchmark or an estimate of general accuracy.

## Data and reference

Downloaded only the two official normal-quality
[GRID demonstration videos](https://spandh.dcs.shef.ac.uk/gridcorpus/), approximately
845 KiB combined. Both contain 75 frames at 25 FPS. No training or parameter
selection was performed. Speaker/checkpoint training overlap is unknown.
The corpus page permits research use; corpus authors are Martin Cooke,
Jon Barker, Stuart Cunningham, and Xu Shao. Downloaded media stays in the
ignored `datasets/grid-smoke/` directory.

Published word transcripts:

- Speaker 2: [set white with P two soon](https://spandh.dcs.shef.ac.uk/gridcorpus/examples/swwp2s.align).
- Speaker 23: [place red in A zero now](https://spandh.dcs.shef.ac.uk/gridcorpus/examples/priazn.align).

References are stress-free broad English dictionary pronunciations checked
against the installed CMUdict via `pronouncing`. P and A are letter names
(`P IY` and `EY`), not function words. These are transcript-derived reference
phones, not manually verified realized speech phones. The manifest records
alternate pronunciations for white, with, and zero. Using the best listed
alternative per clip changes the cropped macro PER to approximately 87.87%;
it does not make the outputs useful. The headline uses the fixed canonical
reference, with 16 phonemes per sentence, selected before inference.

## Frozen results

| Clip | Input | Predicted ARPAbet | Substitutions | Deletions | Insertions | PER |
|---|---|---|---:|---:|---:|---:|
| Speaker 2 | Original | W TH W SH Z | 3 | 11 | 0 | 87.50% |
| Speaker 2 | Face/neck | HH L N P | 3 | 12 | 0 | 93.75% |
| Speaker 23 | Original | T Z V SH L Z | 5 | 10 | 0 | 93.75% |
| Speaker 23 | Face/neck | N N | 0 | 14 | 0 | 87.50% |

PER is `(substitutions + deletions + insertions) / reference length`, computed
by token-level Levenshtein alignment. This is not word error rate or a
frame-classification accuracy. Fewer errors in one cropped example does not
establish a cropper improvement; its prediction contains only two phonemes.

## What failed

1. **Sentence output capacity:** the existing CLI takes 16 uniformly sampled
   frames for the entire clip. The checkpoint emits eight CTC steps; any CTC
   decoding of those steps can produce at most eight phonemes. Both fixed
   references contain 16. Even perfect labels at those steps would have at least
   50% PER from deletions. Beam search alone cannot remove this limit.
2. **Cropping is not reliable across these examples:** the cropper and subsequent
   VPA landmark extractor retained 28/75 frames (37.33%) for speaker 2, and
   75/75 (100%) for speaker 23. Missing crop frames were black, as documented;
   the VALLR model has no missing-frame mask. This test measures the shipped crop
   path, including those failures. The successful lecture example from the
   earlier smoke test was not evidence of general tracking reliability.
3. **VPA is not yet a phoneme predictor:** both full VPA runs validated successfully,
   but returned empty `phoneme_hypotheses`. Geometry gestures are auxiliary
   diagnostics; there is no trained VPA phoneme head or fusion with VALLR yet.
4. **Probabilities remain uncalibrated:** raw softmax outputs are retained for
   inspection. Two clips cannot establish calibration or lattice recall.

The test preserves the existing VALLR CLI preprocessing: 16 sampled RGB frames,
224×224 resize, floating-point pixel values in 0–255, greedy CTC, no Qwen or other
language model. Cropped inputs use the shipped 256×256 letterboxed MP4 output.
This does not reproduce official VALLR evaluation and cannot distinguish model
weakness from checkpoint/preprocessing mismatch. No input normalization or
other tuning was selected using these examples.

The next implementation work should verify checkpoint vocabulary and official
preprocessing, fix sentence-length temporal inference, improve face tracking,
and only then evaluate on a speaker-disjoint labeled set. Adding more gesture
names or a language-model rewrite would not resolve the measured failures.

## Reproduce

From the repository root, use the existing Qwen environment for the VALLR model
and the VPA environment for cropping. The latter is `/tmp/vpa-venv` in this
session; substitute your local `.venv-vpa` as needed.

```bash
curl -L --fail -o datasets/grid-smoke/id2_vcd_swwp2s.mpg \
  https://spandh.dcs.shef.ac.uk/gridcorpus/examples/id2_vcd_swwp2s.mpg
curl -L --fail -o datasets/grid-smoke/id23_vcd_priazn.mpg \
  https://spandh.dcs.shef.ac.uk/gridcorpus/examples/id23_vcd_priazn.mpg
/tmp/vpa-venv/bin/python -m VisualPhoneticAlphabet.crop \
  datasets/grid-smoke/id2_vcd_swwp2s.mpg \
  --output datasets/grid-smoke/id2_vcd_swwp2s-face-neck.mp4
/tmp/vpa-venv/bin/python -m VisualPhoneticAlphabet.crop \
  datasets/grid-smoke/id23_vcd_priazn.mpg \
  --output datasets/grid-smoke/id23_vcd_priazn-face-neck.mp4
.venv-qwen/bin/python -m VisualPhoneticAlphabet.evaluate \
  --output /tmp/grid-smoke-results.json
/tmp/vpa-venv/bin/python -m unittest discover \
  -s VisualPhoneticAlphabet/tests -v
```

Crop commands require new output paths; skip them if the artifacts already exist.
The saved `grid-smoke-results.json` includes source/checkpoint hashes, versions,
full raw posterior distributions, error alignments, CPU forward times, and crop
coverage. All 15 tests passed, including five error-metric tests. The original
and cropped conditions were each evaluated once with the same checkpoint.
