# GRID landmark probe — 2026-09-10

The original six lip landmarks matched the full 40-point representation in
this frame-classification experiment. The validation-selected ten-point subset
performed worse in the nonlinear check. This does not establish that six points
are sufficient for lip reading or that other points are unnecessary.

## Results

MLP test results averaged across seeds 17, 29, and 43:

| Input | Balanced accuracy | Accuracy |
|---|---:|---:|
| Original 6 points | 24.45% | 17.83% |
| Validation-selected 10 points | 22.70% | 15.81% |
| All 40 points | 24.31% | 16.57% |

The paired clip bootstrap difference against all 40 points was +0.15 percentage
points for the original six (95% interval −0.31 to +0.58), and −1.62 points for
the selected ten (−2.00 to −1.26). These intervals describe clip variation within
the two test speakers, not uncertainty across new speakers or training seeds.
The near-zero six-versus-40 difference is not an equivalence test.

Ridge selected ten points using the predeclared smallest-subset-within-one-point
rule on validation balanced accuracy. The selected IDs were
314, 84, 0, 311, 291, 312, 39, 318, 405, and 80. Selection was frozen before
test evaluation; the test results must not be used to select another subset.

## Data and limits

All 12,000 clips completed landmark extraction without extraction errors.
Assembly retained 11,564 clips and excluded 436 under the alignment checks.
There were 206,765 training, 42,304 validation, and 49,155 test frame samples.
Speakers 1–8 trained the probes, 9–10 controlled selection and early stopping,
and 11–12 supplied the test results. Normalization used training samples only.

Targets are audio-generated MFA phoneme alignments, not manually verified
phoneme boundaries. Windows use previous/current/next frames, including 40 ms
of look-ahead. Boundary and missing-landmark filtering restrict the evaluated
samples. The experiment measures phoneme frame classification, not sequence
PER, WER, or a comparison with a learned pixel encoder. Validation/test score
differences and seed variation warrant caution about speaker generalization.

Retain the original six as a low-cost comparison in future experiments. These
results do not justify expanding the landmark inventory or replacing image
features. Any further feature selection should use training/validation data
and a fresh evaluation protocol; this test split has now been inspected.

## Reproduce

From the VALLRITE root, with the existing aligned corpus, landmark cache, and protocol:

```bash
.venv-vpa-gpu/bin/python -m VisualPhoneticAlphabet.measure_landmarks assemble
.venv-vpa-gpu/bin/python -m VisualPhoneticAlphabet.measure_landmarks measure
```

Measurement requires CUDA. This run used the RTX 5090. The trainer's label
conversion was corrected to `torch.long` for indexing and cross-entropy.
All 19 Python tests passed without skips after installing the already-declared
`jsonschema` and `opencv-contrib-python` dependencies in the GPU environment.
`uv pip check` also passed. These packages support schema validation and the
OpenCV crop backend; no training rerun was needed for their installation.

Full metrics and confusion matrices: [grid-landmark-results.json](grid-landmark-results.json).
Local artifacts are in `datasets/grid-pilot/landmark-experiment/`: protocol,
sample quality, frozen selection, nine model checkpoints, and test predictions.
