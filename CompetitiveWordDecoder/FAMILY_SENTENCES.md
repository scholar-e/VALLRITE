# Family-to-sentence pilot

The pilot continues the clean EOS Llama 3.2 3B adapter, leaving it intact.
Input is a complete multi-word phone sequence mapped to 17 visual families.
Output is the original transcript, supervised through EOS. Each family has a
fixed A–Q code recorded in the manifest. Repeated families remain separate
positions. No word boundaries, REST, audio, or preceding transcript are supplied.

Training uses WikiText-2 training rows only: dictionary-covered multi-word
examples, excluding unknown spellings and nonstandard dictionary phones.
Normalized duplicates of LRS3 evaluation transcripts are removed. Complete
input/target/EOS pairs must fit 512 tokens; no truncation is permitted.
Source filtering retains 6,034 training rows before length filtering.
WikiText validation measures loss and a 25-example generation check; the
WikiText test split is not evaluated in this experiment.

Settings: BF16, batch 4, one epoch, learning rate 2e-5, 30-minute training cap.
BF16 replaced FP16 after the smoke test exposed gradient overflow. The existing
thermal watchdog runs during training and final LRS3 evaluation. Checkpoints
save every 250 steps, retaining two, and at a requested pause. The time cap
applies to training, not preparation or final evaluation.

Start: `bash tools/train-family-sentence-pilot`.
Pause: `./pause-training checkpoints/vallr-families-20261011`.
After removing that run's PAUSE file (and root PAUSE if present), resume with
`bash tools/train-family-sentence-pilot --resume`.
Do not use resume after the final evaluation directory already exists.
Monitor: `journalctl --user -u vallrite-family-sentence-20261011 -f`.

Successful completion automatically evaluates the same 1,140 LRS3 reference
sequences as the 13.40% individual-phone WER baseline, mapped into family codes.
A pause or time-limit exit before finishing the epoch does not launch that
evaluation. Results: `checkpoints/vallr-reference-families-20261011/metrics.json`.

This is a reconstruction diagnostic, not visual WER or a clean causal estimate
of information loss: the continuation changes training data and representation.
A bad result may indicate insufficient adaptation rather than an intrinsic
limit of family decoding. No noisy visual inputs or N-best sentence decoding yet.

The decoder environment additionally requires the real PyPI dependency
`opencv-python-headless==4.13.0.92`, because the shared visual-family inventory
imports the visual data module. Installed with:
`uv pip install --python .venv-qwen/bin/python opencv-python-headless==4.13.0.92`.

## Explicit alternatives pilot (2026-10-12)

The A–Q pilot finished one epoch but scored 108.05% WER on the 1,140 reference
family inputs (13,695 edits / 12,675 words), versus 13.40% for reference phones.
This motivates a separate continuation from the original clean EOS adapter,
not from the failed A–Q adapter. Inputs now use `[B/M/P]`, `[F/V]`, etc. with
fixed member ordering independent of the true member. Repetitions are retained;
there are still no word boundaries or unknown-word spellings in the prompt.

Launch: `bash tools/train-family-alternatives-pilot`.
Run: `checkpoints/vallr-alternatives-20261012`.
Evaluation: `checkpoints/vallr-reference-alternatives-20261012/metrics.json`.
Monitor: `journalctl --user -u vallrite-alternatives-20261012 -f`.
Pause: `./pause-training checkpoints/vallr-alternatives-20261012`.
Resume before evaluation exists: remove the requested PAUSE marker and rerun
the launch command with `--resume`.

The context limit is 768 instead of 512: the longest matched evaluation prompt
is 581 tokens, and six exceed 512. No evaluation clip is truncated or excluded.
The longer representation changes which WikiText training examples fit; this
is a practical pilot, not a strictly controlled encoding-only ablation.
Other settings remain one epoch, BF16, batch 4, lr 2e-5, 30-minute training cap,
250-step resumable checkpoints, and thermal protection. Evaluation batch is 8.
If the time cap stops training mid-epoch, the saved partial adapter is evaluated
and its training epoch is included in the report. User-requested PAUSE prevents
automatic evaluation. Preparation and final evaluation are outside the training
cap. The run is monitored with the short-wait script through final scoring.

### Completed result

Training completed all 992 steps (one epoch) in 1,043 seconds, about 17.4 minutes.
Length filtering kept 3,965 training and 421 validation examples. The matched
LRS3 evaluation completed in 198 seconds. Ordered clip IDs, reference words,
and source hashes match both earlier reference-input reports exactly.

| Input / adapter | WER | Word edits /12,675 |
| --- | ---: | ---: |
| Individual phones / clean EOS | 13.40% | 1,699 |
| A–Q families / code continuation | 108.05% | 13,695 |
| Explicit alternatives / alternatives continuation | 115.07% | 14,585 |

The alternatives pilot produced zero exactly correct clips and nine generations
at the 256-token cap. It worsened WER by 7.02 percentage points relative to the
A–Q pilot. Do not promote either family adapter. These are reference-input
reconstruction results, not visual-input WER. Different length-filtered training
pools and context lengths mean this is not a controlled encoding-only ablation.
The pilot does not establish an intrinsic limit on sentence context or families.

Keep the clean individual-phone decoder. A next option is to generate and rank
concrete phone paths from a family lattice using training-only pronunciation
statistics, then pass those paths to the existing decoder. That is a proposed
follow-up, not an experiment started by this run. All 41 targeted tests pass.
