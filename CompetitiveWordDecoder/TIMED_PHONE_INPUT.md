# Timed posterior input adapter pilot

2026-10-29. User authorized teaching the word decoder to use timed probability
evidence after the rolling-average pilots failed. Visual smoothing stays off.

## Implementation

Keep the existing hard-phone prompt and add a small learned residual to each
phone token's input embedding. A 44 → 128 → 3,072 MLP uses GELU, followed by a
bounded 0.1×tanh output. Its output matrix starts at zero, so initial prompts
are unchanged. The frozen Llama base and existing mixed LoRA adapter receive
no parameter updates. Only the new input MLP is optimized. This isolates a new
input path while preventing the language-model weight drift seen in the earlier
continuations. It does not guarantee that altered inputs cannot harm decoding.

For each predicted phone path, Viterbi-align it to the fixed visual CTC logits.
This is alignment of a **prediction**, never of the reference transcript. Use
blank bias -1 and the existing two repeated alignment states per video frame.
The 44 features per predicted phone are:

- The mean 40-class posterior across its assigned emission states, including blank.
- Start time in seconds divided by ten.
- Assigned duration in seconds.
- Preceding alignment gap in seconds.
- Normalized entropy of its mean posterior.

Map those features to the phone's actual BPE token offsets in the unchanged
VALLR-format prompt. Prompt delimiters, padding and target tokens get a zero
residual. Multiple tokens for one phone receive the same evidence. No label,
transcript, reference-phone sequence or reference boundary is included in the
input features. Transcript text supplies only supervised target labels.

This first interface summarizes distributions per predicted phone; it does not
preserve every frame, encode the entire phone lattice, or introduce missing-phone
positions. Blank timing is an alignment signal, not audio silence. Viterbi times
and posterior mass are model estimates, not calibrated truth. These limitations
remain even if the pilot improves word error.

## Fixed experiment

Use the frozen 750/100/100 source-video-disjoint manifests and cached fixed-crop
visual logits. Build rank-one training examples and five independent candidate
prompts per validation/evaluation clip. All complete prompt/target pairs must fit
512 tokens without truncation. Exact source, adapter and prepared-data hashes
are recorded. Historical encoder/adapter exposure is not fully audited; these
are pilot splits, not an untouched end-to-end test.

One epoch gives 188 updates at batch four (the last batch contains two clips),
AdamW lr1e-3, weight decay .01, gradient norm cap one, seed42. Save at47/94/141/188.
Train only the small FP32 input adapter; the frozen word model runs BF16 throughout
both control and adapted evaluation. EOS is supervised, prompt/padding labels
are masked, and model dropout remains disabled. Cap the training stage at15
minutes, including intermediate evaluation overhead, with root/run PAUSE,
atomic optimizer/RNG resume state and the existing thermal guard.

Evaluate generated validation WER at0/94/188, always generating five sentences
and applying the same phone-score-plus-exact-token-probability ranker. Conditional
scores use the same timed embeddings that produced the sentences. Choose minimum
validation ranked word errors, ties favor unchanged baseline then earlier step.
The zero-adapter generation route must reproduce ordinary hard-prompt generation
on the first validation batch before any training occurs. All base parameters
must have `requires_grad=False` and no gradients after the first update.

Only if a learned checkpoint beats baseline on validation, freeze that choice and
compare it with baseline on the100-clip evaluation partition. Do not evaluate
multiple candidate checkpoints on that partition. BF16 control scores may differ
from earlier FP16 evaluations; compare within this matched run. If baseline wins,
retain it and stop without another evaluation-set pass.

## Reproduction and artifacts

```bash
.venv-qwen/bin/python -m CompetitiveWordDecoder.timed_phone_dataset \
  --output datasets/decoder-training/NEW-TIMED-FEATURES

tools/with-vpa-gpu .venv-qwen/bin/python -m CompetitiveWordDecoder.train_timed_phone_adapter \
  --data datasets/decoder-training/NEW-TIMED-FEATURES \
  --output checkpoints/NEW-TIMED-ADAPTER --max-minutes 15
# Same command plus --resume restores the adapter, optimizer and RNG states.
# Original uncapped pilot reproduction/resume additionally requires
# --legacy-coordinate-bound; new runs default to a 5% relative vector bound.
```

Current data: `datasets/decoder-training/timed-phone-features-20261029`.
Current run: `checkpoints/timed-phone-input-pilot-20261029`.
## Initial results and targeted correction

The zero-residual generation identity check passed. Exactly 398,976 parameters
were trainable; the frozen language model had zero trainable parameters and no
gradients after the first update. All188 updates completed, with optimizer/RNG
resume state and checkpoints at0/47/94/141/188. On100 validation clips /1,345 words:

| Checkpoint | Ranked WER | Oracle WER@5 | Ranked insertions | All candidate cap hits |
| --- | ---: | ---: | ---: | ---: |
| Unchanged, step0 | 93.31% | 89.89% | 10 | 2 |
| Step94 | 126.84% | 123.87% | 461 | 20 |
| Step188 | 152.42% | 133.38% | 804 | 23 |

The baseline wins, so no evaluation-set pass was launched for this original
adapter. This is a negative pilot: the numeric input path runs and learns, but
its learned input changes produce more incorrect words. It does not establish
that timing/confidence information is unhelpful.

Post-run diagnosis found an architectural problem in the perturbation bound:
0.1 bounds each coordinate, not the full vector relative to the token embedding.
At step188 the median residual/base-embedding norm ratio was84.33%, maximum102.34%
on rank-one validation phone tokens. Calling that a small addition of evidence
would be misleading. Frozen model weights do not prevent large input drift.

Before observing corrected results, add a single diagnostic that clips each
residual vector to at most5% of its token embedding's norm. Reuse the learned
step188 direction; no further optimization or parameter sweep. Re-evaluate the
matched zero baseline and clipped residual on validation. Only a validation
winner proceeds to the same100-clip evaluation partition. This exploratory
extension is motivated by the measured magnitude problem, not a predeclared
independent trial. `bound_embedding_delta` tests both the vector bound and a
nonzero learning gradient at zero initialization.

Diagnostic: `checkpoints/timed-phone-input-capped-20261029`.
The diagnostic completed. The repeated zero baseline exactly reproduced its
prior metrics. With the5% relative bound, ranked validation WER was93.68%
(1,260 errors), versus93.31% (1,255 errors) for baseline. Oracle WER@5 was90.48%
versus89.89%. This removes most of the previous catastrophic regression but
does not improve the baseline, which remains selected. No evaluation-set pass
was launched, and no new adapter is promoted.

New training runs now default to the relative5% vector bound, with the bound
recorded in their manifest. `--legacy-coordinate-bound` explicitly reproduces
or resumes the original uncapped experiment.

## Training under the corrected bound

Completed one final controlled run from zero with the 5% relative bound applied
throughout training: `checkpoints/timed-phone-input-bounded-train-20261029`.
Its manifest differs from the original pilot only in the relative bound. It
completed all 188 updates with the same 750 examples, initialization seed,
optimizer settings and frozen upstream weights. All planned checkpoints and
optimizer/RNG resume state were verified.

| Checkpoint | Ranked validation WER | Word errors | Oracle WER@5 |
| --- | ---: | ---: | ---: |
| Unchanged baseline | **93.31%** | **1,255** | **89.89%** |
| Corrected step 94 | 93.75% | 1,261 | 90.56% |
| Corrected step 188 | 93.53% | 1,258 | 90.71% |

The baseline again wins the predeclared selection. Three extra errors at the
final step is a small difference on this sample, not evidence of a reliable
performance difference, but it supplies no measured improvement to promote.
No candidate checkpoint proceeds to another evaluation-set pass. No further
training runs were launched. Sixty-one relevant tests and whitespace checks pass.

The timed probability input path is implemented and actually trained; it is not
merely metadata attached to an unchanged prompt. The relative bound resolves
the large input perturbation problem in this pilot, while useful word recovery
remains unproven. Its per-predicted-phone summaries still cannot preserve every
frame or represent missing phone positions. Retain the existing decoder, and
do not infer from this small negative pilot that all timing-aware decoders fail.
