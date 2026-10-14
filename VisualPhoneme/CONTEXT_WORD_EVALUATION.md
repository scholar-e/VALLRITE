# Context checkpoint through the frozen word decoder

Completed 2026-10-14. This is development evaluation, not a new training run.

Evaluate `fullclip-phone-context-20261013/phones-step-0750.pt`, selected previously
at 82.6717% exact greedy PER. `evaluate_nbest.py` now supports the explicitly
tagged context architecture via `load_phone_export`, image-only data and image
collation. The checkpoint weights and decoder are unchanged.

Export five candidates using beam width 64, token top-k 12, training-only phone
bigram LM weight 0.5, smoothing 1, zero token bonus, CTC upsampling two, and
batch eight. All 1,140 clip IDs, reference phone sequences and reference words
match the old evaluation exactly. Greedy PER reproduces the training evaluation.

Decode each candidate with `CompetitiveWordDecoder.evaluate_llama_visual`,
adapter `checkpoints/vallr-eos-20261010/adapter`, batch 16, deterministic generation,
maximum 256 new tokens and unchanged word normalization. Adapter SHA-256, base
model/revision, seed, generation settings, counts and normalization match
`checkpoints/vallr-visual-eos-20261011/metrics.json`. References contain 44,586
phones and 12,675 words. No candidate ranking or decoder training was added.

| Metric | Older hybrid + frozen decoder | Context checkpoint + same decoder |
| --- | ---: | ---: |
| Greedy exact-phone PER | 84.67% | 82.67% |
| Oracle exact-phone PER@5 | 78.14% | 79.96% |
| Top-ranked word WER | 99.59% | **95.48%** |
| Oracle word WER@5 | 96.44% | **92.53%** |
| Exactly correct sentences anywhere in five | 0 | 0 |
| Capped generations / 5,700 | 24 | 11 |

Top-ranked WER removes 521 edits (12,623 to 12,102), or 4.11 percentage points.
Oracle word WER removes 496 edits (12,224 to 11,728), or 3.91 points. Oracle
selection uses the reference and is not deployable accuracy. The worse phone
oracle but better word results show that these aggregate metrics need not move
together; no causal explanation for that difference has been established.

The change improves the downstream result, but 95.48% WER remains poor. No
deployment promotion follows. The checkpoint was selected on this repeatedly
inspected development set; a fresh held-out confirmation is still needed.
Next candidates for investigation are realistic visual-error decoder training,
candidate quality, and the later-step visual regression, not merely broader
family merging. No further experiments were launched.

Artifacts are under `checkpoints/context-word-eval-20261014`: `phone-eval.json`,
`phone-hypotheses.jsonl`, and `words/{manifest,metrics}.json`, with generated text
in `words/generations.jsonl`. Phone export took 264 seconds and word generation
196 seconds, excluding model loading; these are not controlled speed benchmarks.

Operational caveat: the initial thermal monitor could not open its log because
the output directory did not yet exist. GPU encoding completed before discovery;
the monitor was then attached to the verified running export process. The word
stage used a pre-created monitor directory and active thermal protection.
The monitor-startup failure must not be represented as continuous protection.
