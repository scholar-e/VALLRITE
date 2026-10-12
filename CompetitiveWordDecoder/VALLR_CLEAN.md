# Original-recipe clean phoneme baseline

Scope: original Llama/WikiText training and clean held-out reconstruction only.
No visual hypotheses, phoneme corruption, or matched visual WER comparison.

`train_vallr_clean.py` follows upstream VALLR commit
`db932ca83ff8b2b858eb6a4c78d02b5e3805c299`, `Models/Llama.py`: Llama-3.2-3B-Instruct,
CMUdict first pronunciations without stress, the four special tags, target-only
loss, 512 tokens, rank16/alpha32/dropout0.05 attention+MLP LoRA, 10 epochs,
learning rate 2e-4, weight decay .01, warmup .03, batch1/accumulation4, FP16.
Existing `.venv-qwen` supplies torch, transformers, datasets, peft and pronouncing;
the environment name does not imply this run uses Qwen. No new dependencies.

Documented operational differences: gradient checkpointing, deterministic seed,
fixed model/dataset revisions, deduplicated normalized text across all splits,
250-step resumable checkpoints (latest two retained), and persistent progress logs.
Epoch validation measures loss; the final model is evaluated on the separate test
split, never used for training or checkpoint selection.

Original quirks retained: long prompts truncate to 508 tokens, possibly dropping
the final tags and most target supervision; no EOS is appended to training
targets; unknown words become `UNK(word)` and leak spelling. Clean test WER is
reported both overall and on the dictionary-only subset. It uses complete pairs
fitting 512 tokens with targets below 256 tokens, greedy generation capped at256,
and lowercase ASCII word/apostrophe normalization. Excluded rows and capped
outputs are counted. These are text-reconstruction WERs, not lipreading WER or
oracle WER@5. WikiText may also be present in the base model's pretraining.

## Run and monitor

The 2026-10-10 detached pipeline is `vallrite-vallr-clean-20261010.service`.
It downloads the base model, runs a separate real-model two-step smoke test,
then starts full training only if the smoke command succeeds. Follow all phases:

```bash
journalctl --user -u vallrite-vallr-clean-20261010.service -f
```

Do not launch the manual command below while that service is active.

```bash
tools/with-vpa-gpu .venv-qwen/bin/python -m CompetitiveWordDecoder.train_vallr_clean \
  --output checkpoints/vallr-clean-20261010
tail -f checkpoints/vallr-clean-20261010/train.log
```

Run with host GPU access. `--smoke` uses eight rows per split and two steps; use
a separate output directory for it. `--max-minutes N` optionally stops and saves
at an optimizer-step boundary. The full experiment is epoch-bounded, not subject
to the historical 30-minute pilot limit.

Pause with the existing root helper and this explicit output directory:

```bash
./pause-training checkpoints/vallr-clean-20261010
```

The trainer also recognizes repository-root `PAUSE` and SIGINT/SIGTERM. Remove
the intended pause marker before resuming (a root marker may pause other runs).
Resume with the same command plus `--resume`; optimizer/scheduler/RNG state is
restored. The older root `resume-training` helper is VisualPhoneme-specific and
does not apply to this trainer. `--evaluate-only` reloads the final adapter and
restarts clean test generation if evaluation was interrupted. Interrupted
evaluation does not publish a completed metrics file.

Outputs: manifest.json, train.log, checkpoint-*/ (resumable), adapter/,
training-metrics.json, clean-test-predictions.jsonl, clean-test-metrics.json.

## EOS continuation (2026-10-10)

The original run was stopped cleanly at step 15,398 (epoch 2.8978). Its adapter,
resumable checkpoint, logs and PAUSE marker remain under
`checkpoints/vallr-clean-20261010`. The continuation is a separate experiment,
not a claim to reproduce upstream exactly.

DeepSeek drafted the change; supervisor review found syntax errors and an
incorrectly masked EOS label. These were corrected before execution. Nine unit
tests cover the original encoding and complete-pair EOS encoding, including EOS
equal to PAD. Actual GPU save/resume is tested in a separate smoke directory.

The new mode appends tokenizer EOS (`<|eot_id|>`, ID 128009) to every target and
supervises it, while masking padding even though PAD has the same ID. Complete
prompt + target + EOS must fit 512 tokens; longer pairs are skipped, not cut off.
Thus both stopping supervision and the length-filtered training distribution
differ from the baseline. Generation stops on the supervised EOS token.

```bash
tools/with-vpa-gpu .venv-qwen/bin/python -m CompetitiveWordDecoder.train_vallr_clean \
  --output checkpoints/vallr-eos-20261010 \
  --end-of-answer --initial-adapter checkpoints/vallr-clean-20261010/adapter \
  --epochs 1 --learning-rate 2e-5 --test-limit 100 --max-minutes 60
```

This loads the saved adapter weights with a fresh lower-learning-rate optimizer
and scheduler. `--resume` instead restores the continuation's optimizer state;
repeat its other arguments unchanged. The initial adapter hash is recorded and
checked. The 60-minute training limit saves and stops if reached; it does not
cap final evaluation. There is no new thermal watchdog.

On completion, clean WER uses the same seed-42 sample of 100 eligible held-out
test examples as the step-13,000 diagnostic. `test-sample.json` records the
selected targets. It remains clean text reconstruction, not visual-input WER.
Use `./pause-training checkpoints/vallr-eos-20261010` to request a safe pause.
The old root `resume-training` helper still does not apply to this trainer.
