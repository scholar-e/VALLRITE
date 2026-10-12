# Soft audio-to-visual distillation

The pilot uses the local `facebook/wav2vec2-lv-60-espeak-cv-ft` audio teacher
and continues the existing large-image visual-groups-rest checkpoint. Audio is
available only during training. Inference still consumes video alone.

IPA token probabilities are summed into the checkpoint's visual phoneme
families. CTC blank and unmapped audio tokens are excluded; remaining speech
mass weights the KL loss, with a minimum mass of 0.2. Acoustic blank is never
treated as REST. Existing forced inter-word gaps supervise REST separately.
Audio receptive-field centers are mapped to video frame centers using the
teacher's convolution settings. Frame KL is added with weight 0.2 to transcript
CTC plus aligned frame cross entropy at weight 0.1.

Both arms start from the same checkpoint and use the same 128 accepted aligned
training chunks in the same order, without augmentation. The control sets KL
weight to zero. Each arm runs 100 optimizer steps at learning rate 1e-5, with
checkpoints every 25 steps. Validation source videos are disjoint from training.
These training clips may already have been seen by the starting visual model.

## Initial development result, 2026-10-02

| Model | REST-stripped group PER | REST-inclusive group PER |
| --- | ---: | ---: |
| Starting checkpoint | 76.00% | 75.57% |
| CTC/frame control | 78.46% | 76.71% |
| Audio-distilled student | 78.29% | 76.45% |

Evaluation covers the same 127 usable validation chunks and 3,450 speech-family
tokens. This is a small development pilot, not exact-phone PER or a full-corpus
benchmark. The improvement over control is six group-token errors; both arms
regress from the starting checkpoint. Neither pilot should replace that model.

The teacher's mapped speech mass averaged 0.267 across 6,210 training frames;
29.65% exceeded the confidence threshold. Conditional family distributions
normalized correctly, with no blank or REST probability assigned by the audio
projection. No gain from scaling or iterative distillation is established.

Reports and cached teacher probabilities are in
`checkpoints/visual-audio-distillation-pilot-20261002/`. The next comparison
should use a larger representative training sample with shuffled batches,
retain the paired control, and check acoustic target quality before increasing
training duration. The existing REST checkpoint remains preferred.

```bash
tools/with-vpa-gpu .venv-vpa-gpu/bin/python -m VisualPhoneme.distill_audio \
  --checkpoint checkpoints/lrs3-rest-transcript-audio-20260929/best.pt \
  --teacher-labels datasets/lrs3/audio-teacher-full-forced \
  --output checkpoints/visual-audio-distillation-pilot-20261002 \
  --train-limit 128 --validation-limit 128 --steps 100 --max-minutes 10
```

The thermal watchdog covers audio export and visual training. `PAUSE` stops
training at a step boundary after saving state; remove it and repeat the command
with `--resume`. Cached audio targets are reused. The time limit bounds both
training arms together after audio preparation; preparation is separately
bounded by the number of selected training chunks.
