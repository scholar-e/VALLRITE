# Word decoder that reads phone posteriors, 2026-11-07

Goal: let the word decoder read the phone head's probabilities directly, rather
than only hard phone strings with posteriors used for rescoring or search.
Phones remain the only interface to the video.

## Training inputs

Training clips were seen by the Auto-AVSR encoder, so their phone posteriors are
far cleaner than test posteriors. Two steps make them test-like:

1. **Cross-fitting.** Two head-C-configuration heads were trained, each without
   one half of the source videos (`train_autoavsr_phone_head.py --folds 2
   --held-out-fold k`; test-select greedy PER 27.65% and 28.09%). Each training
   clip is scored by the head that never saw it. This alone leaves 17.1% PER:
   encoder memorization, not head memorization, dominates.
2. **Calibrated degradation** (`VisualPhoneme/export_phone_posteriors.py`).
   Gaussian noise of 1.3 × the per-dimension feature std is added to the
   encoder features. Statistics against head C on test-select:

| Input | Greedy PER | Blank rate | Mean max prob (emitted) | Entropy (nats/step) |
| --- | ---: | ---: | ---: | ---: |
| Test-select, head C (target) | 26.90% | 75.0% | 0.903 | 0.130 |
| Train, cross-fitted, clean | 17.14% | 69.9% | 0.948 | 0.068 |
| Train, cross-fitted, noise 1.0 | 20.11% | 70.1% | 0.921 | 0.124 |
| **Train, cross-fitted, noise 1.3 (used)** | **25.63%** | 70.3% | 0.892 | 0.180 |
| Train, cross-fitted, noise 1.0 + 2 time masks | 24.5% | 71.6% | 0.909 | 0.197 |

No setting matches every statistic; noise 1.3 matches error rate and
confidence, with somewhat higher entropy. Artifacts:
`checkpoints/posterior-decoder-20261107/posteriors-{train,test}`.

## Model

`posterior_decoder.py`: a 4.2M-parameter convolutional encoder maps 40-class
log posteriors (50 steps/s) to soft input embeddings at 12.5/s, inserted between
`</PHONEMES>` and `<TEXT>` of the usual prompt (greedy phone path). The output
projection starts at zero. Base: the matched-noise LoRA decoder. Greedy WER on
test-select every 400 steps selects checkpoints; stop after four evaluations
without improvement. Teacher-forced test-select loss is logged with real
posteriors and with the soft tokens zeroed.

## Runs

| Run | Trained | Test-select greedy WER by step (0 / 400 / 800 / 1200 / 1600) | Test-select token loss, posteriors vs zeroed |
| --- | --- | --- | --- |
| 1: encoder + LoRA (lr 5e-4 / 2e-5) | 1,600 steps, stopped | 40.49 / 43.03 / 41.83 / 41.29 / 42.49% | not logged |
| 2: encoder only, LoRA frozen (lr 1e-3) | 1,600 steps, stopped | 40.49 / 42.57 / 41.71 / 43.76 / 43.90% | 1.686 → 1.487–1.515 vs 1.686 |

The posteriors carry real information: with LoRA frozen they lower the loss of
reference sentences by about 0.19 nats per token, while the zeroed control is
unchanged. Greedy generation nevertheless gets worse. Diagnosis on test-select
(run 2, last step): insertions rise from 212 to 344 and output length from 4,899
to 5,039 words; outputs become more fluent and less faithful ("on his world tour
he graduated…" for "china is the world's greatest…"). The decoder learned to
trust posteriors in the way the degraded training posteriors allow, and that
does not transfer to real test posteriors when it writes freely.

## As a scorer

`posterior_decoder_rescoring.py` pools, per clip, the 20 decoded phone paths,
the fused decoder's hypotheses and five beam outputs of the posterior decoder
(about 18 unique sentences), and scores each by posterior-decoder log probability
+ CTC terms (weights fitted on test-select). Control: identical pipeline with the
soft tokens zeroed (plain decoder for both generation and scoring).

| System | test-select | test-report |
| --- | ---: | ---: |
| Head-C combined (previous selected) | 36.47% | 36.88% |
| Posterior decoder, beam top-1 | 41.65% | 43.21% |
| Control (zeroed), beam top-1 | 45.22% | 45.98% |
| Control (zeroed), pool scored + CTC | 36.47% | 36.59% |
| **Posterior decoder, pool scored + CTC** | **35.69%** | **36.39%** |
| Oracle of pool (not deployable) | 28.44% | 27.33% |

Test-report against the control: 47 clips better, 38 worse; paired bootstrap 95%
interval for the reduction −0.3 to +0.7 points. Against head-C combined:
−0.1 to +1.1 points. Most of the gain over the previous system comes from the
larger pool and decoder scoring, not from reading posteriors. Single seed.

## Conclusion

Reading posteriors directly improves the decoder's probability of the right
sentence but not, measurably, its word error. The bottleneck is training data:
encoder-seen clips need synthetic degradation, and the decoder learns how to use
those synthetic posteriors, not real ones. Real test-like posteriors require
clips the encoder never saw (another corpus, or a re-cropped LRS3 pretrain with
a face detector, which Auto-AVSR did train on, so only other corpora qualify),
or an encoder fine-tuned on a held-out split. Further LoRA/encoder tuning on the
current synthetic inputs is not recommended.

Operational note: `resize_token_embeddings` defaulted to mean resizing, a
full-vocabulary covariance on all CPU cores that pushed the CPU to the 95 °C
thermal kill threshold once. The adapters store their embeddings, so the new
loaders pass `mean_resizing=False`.
