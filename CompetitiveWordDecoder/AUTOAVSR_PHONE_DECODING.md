# Word decoding from Auto-AVSR phone heads, 2026-11-02

Phase 3 of the [training plan](../FUTURE_TRAINING_PLAN.md): does the phoneme
route (visual phones → LLM decoder) beat word-first decoding once the phones
come from the [Auto-AVSR phone head](../VisualPhoneme/AUTOAVSR_PHONE_HEAD.md)?
No training; the existing decoders were run unchanged.

## Setup

- Clips: the 661-clip `test-report` half of the official LRS3 test set (4,867
  words), never used for selection.
- Phones: five-best from the two-layer transformer head (27.99% beam PER).
- Decoder: clean EOS Llama-3.2-3B adapter (`checkpoints/vallr-eos-20261010/adapter`),
  the plan's robustness baseline, with the existing fixed probability ranking.
- Comparison: Auto-AVSR joint decoding (no LM) on the same clips.
- Artifacts: `checkpoints/phase3-autoavsr-phones-20261102/`.

## Results

| System on the same 661 clips | WER |
| --- | ---: |
| Auto-AVSR words, joint CTC/attention, no LM | **25.54%** |
| Auto-AVSR words, greedy CTC | 31.70% |
| Phones → decoder, top-ranked phone candidate | 54.74% |
| Phones → decoder, fixed probability ranking | 49.83% |
| Phones → decoder, oracle of five (not deployable) | 44.61% |
| Reference phones → decoder (ceiling) | 15.33% |
| Previous VALLRITE pipeline (1,140-clip dev set) | 93.29% |

Per clip, the phoneme route beats Auto-AVSR on 47 clips, ties on 140 and
loses on 474. The top-ranked output's errors are 1,799 substitutions, 490
deletions and 375 insertions. Outputs keep WikiText-style split contractions
("I 'm"); rejoining them changes top-1 WER only from 54.74% to 53.67%.

## Interpretation

1. **The phone upgrade carries through to words:** 93% → 50% WER.
2. **The decoder is not the ceiling.** On correct phones it reaches 15.33%,
   better than Auto-AVSR's 25.54%. The phoneme route has headroom if its input
   improves or the decoder learns to handle realistic errors.
3. **The decoder amplifies phone errors.** About 28% phone error becomes
   about 50% word error. It was trained on clean phones (and earlier adapters on
   ~70%-error phones), never on errors at this level. Its outputs are fluent but
   follow wrong phones ("early" → "hearty", "mosque" → "boss").

## Next levers that keep phones central

In expected order of payoff:

1. Adapt the decoder to realistic ~28%-error phones. Training-side phone
   predictions from Auto-AVSR features are too clean (17% PER on encoder-seen
   clips), so use cross-fitting on encoder-unseen data or synthetic noise
   matched to the measured confusion profile.
2. Pass phone probabilities or a lattice rather than five fixed sequences
   (timed-input adapter groundwork exists).
3. Improve phones: longer context-head training, encoder-unseen training data
   (pretrain needs a face detector), Auto-AVSR word CTC as an auxiliary loss, or
   word-derived phone pseudo-labels (18.80% PER).

## Matched-noise decoder adaptation, 2026-11-02

`matched_phone_noise.py` measures the transformer phone head's errors on the
**test-select** half (rank-one beam output, 28.51% PER): per-phone keep,
delete and substitution distributions, insertion counts and identities, and
the per-clip error-rate spread (sampled per sentence so errors stay bursty).
It applies them to LRS3 trainval transcripts with test transcripts excluded:
9,000 noisy rows (realized 28.79% PER) and 3,000 clean rows, plus 600
validation rows. Targets are normalized lowercase words.

`train_vallr_clean.py --prepared-rows` continued the clean EOS adapter for one
epoch (3,000 updates, batch 4, learning rate 2e-5, bf16). Final validation loss
0.789. Artifacts: `checkpoints/decoder-matched-noise-20261102/`,
`datasets/decoder-training/matched-noise-20261102/`.

| Decoder, same phones | Split | Top-1 WER | Ranked WER | Oracle@5 | Reference phones |
| --- | --- | ---: | ---: | ---: | ---: |
| Clean EOS (unchanged) | test-select | 54.62% | 50.24% | 45.58% | — |
| Matched noise | test-select | 44.06% | 40.85% | — | — |
| Clean EOS (unchanged) | test-report | 54.74% | 49.83% | 44.61% | 15.33% |
| **Matched noise (selected)** | **test-report** | **42.61%** | **40.13%** | 35.44% | **2.88%** |
| Auto-AVSR words, joint, no LM | test-report | — | 25.54% | — | — |

Selection used only test-select; test-report was evaluated once afterwards.

Caveats:

- **Domain and format are confounded with noise.** The old adapter learned
  WikiText formatting (digits, split contractions: 78 of its reference-phone
  outputs contain digits; rejoining contractions alone moves 15.33% to 13.77%).
  The new training text is LRS3-style lowercase words. A clean-only LRS3
  continuation (the plan's domain control) is needed to attribute the gain.
- Training text overlap with test-report is negligible: 3 of 380 test-report
  sentences with at least six words share any 6-gram with the training targets.
- One seed; no uncertainty interval yet.

Operational note: launched directly from a background shell, the trainer
stopped at every thermal pause, because the shell saw the guard's SIGSTOP
(exit 147) and its cleanup ended the trainer. Wrapping the command in
`timeout`, as the repository scripts do, keeps the pause transparent.
