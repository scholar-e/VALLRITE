# LLM memorization probe on LRS3 test, 2026-11-26

TED transcripts are public text, so an LLM used to decode or correct LRS3
might recall test sentences instead of reading the video.
`memorization_probe.py` cuts each test-select sentence of 8+ words in half and
lets the model continue the first half greedily from raw text (no chat
template). Control: the same number of Seamless Interaction sentences
(recent conversations, not public text; more formulaic than TED, so a
conservative baseline). 253 sentences per set (test-select has 253 of 8+
words). `checkpoints/memorization-probe-20261126/`.

| Model | LRS3 exact rest | LRS3 next 4 words | Seamless next 4 words | LRS3 / Seamless words matched |
| --- | ---: | ---: | ---: | ---: |
| Llama-3.2-3B-Instruct (our decoder's base) | 0.0% | 1.2% | 0.0% | 6.9% / 4.9% |
| Qwen3.6-35B-A3B (GGUF, LM Studio) | 0.4% | 1.2% | 0.0% | 6.7% / 4.2% |
| DeepSeek-V4-Pro-distilled Qwen3.5-9B (GGUF) | 0.4% | 0.8% | 0.0% | 7.1% / 4.7% |

Every 4-word hit is a stock phrase ("... the last | thing you want to do",
"we're not going to | be able to do that", "shown | to be effective"); the
single full match per GGUF model is the same "be able to do that" type.
No sign that any of the three models recalls LRS3 test sentences from a
half-sentence prefix. Limits: LRS3 clips are short (half-sentences of ~5
words give little context), the sample is small, and a corrector sees whole
noisy candidates, a stronger cue than a prefix; a corrector's gain should
still be compared with the same step on held-out Seamless clips.
