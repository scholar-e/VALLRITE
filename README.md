# VALLRITE

VALLRITE is an experimental visual speech-recognition system based on
[VALLR](https://github.com/MarshallT-99/VALLR). It converts silent face video
into phonetic evidence and then into text.

The main change is to preserve uncertainty. Sounds such as `/p/`, `/b/`, and
`/m/` can look nearly identical on the lips, so VALLRITE is designed to pass a
ranked phoneme lattice—not one prematurely selected phoneme string—into a
probabilistic lexical and language decoder.

```text
                                      training only
                               audio -> teacher labels
                                         |
                                         v
video -> visual encoder -> VPA/phoneme lattice -> lexical decoder -> reranker -> text
```

VPA is the proposed Visual Phonetic Alphabet: a timed representation of
camera-observable articulation, confidence, and compatible IPA/ARPAbet
phonemes. It complements phonetic alphabets rather than claiming that voicing,
nasality, or hidden tongue positions are visible.

## Status

- The published VALLR visual checkpoint loads and runs on CPU.
- Qwen3.5-2B is installed as an experimental phoneme-to-text reranker.
- The pipeline emits a greedy sequence plus ranked per-step probabilities.
- A real AVSpeech clip has passed end-to-end inference.
- Full AVSpeech train/test manifests are stored locally: 2,805,118 segments.
- A reusable CMUdict LoRA trainer and an initial 1,024-entry adapter are present.
- Proper CTC lattice decoding, VPA extraction, audio-teacher labels, WFST
  decoding, evaluation, and mobile export remain planned work.

The current Qwen output is not accurate: the base model has not yet been
fine-tuned on real VALLR lattices. A successful execution is presently a smoke
test, not an accuracy result.

## Run

```bash
source .venv-qwen/bin/activate

# Video -> phonemes and ranked probabilities
python vallrite.py visual /path/to/video.mp4

# Phonemes -> text
python vallrite.py text DH AH K AE T

# Complete experimental pipeline
python vallrite.py pipeline /path/to/video.mp4

# Single dictionary word through the trained LoRA adapter
python vallrite.py text G L AE S P ER \
  --word \
  --adapter checkpoints/qwen-phoneme-dictionary-lora
```

## Prepare AVSpeech samples

```bash
python tools/prepare_avspeech.py \
  datasets/avspeech/manifests/avspeech_test.csv \
  --output-dir datasets/avspeech/clips \
  --limit 1
```

Dataset media and model checkpoints are local and ignored by Git. AVSpeech
manifests reference YouTube sources, so removed, private, and restricted videos
are skipped and recorded in `index.jsonl`.

## Project plan

The implementation is organized around six workstreams:

1. Reproduce and measure the original VALLR baseline.
2. Build proper CTC beam/lattice decoding with calibrated probabilities.
3. Develop and validate the Visual Phonetic Alphabet.
4. Improve training labels with synchronized audio and scale permitted data.
5. Compare WFST, compact neural, and Qwen-based language decoding.
6. Distill, quantize, and benchmark the selected system on a phone.

The complete engineering and research blueprint is in
[ai-readme.md](ai-readme.md). The focused VPA specification is in
[VisualPhoneticAlphabet/README.md](VisualPhoneticAlphabet/README.md).

## Repository layout

```text
VALLRITE/
|- AudioPhonemeLabeler/
|- PhonemeDecoder/
|- VisualPhoneme/
|- VisualPhoneticAlphabet/
|- tools/
|- checkpoints/                # local, ignored
|- datasets/                   # local, ignored
|- vallrite.py
|- README.md
|- ai-readme.md
`- VALLR/                      # original upstream project
   |- Data/
   |- Models/
   |- config.py
   |- face_cropper.py
   `- main.py
```

## Success criteria

VALLRITE must beat a reproduced VALLR baseline on the same held-out split,
demonstrate that calibrated uncertainty improves WER over top-1 phonemes, show
that audio teaching helps video-only inference, and run offline within measured
memory and latency limits on a named reference phone.

## Attribution

VALLRITE extends the original VALLR project by Marshall Thomas, Edward Fish,
and Richard Bowden. Upstream code, datasets, and checkpoints remain subject to
their respective licenses and access terms.
