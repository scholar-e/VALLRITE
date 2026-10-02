# VALLRITE

VALLRITE is an experimental visual speech-recognition system based on
[VALLR](https://github.com/MarshallT-99/VALLR). It converts silent face video
into phonetic evidence and then into text.

VALLRITE preserves uncertainty in the visual predictions. Sounds such as
`/p/`, `/b/`, and `/m/` can look nearly identical on the lips, so the pipeline
retains ranked per-step phoneme probabilities for downstream decoding.

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

See the [VPA specification](VisualPhoneticAlphabet/README.md) for details.

## Status

- The published VALLR visual checkpoint loads and runs on CPU.
- Qwen3.5-2B is installed as an experimental phoneme-to-text reranker.
- The pipeline emits a greedy sequence plus ranked per-step probabilities.
- A real AVSpeech clip has passed end-to-end inference.
- Full AVSpeech train/test manifests are stored locally: 2,805,118 segments.
- A reusable CMUdict LoRA trainer and an initial 1,024-entry adapter are present.

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
`- VALLR/                      # original upstream project
   |- Data/
   |- Models/
   |- config.py
   |- face_cropper.py
   `- main.py
```

## Attribution

VALLRITE extends the original VALLR project by Marshall Thomas, Edward Fish,
and Richard Bowden. Upstream code, datasets, and checkpoints remain subject to
their respective licenses and access terms.
