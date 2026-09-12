"""Offline command-line inference for the VALLR visual model and Qwen decoder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from VALLR.Models.VALLR import VALLR
from peft import PeftModel
from transformers import (
    AutoModelForImageTextToText,
    AutoTokenizer,
    VideoMAEConfig,
    Wav2Vec2Config,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_VISUAL_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "VALLR.path"
DEFAULT_TEXT_MODEL = PROJECT_ROOT / "checkpoints" / "Qwen3.5-2B"

PHONEMES = (
    "<pad>",
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "ER",
    "EY",
    "F",
    "G",
    "HH",
    "IH",
    "IY",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OY",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UW",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
)
BLANK_ID = 0


def _existing_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"path does not exist: {path}")
    return path


def load_video(path: Path, num_frames: int = 16, image_size: int = 224) -> torch.Tensor:
    """Load evenly sampled RGB frames as ``(1, T, C, H, W)`` float32."""
    reader = VideoReader(str(path), ctx=cpu(0), num_threads=4)
    if len(reader) < num_frames:
        raise ValueError(
            f"video has {len(reader)} frames; at least {num_frames} are required"
        )

    indices = np.linspace(0, len(reader) - 1, num_frames).round().astype(np.int64)
    frames = torch.from_numpy(reader.get_batch(indices).asnumpy()).permute(0, 3, 1, 2)
    frames = F.interpolate(
        frames.float(),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    return frames.unsqueeze(0)


def load_visual_model(checkpoint: Path, device: torch.device) -> VALLR:
    """Construct the published V1 architecture and strictly load its weights."""
    model = VALLR(
        videomae_config=VideoMAEConfig(),
        wav2vec_config=Wav2Vec2Config(vocab_size=len(PHONEMES)),
        adapter_dim=256,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    # Transformers 5 stores VideoMAE Q/K/V biases on the Linear modules. The
    # published Transformers 4 checkpoint stores separate Q and V tensors and
    # implicitly uses a zero K bias. Migrate those names before strict loading.
    for layer_index in range(model.videomae.config.num_hidden_layers):
        prefix = f"videomae.encoder.layer.{layer_index}.attention.attention"
        query_bias = state.pop(f"{prefix}.q_bias")
        value_bias = state.pop(f"{prefix}.v_bias")
        state[f"{prefix}.query.bias"] = query_bias
        state[f"{prefix}.key.bias"] = torch.zeros_like(query_bias)
        state[f"{prefix}.value.bias"] = value_bias
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def ctc_decode(logits: torch.Tensor) -> list[str]:
    """Greedily collapse CTC logits into one phoneme sequence."""
    token_ids = logits.argmax(dim=-1)[0].tolist()
    decoded: list[str] = []
    previous = BLANK_ID
    for token_id in token_ids:
        if token_id != BLANK_ID and token_id != previous:
            decoded.append(PHONEMES[token_id])
        previous = token_id
    return decoded


def visual_to_phonemes(
    video: Path,
    checkpoint: Path = DEFAULT_VISUAL_CHECKPOINT,
) -> tuple[list[str], list[list[dict[str, float | str]]]]:
    """Return greedy phonemes and per-step ranked alternatives for a video."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_visual_model(checkpoint, device)
    frames = load_video(video).to(device)
    with torch.inference_mode():
        logits, _ = model(frames)

    probabilities = logits.softmax(dim=-1)[0].cpu()
    values, indices = probabilities.topk(k=min(5, probabilities.shape[-1]), dim=-1)
    ranked = [
        [
            {"phoneme": PHONEMES[token_id], "probability": round(float(probability), 6)}
            for probability, token_id in zip(step_values, step_indices, strict=True)
        ]
        for step_values, step_indices in zip(
            values.tolist(), indices.tolist(), strict=True
        )
    ]
    return ctc_decode(logits.cpu()), ranked


def phonemes_to_text(
    phonemes: list[str],
    model_path: Path = DEFAULT_TEXT_MODEL,
    max_new_tokens: int = 64,
    ranked_steps: list[list[dict[str, float | str]]] | None = None,
    adapter: Path | None = None,
    word_mode: bool = False,
) -> str:
    """Generate text from phonemes with the local Qwen3.5-2B base model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, use_fast=True
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
    model = model.eval()
    unit = "word" if word_mode else "sentence"
    prompt_lines = [
        f"Convert the visual speech phonemes into the most likely English {unit}.",
        f"Return only the {unit}.",
        "Top phoneme sequence: " + " ".join(phonemes),
    ]
    if word_mode:
        prompt_lines.append("Answer:")
    if ranked_steps:
        alternatives = []
        for step_number, candidates in enumerate(ranked_steps, start=1):
            choices = ", ".join(
                f"{candidate['phoneme']}={float(candidate['probability']):.6f}"
                for candidate in candidates
            )
            alternatives.append(f"{step_number}: {choices}")
        prompt_lines.extend(
            [
                "Ranked phoneme probabilities by CTC output step:",
                *alternatives,
                "Use sentence context to resolve visually ambiguous alternatives.",
            ]
        )
    prompt = "\n".join(prompt_lines)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    continuation = generated[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(continuation, skip_special_tokens=True).strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local VALLRITE models offline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    visual = subparsers.add_parser("visual", help="decode a video into phonemes")
    visual.add_argument("video", type=_existing_path)
    visual.add_argument(
        "--checkpoint", type=_existing_path, default=DEFAULT_VISUAL_CHECKPOINT
    )

    text = subparsers.add_parser("text", help="decode an ARPAbet sequence into text")
    text.add_argument("phonemes", nargs="+", help='for example: "DH AH K AE T"')
    text.add_argument("--model", type=_existing_path, default=DEFAULT_TEXT_MODEL)
    text.add_argument("--max-new-tokens", type=int, default=64)
    text.add_argument("--adapter", type=_existing_path)
    text.add_argument(
        "--word", action="store_true", help="use the CMUdict single-word prompt"
    )

    pipeline = subparsers.add_parser("pipeline", help="run video → phonemes → text")
    pipeline.add_argument("video", type=_existing_path)
    pipeline.add_argument(
        "--checkpoint", type=_existing_path, default=DEFAULT_VISUAL_CHECKPOINT
    )
    pipeline.add_argument("--model", type=_existing_path, default=DEFAULT_TEXT_MODEL)
    pipeline.add_argument("--max-new-tokens", type=int, default=64)
    pipeline.add_argument("--adapter", type=_existing_path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "visual":
        phonemes, ranked = visual_to_phonemes(args.video, args.checkpoint)
        print(json.dumps({"phonemes": phonemes, "ranked_steps": ranked}, indent=2))
        return 0

    if args.command == "text":
        print(
            phonemes_to_text(
                args.phonemes,
                args.model,
                args.max_new_tokens,
                adapter=args.adapter,
                word_mode=args.word,
            )
        )
        return 0

    phonemes, ranked = visual_to_phonemes(args.video, args.checkpoint)
    print(
        "Warning: Qwen3.5-2B is a base model and has not yet been fine-tuned on VALLR phonemes.",
        file=sys.stderr,
    )
    transcript = phonemes_to_text(
        phonemes,
        args.model,
        args.max_new_tokens,
        ranked_steps=ranked,
        adapter=args.adapter,
    )
    print(
        json.dumps(
            {"phonemes": phonemes, "ranked_steps": ranked, "transcript": transcript},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
