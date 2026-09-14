"""Predict a phoneme sequence from a silent GRID-style face video."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np
import torch

from VisualPhoneme.data import (CROPS, PHONEMES, ctc_prefix_beam_search,
                                      decode_video, greedy_decode, landmark_feature_dimensions,
                                      transform_landmarks)
from VisualPhoneme.model import (CompactFusionVisualPhoneme,
                                       CompactGatedFusionVisualPhoneme,
                                       CompactLandmarkPhoneme,
                                       CompactTongueGatedFusionVisualPhoneme,
                                       CompactVisualPhoneme,
                                       LargeGatedFusionVisualPhoneme)
from VisualPhoneme.train import upsample_ctc_logits
from VisualPhoneme.visemes import (VISUAL_GROUPS, VISUAL_PHONE_GROUPS,
                                   visual_group_alternatives,
                                   visual_phone_alternatives)

LOGGER = logging.getLogger("visual_phoneme.predict")


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.handlers.clear()
    file_handler = logging.FileHandler(log_path, delay=False)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(console_handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/visual-phoneme-mouth/best.pt"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--landmarks-npz", type=Path,
                        help="VPA coordinate cache required by a fusion checkpoint")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--emissions-output", type=Path,
                        help="export full 40-class probabilities for word decoding")
    parser.add_argument("--word-lexicon", type=Path,
                        help="decode words using a PhonemeDecoder JSON lexicon")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--beam-token-top-k", type=int, default=8)
    parser.add_argument("--lm-weight", type=float,
                        help="override checkpoint phone-bigram weight")
    args = parser.parse_args()
    if not args.video.is_file() or not args.checkpoint.is_file():
        parser.error("video and checkpoint must exist")
    if args.top_n < 1 or args.beam_width < args.top_n or args.beam_token_top_k < 1:
        parser.error("require --beam-width >= --top-n >= 1 and positive --beam-token-top-k")
    output = args.output or args.video.with_suffix(".visual-phonemes.json")
    configure_logging(output.with_suffix(".log"))
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    phones = tuple(checkpoint["phones"])
    target_inventory = checkpoint.get("target_inventory", "phones")
    expected_labels = PHONEMES if target_inventory == "phones" else VISUAL_GROUPS
    if phones != expected_labels:
        raise ValueError("checkpoint target vocabulary does not match this code")
    expand = (visual_phone_alternatives if target_inventory == "phones"
              else visual_group_alternatives)
    if target_inventory == "visual-groups" and (args.emissions_output or args.word_lexicon):
        parser.error("group checkpoints cannot use the 39-phone emission/lexicon decoder")
    crop_name = checkpoint["crop"]
    architecture = checkpoint.get("architecture", "image")
    if architecture in {"coordinates", "fusion", "gated-fusion", "tongue-gated-fusion",
                        "large-gated-fusion"} and (
            args.landmarks_npz is None or not args.landmarks_npz.is_file()):
        parser.error("a coordinate checkpoint requires an existing --landmarks-npz cache")
    model_classes = {"image": CompactVisualPhoneme, "coordinates": CompactLandmarkPhoneme,
                     "fusion": CompactFusionVisualPhoneme,
                     "gated-fusion": CompactGatedFusionVisualPhoneme,
                     "tongue-gated-fusion": CompactTongueGatedFusionVisualPhoneme,
                     "large-gated-fusion": LargeGatedFusionVisualPhoneme}
    model_args = {"classes": len(phones) + 1}
    if architecture in {"coordinates", "fusion", "gated-fusion", "tongue-gated-fusion",
                        "large-gated-fusion"}:
        model_args["landmark_points"] = int(checkpoint["landmark_points"])
        model_args["coordinate_dimensions"] = int(checkpoint.get("coordinate_dimensions", 2))
    if architecture in {"coordinates", "gated-fusion", "tongue-gated-fusion",
                        "large-gated-fusion"}:
        model_args["landmark_bottleneck"] = checkpoint.get("landmark_bottleneck")
    if architecture in {"gated-fusion", "tongue-gated-fusion", "large-gated-fusion"}:
        model_args["image_gate_probability"] = checkpoint.get(
            "image_gate_initial_probability", 0.002472623
        )
    if architecture == "tongue-gated-fusion":
        model_args["inner_mouth_gate_probability"] = checkpoint.get(
            "inner_mouth_gate_initial_probability", 0.05
        )
    model = model_classes[architecture](**model_args).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    video = (decode_video(args.video, CROPS[crop_name], int(checkpoint["image_size"]))
             if architecture in {"image", "fusion", "gated-fusion",
                                 "tongue-gated-fusion", "large-gated-fusion"} else None)
    landmarks = landmark_mask = None
    if architecture in {"coordinates", "fusion", "gated-fusion", "tongue-gated-fusion",
                        "large-gated-fusion"}:
        with np.load(args.landmarks_npz) as cached:
            landmarks = torch.from_numpy(cached["coordinates"].astype(np.float32))
        frames = min(len(video), len(landmarks)) if video is not None else len(landmarks)
        if video is not None:
            video = video[:frames]
        landmarks = landmarks[:frames]
        coordinate_mode = checkpoint.get("coordinate_mode", "eye-normalized")
        coordinate_features = checkpoint.get("coordinate_features", "position")
        expected_dimensions = landmark_feature_dimensions(coordinate_features)
        if expected_dimensions != int(checkpoint.get("coordinate_dimensions", 2)):
            raise ValueError("checkpoint coordinate feature dimensions are inconsistent")
        landmarks, landmark_mask = transform_landmarks(
            landmarks, coordinate_mode, coordinate_features)
    input_frames = len(video) if video is not None else len(landmarks)
    with torch.inference_mode():
        if architecture in {"fusion", "gated-fusion", "tongue-gated-fusion",
                            "large-gated-fusion"}:
            logits = model(video.unsqueeze(0).to(device), landmarks.unsqueeze(0).to(device),
                           landmark_mask.unsqueeze(0).to(device))
        elif architecture == "coordinates":
            logits = model(landmarks.unsqueeze(0).to(device),
                           landmark_mask.unsqueeze(0).to(device))
        else:
            logits = model(video.unsqueeze(0).to(device))
        logits, output_lengths = upsample_ctc_logits(
            logits, torch.tensor([input_frames]),
            int(checkpoint.get("ctc_upsample_factor", 1)),
        )
        probabilities = logits.softmax(-1)
    ids = greedy_decode(logits, output_lengths)[0]
    prediction = [phones[index - 1] for index in ids]
    decoding = checkpoint.get("decoding", {})
    lm_weight = args.lm_weight if args.lm_weight is not None else decoding.get("lm_weight", 0.0)
    transition_log_probs = checkpoint.get("bigram_log_probs")
    ranked = ctc_prefix_beam_search(logits[:, 0].log_softmax(-1),
                                    args.beam_width, args.top_n,
                                    args.beam_token_top_k,
                                    transition_log_probs, lm_weight)
    normalization = torch.logsumexp(torch.tensor([score for _, score in ranked]), dim=0)
    candidates = [
        {"phonemes": [phones[index - 1] for index in sequence],
         "visual_phone_alternatives": expand(
             [phones[index - 1] for index in sequence]),
         "combined_log_score": score,
         "relative_probability_within_top_n": float(
             torch.exp(torch.tensor(score) - normalization))}
        for sequence, score in ranked
    ]
    result = {
        "video": str(args.video.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint["epoch"],
        "crop": crop_name,
        "image_size": checkpoint["image_size"],
        "architecture": architecture,
        "coordinate_mode": checkpoint.get("coordinate_mode", "eye-normalized"),
        "frames": input_frames,
        "ctc_steps": int(output_lengths[0]),
        "phonemes": prediction,
        "target_inventory": target_inventory,
        "visual_phone_alternatives": expand(prediction),
        "visual_phone_groups": {
            name: list(members) for name, members in VISUAL_PHONE_GROUPS.items()
        },
        "top_n_candidates": candidates,
        "beam_width": args.beam_width,
        "beam_token_top_k": args.beam_token_top_k,
        "lm_weight": lm_weight,
        "mean_blank_probability": float(probabilities[:, 0, 0].mean()),
        "decoder": "greedy CTC",
    }
    if args.emissions_output or args.word_lexicon:
        from PhonemeDecoder.decoder import (Lexicon, StreamingDecoder,
                                                  VOCABULARY, probabilities_from_logits)
        rows = probabilities_from_logits(logits, phones, int(output_lengths[0]))
        if args.emissions_output:
            args.emissions_output.write_text(json.dumps({
                "format": "visual-phoneme-emissions-0.1",
                "vocabulary": list(VOCABULARY), "probabilities": rows,
                "valid_steps": len(rows), "checkpoint": str(args.checkpoint.resolve()),
                "architecture": architecture,
            }, allow_nan=False) + "\n")
        if args.word_lexicon:
            decoder = StreamingDecoder(Lexicon(json.loads(args.word_lexicon.read_text())),
                                       beam_width=args.beam_width)
            decoder.accept(rows)
            result["word_decoding"] = decoder.result(args.top_n)
    output.write_text(json.dumps(result, indent=2) + "\n")
    LOGGER.info("phonemes=%s", " ".join(prediction) if prediction else "(empty)")
    LOGGER.info("result=%s", output)
    for handler in LOGGER.handlers:
        handler.flush()


if __name__ == "__main__":
    main()
