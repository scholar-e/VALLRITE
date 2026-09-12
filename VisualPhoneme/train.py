"""Train and evaluate the compact visual-phoneme CTC model."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from VisualPhoneme.data import (GridClips, PHONEMES, collate_clips,
                                collate_fusion_clips, collate_landmark_clips,
                                ctc_prefix_beam_search, fit_bigram_log_probs,
                                greedy_decode, landmark_feature_dimensions,
                                phoneme_target)
from VisualPhoneme.model import (CompactFusionVisualPhoneme,
                                 CompactGatedFusionVisualPhoneme,
                                 CompactLandmarkPhoneme,
                                 CompactVisualPhoneme)
LOGGER = logging.getLogger("visual_phoneme")


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.handlers.clear()
    file_handler = logging.FileHandler(output_dir / "train.log", delay=False)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(console_handler)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def edit_totals(reference: list[int], hypothesis: list[int]) -> tuple[int, int]:
    previous = list(range(len(hypothesis) + 1))
    for index, reference_token in enumerate(reference, 1):
        current = [index]
        for offset, hypothesis_token in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[offset] + 1,
                               previous[offset - 1] + (reference_token != hypothesis_token)))
        previous = current
    return previous[-1], len(reference)


def run_epoch(model, loader, loss_fn, device, architecture, optimizer=None, deadline=None,
              top_n=1, beam_width=1, beam_token_top_k=None,
              transition_log_probs=None, lm_weight=0.0,
              image_modality_dropout=0.0, coordinate_modality_dropout=0.0):
    training = optimizer is not None
    model.train(training)
    total_loss = total_errors = total_phones = clips = 0
    oracle_errors = exact_hits = 0
    started = time.perf_counter()
    for step, batch in enumerate(loader, 1):
        if architecture in {"fusion", "gated-fusion"}:
            video, landmarks, landmark_mask, targets, lengths, target_lengths, _ = batch
            landmarks = landmarks.to(device, non_blocking=True)
            landmark_mask = landmark_mask.to(device, non_blocking=True)
            video = video.to(device, non_blocking=True)
        elif architecture == "coordinates":
            landmarks, landmark_mask, targets, lengths, target_lengths, _ = batch
            landmarks = landmarks.to(device, non_blocking=True)
            landmark_mask = landmark_mask.to(device, non_blocking=True)
            video = None
        else:
            video, targets, lengths, target_lengths, _ = batch
            video = video.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training and architecture in {"fusion", "gated-fusion"}:
            if image_modality_dropout:
                dropped = torch.rand(len(video), device=device) < image_modality_dropout
                video[dropped] = 0
            if coordinate_modality_dropout:
                dropped = torch.rand(len(landmarks), device=device) < coordinate_modality_dropout
                landmarks[dropped] = 0
                landmark_mask[dropped] = False
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            if architecture in {"fusion", "gated-fusion"}:
                logits = model(video, landmarks, landmark_mask)
            elif architecture == "coordinates":
                logits = model(landmarks, landmark_mask)
            else:
                logits = model(video)
            loss = loss_fn(logits.log_softmax(-1), targets, lengths, target_lengths)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        hypotheses = greedy_decode(logits.detach(), lengths)
        beam_candidates = None
        if not training and top_n > 1:
            log_probabilities = logits.detach().log_softmax(-1).transpose(0, 1).cpu()
            beam_candidates = [
                ctc_prefix_beam_search(scores[:int(length)], beam_width, top_n,
                                       beam_token_top_k, transition_log_probs, lm_weight)
                for scores, length in zip(log_probabilities, lengths)
            ]
        offset = 0
        for item_index, (hypothesis, target_length) in enumerate(zip(hypotheses, target_lengths)):
            reference = targets[offset:offset + int(target_length)].tolist()
            offset += int(target_length)
            errors, phones = edit_totals(reference, hypothesis)
            total_errors += errors
            total_phones += phones
            if beam_candidates is not None:
                sequences = [sequence for sequence, _ in beam_candidates[item_index]]
                exact_hits += int(reference in sequences)
                oracle_errors += min(edit_totals(reference, sequence)[0]
                                     for sequence in sequences)
        batch_clips = len(target_lengths)
        clips += batch_clips
        total_loss += float(loss.detach()) * batch_clips
        if training and (step == 1 or step % 100 == 0):
            LOGGER.info("step=%d clips=%d loss=%.4f running_PER=%.2f%%", step, clips,
                        total_loss / clips, 100 * total_errors / total_phones)
        if deadline is not None and time.monotonic() >= deadline:
            LOGGER.warning("training time budget reached after %d clips", clips)
            break
    result = {"loss": total_loss / clips, "per": total_errors / total_phones,
              "errors": total_errors, "phones": total_phones, "clips": clips,
              "seconds": time.perf_counter() - started,
              "complete": clips == len(loader.dataset)}
    if not training and top_n > 1:
        result.update({"top_n": top_n, "beam_width": beam_width,
                       "beam_token_top_k": beam_token_top_k,
                       "top_n_exact_hits": exact_hits,
                       "top_n_exact_accuracy": exact_hits / clips,
                       "oracle_errors_at_n": oracle_errors,
                       "oracle_per_at_n": oracle_errors / total_phones})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/grid-pilot"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/visual-phoneme-mouth"))
    parser.add_argument("--crop", choices=("face", "mouth", "full"), default="mouth")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--max-minutes", type=float,
                        help="stop cleanly before starting validation once this wall-time budget is reached")
    parser.add_argument("--landmark-fusion", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--architecture", choices=("image", "coordinates", "fusion",
                                                    "gated-fusion"),
                        default="image", help="input modality used by this ablation")
    parser.add_argument("--horizontal-flip", action=argparse.BooleanOptionalAction, default=None,
                        help="randomly reflect training images (default: image model only)")
    parser.add_argument("--coordinate-mode",
                        choices=("eye-normalized", "clip-centered", "constant"),
                        default="eye-normalized",
                        help="landmark representation; constant is a duration/grammar control")
    parser.add_argument("--coordinate-features", choices=("position", "motion"),
                        default="position")
    parser.add_argument("--landmark-jitter", type=float, default=0.0)
    parser.add_argument("--point-dropout", type=float, default=0.0)
    parser.add_argument("--frame-span-dropout", type=float, default=0.0)
    parser.add_argument("--max-frame-span", type=int, default=5)
    parser.add_argument("--landmark-bottleneck", type=int, default=0)
    parser.add_argument("--image-modality-dropout", type=float, default=0.0)
    parser.add_argument("--coordinate-modality-dropout", type=float, default=0.0)
    parser.add_argument("--initialize-coordinate-checkpoint", type=Path,
                        help="initialize a gated fusion model's shared coordinate path")
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--frame-cache-dir", type=Path,
                        help="optional persistent cache of decoded uint8 model input frames")
    parser.add_argument("--top-n", type=int, default=1,
                        help="number of validation CTC sequences to score")
    parser.add_argument("--beam-width", type=int, default=16,
                        help="prefix-beam width when top-n is greater than one")
    parser.add_argument("--beam-token-top-k", type=int, default=8,
                        help="highest-scoring frame tokens expanded by the beam")
    parser.add_argument("--selection-metric",
                        choices=("greedy-per", "oracle-per-at-n", "top-n-exact"),
                        default="greedy-per")
    parser.add_argument("--lm-weight", type=float, default=0.0,
                        help="smoothed phone-bigram shallow-fusion weight")
    parser.add_argument("--lm-smoothing", type=float, default=1.0)
    args = parser.parse_args()
    if (args.image_size < 32 or args.epochs < 1 or args.batch_size < 1 or args.workers < 0
            or (args.max_minutes is not None and args.max_minutes <= 0)):
        parser.error("invalid image size, epoch, batch, or worker count")
    if args.top_n < 1 or args.beam_width < args.top_n or args.beam_token_top_k < 1:
        parser.error("require --beam-width >= --top-n >= 1 and positive --beam-token-top-k")
    if args.selection_metric != "greedy-per" and args.top_n == 1:
        parser.error("top-n checkpoint selection requires --top-n greater than one")
    probabilities = (args.point_dropout, args.frame_span_dropout,
                     args.image_modality_dropout, args.coordinate_modality_dropout)
    if (args.landmark_jitter < 0 or any(not 0 <= value < 1 for value in probabilities)
            or args.max_frame_span < 1 or args.landmark_bottleneck < 0
            or args.weight_decay < 0 or args.lm_weight < 0 or args.lm_smoothing <= 0):
        parser.error("invalid regularization, bottleneck, weight-decay, or LM setting")

    if args.landmark_fusion:
        if args.architecture != "image":
            parser.error("--landmark-fusion cannot be combined with --architecture")
        args.architecture = "fusion"
    if args.horizontal_flip is None:
        args.horizontal_flip = args.architecture == "image"
    if args.horizontal_flip and args.architecture != "image":
        parser.error("horizontal reflection requires an image-only model until landmark permutation exists")
    if args.coordinate_mode != "eye-normalized" and args.architecture == "image":
        parser.error("--coordinate-mode applies only to coordinates or fusion")

    configure_logging(args.output_dir)
    deadline = time.monotonic() + args.max_minutes * 60 if args.max_minutes is not None else None
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    include_landmarks = args.architecture in {"coordinates", "fusion", "gated-fusion"}
    include_video = args.architecture in {"image", "fusion", "gated-fusion"}
    train = GridClips(args.data_root, "train", args.image_size, args.crop, args.train_limit,
                      True, include_landmarks, include_video, args.horizontal_flip,
                      args.coordinate_mode, args.frame_cache_dir, args.coordinate_features,
                      args.landmark_jitter, args.point_dropout, args.frame_span_dropout,
                      args.max_frame_span)
    validation = GridClips(args.data_root, "validation", args.image_size, args.crop,
                           args.validation_limit, False, include_landmarks, include_video, False,
                           args.coordinate_mode, args.frame_cache_dir, args.coordinate_features)
    collate_functions = {"image": collate_clips, "coordinates": collate_landmark_clips,
                         "fusion": collate_fusion_clips,
                         "gated-fusion": collate_fusion_clips}
    loader_args = {"batch_size": args.batch_size, "num_workers": args.workers,
                   "collate_fn": collate_functions[args.architecture],
                   "pin_memory": device.type == "cuda",
                   "persistent_workers": args.workers > 0}
    train_loader = DataLoader(train, shuffle=True, **loader_args)
    validation_loader = DataLoader(validation, shuffle=False, **loader_args)
    model_classes = {"image": CompactVisualPhoneme, "coordinates": CompactLandmarkPhoneme,
                     "fusion": CompactFusionVisualPhoneme,
                     "gated-fusion": CompactGatedFusionVisualPhoneme}
    model_class = model_classes[args.architecture]
    model_args = {"classes": len(PHONEMES) + 1}
    if include_landmarks:
        model_args["coordinate_dimensions"] = landmark_feature_dimensions(args.coordinate_features)
    if args.architecture in {"coordinates", "gated-fusion"}:
        model_args["landmark_bottleneck"] = args.landmark_bottleneck or None
    model = model_class(**model_args).to(device)
    if args.initialize_coordinate_checkpoint:
        if args.architecture != "gated-fusion" or not args.initialize_coordinate_checkpoint.is_file():
            parser.error("coordinate initialization requires gated-fusion and an existing checkpoint")
        initial = torch.load(args.initialize_coordinate_checkpoint, map_location="cpu",
                             weights_only=True)
        source = initial["model"]
        compatible = {name: value for name, value in source.items()
                      if name in model.state_dict() and model.state_dict()[name].shape == value.shape}
        required_prefixes = ("landmark_encoder.", "temporal.", "classifier.")
        missing_shared = [name for name in model.state_dict()
                          if name.startswith(required_prefixes) and name not in compatible]
        if missing_shared:
            raise ValueError(f"coordinate checkpoint is incompatible: {missing_shared[:3]}")
        model.load_state_dict(compatible, strict=False)
        LOGGER.info("initialized %d coordinate-path tensors from %s", len(compatible),
                    args.initialize_coordinate_checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    transition_log_probs = None
    if args.lm_weight:
        sequences = []
        for row in train.rows:
            stem = Path(row["video"]).stem
            alignment = (args.data_root / "landmark-experiment" / "aligned"
                         / f"s{row['speaker_id']}" / f"{stem}.json")
            sequences.append(phoneme_target(str(alignment)))
        transition_log_probs = fit_bigram_log_probs(sequences, len(PHONEMES) + 1,
                                                     args.lm_smoothing)
    LOGGER.info("device=%s parameters=%d train_clips=%d validation_clips=%d crop=%s architecture=%s horizontal_flip=%s",
                device, model.parameter_count, len(train), len(validation), args.crop,
                args.architecture, args.horizontal_flip)
    if train.excluded or validation.excluded:
        LOGGER.warning("excluded invalid CTC clips: train=%d validation=%d",
                       len(train.excluded), len(validation.excluded))

    history = []
    best_per = selected_checkpoint_per = float("inf")
    best_selection_key = (float("inf"),)
    stale = 0
    for epoch in range(1, args.epochs + 1):
        training = run_epoch(model, train_loader, loss_fn, device, args.architecture,
                             optimizer, deadline,
                             image_modality_dropout=args.image_modality_dropout,
                             coordinate_modality_dropout=args.coordinate_modality_dropout)
        if not training["complete"] or (deadline is not None and time.monotonic() >= deadline):
            LOGGER.warning("stopping before validation because the training time budget is exhausted")
            break
        with torch.inference_mode():
            valid = run_epoch(model, validation_loader, loss_fn, device, args.architecture,
                              top_n=args.top_n, beam_width=args.beam_width,
                              beam_token_top_k=args.beam_token_top_k,
                              transition_log_probs=transition_log_probs,
                              lm_weight=args.lm_weight)
        if args.selection_metric == "top-n-exact":
            selection_key = (-valid["top_n_exact_accuracy"], valid["oracle_per_at_n"])
            # Exact sequence recall is deliberately the selection priority, but
            # it is sparse early in training. Use its dense oracle-PER tie-break
            # directly for learning-rate scheduling.
            scheduler_value = valid["oracle_per_at_n"]
        elif args.selection_metric == "oracle-per-at-n":
            selection_key = (valid["oracle_per_at_n"],)
            scheduler_value = valid["oracle_per_at_n"]
        else:
            selection_key = (valid["per"],)
            scheduler_value = valid["per"]
        scheduler.step(scheduler_value)
        row = {"epoch": epoch, "train": training, "validation": valid,
               "learning_rate": optimizer.param_groups[0]["lr"]}
        if hasattr(model, "image_gate"):
            row["image_gate"] = model.image_gate
        history.append(row)
        best_per = min(best_per, valid["per"])
        LOGGER.info("epoch=%d train_loss=%.4f train_PER=%.2f%% validation_loss=%.4f validation_PER=%.2f%%",
                    epoch, training["loss"], 100 * training["per"], valid["loss"], 100 * valid["per"])
        if args.top_n > 1:
            LOGGER.info("epoch=%d top_%d_exact=%.2f%% oracle_PER@%d=%.2f%% beam_width=%d",
                        epoch, args.top_n, 100 * valid["top_n_exact_accuracy"], args.top_n,
                        100 * valid["oracle_per_at_n"], args.beam_width)
        if hasattr(model, "image_gate"):
            LOGGER.info("epoch=%d image_gate=%.5f", epoch, model.image_gate)
        if selection_key < best_selection_key:
            best_selection_key = selection_key
            selected_checkpoint_per = valid["per"]
            stale = 0
            torch.save({"model": model.state_dict(), "phones": PHONEMES, "crop": args.crop,
                        "image_size": args.image_size, "epoch": epoch,
                        "validation": valid, "architecture": args.architecture,
                        "coordinate_mode": args.coordinate_mode,
                        "coordinate_features": args.coordinate_features,
                        "coordinate_dimensions": landmark_feature_dimensions(
                            args.coordinate_features) if include_landmarks else None,
                        "landmark_bottleneck": args.landmark_bottleneck or None,
                        "selection_metric": args.selection_metric,
                        "decoding": {"top_n": args.top_n, "beam_width": args.beam_width,
                                     "beam_token_top_k": args.beam_token_top_k,
                                     "lm_weight": args.lm_weight,
                                     "lm_smoothing": args.lm_smoothing},
                        "bigram_log_probs": transition_log_probs,
                        "landmark_points": 41 if include_landmarks else None}, args.output_dir / "best.pt")
        else:
            stale += 1
        (args.output_dir / "metrics.json").write_text(json.dumps({
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "task": "visual-only face-video to phoneme sequence using CTC",
            "split": {"train_speakers": list(range(1, 9)), "validation_speakers": [9, 10],
                      "test_speakers": [11, 12], "test_policy": "not evaluated during development"},
            "excluded_invalid_ctc": {"train": train.excluded, "validation": validation.excluded},
            "model_parameters": model.parameter_count, "device": str(device),
            "arguments": vars(args) | {
                "data_root": str(args.data_root),
                "output_dir": str(args.output_dir),
                "frame_cache_dir": str(args.frame_cache_dir) if args.frame_cache_dir else None,
                "initialize_coordinate_checkpoint": (
                    str(args.initialize_coordinate_checkpoint)
                    if args.initialize_coordinate_checkpoint else None),
            },
            "history": history, "best_validation_per": best_per,
            "selected_checkpoint_validation_per": selected_checkpoint_per,
            "selection_metric": args.selection_metric,
            "best_selection_key": list(best_selection_key),
        }, indent=2) + "\n")
        if stale >= args.patience:
            LOGGER.info("early stopping after %d epochs", epoch)
            break
    for handler in LOGGER.handlers:
        handler.flush()


if __name__ == "__main__":
    main()
