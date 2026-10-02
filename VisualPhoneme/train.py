"""Train and evaluate the compact visual-phoneme CTC model."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from VisualPhoneme.data import (GridClips, PHONEMES, collate_aligned_clips,
                                collate_aligned_fusion_clips,
                                collate_aligned_landmark_clips, collate_clips,
                                collate_fusion_clips, collate_landmark_clips,
                                ctc_prefix_beam_search, fit_bigram_log_probs,
                                greedy_decode, landmark_feature_dimensions)
from VisualPhoneme.lrs3_data import Lrs3Clips
from VisualPhoneme.model import (AutoAvsrFusionVisualPhoneme,
                                 CompactFusionVisualPhoneme,
                                 CompactGatedFusionVisualPhoneme,
                                 CompactLandmarkPhoneme,
                                 CompactTongueGatedFusionVisualPhoneme,
                                 CompactVisualPhoneme,
                                 LargeGatedFusionVisualPhoneme,
                                 LargeVisualPhoneme,
                                 LargeTransformerFusionVisualPhoneme)
from VisualPhoneme.visemes import (PHONE_ID_TO_VISUAL_GROUP_ID,
                                   PHONE_ID_TO_VISUAL_GROUP_ID_WITH_REST,
                                   VISUAL_GROUPS, VISUAL_GROUPS_WITH_REST)
from VisualPhoneme.thermal import ThermalGuard
LOGGER = logging.getLogger("visual_phoneme")


class PauseController:
    def __init__(self, marker: Path):
        self.marker = marker
        self.requested = False

    def request(self, signum, _frame) -> None:
        self.requested = True
        LOGGER.warning("pause requested by signal %s; saving after the current batch", signum)

    def should_pause(self) -> bool:
        return self.requested or self.marker.exists()


def random_state() -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), numpy_state[2],
                  numpy_state[3], numpy_state[4]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_random_state(state: dict) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                         numpy_state[2], numpy_state[3], numpy_state[4]))
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_training_state(path: Path, model, optimizer, scheduler, args, resume_epoch: int,
                        history: list, best_per: float, selected_checkpoint_per: float,
                        best_selection_key: tuple, stale: int) -> None:
    """Save all state needed to restart training at an epoch boundary."""
    state = {
        "format": "visual-phoneme-training-state-0.1",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "resume_epoch": resume_epoch,
        "history": history,
        "best_per": best_per,
        "selected_checkpoint_per": selected_checkpoint_per,
        "best_selection_key": list(best_selection_key),
        "stale": stale,
        "random_state": random_state(),
        "architecture": args.architecture,
        "coordinate_mode": args.coordinate_mode,
        "coordinate_features": args.coordinate_features,
        "ctc_upsample_factor": args.ctc_upsample_factor,
        "target_inventory": args.target_inventory,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    LOGGER.info("saved resumable training state at epoch=%d to %s", resume_epoch, path)


def validate_resume_configuration(state: dict, args) -> None:
    expected = {
        "architecture": args.architecture,
        "coordinate_mode": args.coordinate_mode,
        "coordinate_features": args.coordinate_features,
        "ctc_upsample_factor": args.ctc_upsample_factor,
        "target_inventory": args.target_inventory,
    }
    mismatches = {key: (state.get(key), value) for key, value in expected.items()
                  if state.get(key) != value}
    if mismatches:
        raise ValueError(f"resume-state configuration mismatch: {mismatches}")


def total_failure_reason(training: dict, valid: dict, min_valid_loss: float,
                         divergence_factor: float) -> str | None:
    """Return a reason string when a run has totally failed (diverged), else None.

    A run counts as a total failure when a tracked loss or metric becomes
    non-finite, or when the validation loss explodes past ``divergence_factor``
    times the lowest validation loss seen so far in the run.
    """
    if not (math.isfinite(training["loss"])
            and math.isfinite(valid["loss"])
            and math.isfinite(valid["per"])
            and math.isfinite(valid.get("oracle_per_at_n", 0.0))):
        return "non-finite loss or PER (diverged)"
    if (divergence_factor > 0 and math.isfinite(min_valid_loss)
            and valid["loss"] > divergence_factor * min_valid_loss):
        return (f"validation loss {valid['loss']:.4f} exceeded "
                f"{divergence_factor:.1f}x the running minimum {min_valid_loss:.4f}")
    return None


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


def upsample_ctc_logits(logits: torch.Tensor, lengths: torch.Tensor,
                        factor: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Repeat temporal emissions to give dense-label CTC more alignment states."""
    if factor < 1:
        raise ValueError("CTC upsample factor must be positive")
    if factor == 1:
        return logits, lengths
    return logits.repeat_interleave(factor, dim=0), lengths * factor


def run_epoch(model, loader, loss_fn, device, architecture, optimizer=None, deadline=None,
              top_n=1, beam_width=1, beam_token_top_k=None,
              transition_log_probs=None, lm_weight=0.0,
              image_modality_dropout=0.0, coordinate_modality_dropout=0.0,
              freeze_coordinate_path=False, freeze_image_path=False,
              ctc_upsample_factor=1,
              aligned_frame_loss_weight=0.0, ctc_loss_weight=1.0,
              target_id_map=None, should_pause=None):
    training = optimizer is not None
    model.train(training)
    if training and freeze_coordinate_path:
        for name in ("landmark_encoder", "temporal", "classifier"):
            getattr(model, name).eval()
    if training and freeze_image_path:
        model.frame_encoder.eval()
    total_loss = total_errors = total_phones = clips = 0
    oracle_errors = exact_hits = 0
    started = time.perf_counter()
    for step, batch in enumerate(loader, 1):
        if should_pause is not None and should_pause():
            LOGGER.warning("pause observed before step=%d", step)
            break
        frame_targets = None
        if architecture in {"fusion", "gated-fusion", "tongue-gated-fusion",
                            "large-gated-fusion", "large-transformer-fusion",
                            "autoavsr-fusion"}:
            if training and aligned_frame_loss_weight:
                (video, landmarks, landmark_mask, targets, lengths, target_lengths,
                 frame_targets, _) = batch
            else:
                video, landmarks, landmark_mask, targets, lengths, target_lengths, _ = batch
            landmarks = landmarks.to(device, non_blocking=True)
            landmark_mask = landmark_mask.to(device, non_blocking=True)
            video = video.to(device, non_blocking=True)
        elif architecture == "coordinates":
            if training and aligned_frame_loss_weight:
                landmarks, landmark_mask, targets, lengths, target_lengths, frame_targets, _ = batch
            else:
                landmarks, landmark_mask, targets, lengths, target_lengths, _ = batch
            landmarks = landmarks.to(device, non_blocking=True)
            landmark_mask = landmark_mask.to(device, non_blocking=True)
            video = None
        else:
            if training and aligned_frame_loss_weight:
                video, targets, lengths, target_lengths, frame_targets, _ = batch
            else:
                video, targets, lengths, target_lengths, _ = batch
            video = video.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if target_id_map is not None:
            targets = target_id_map[targets]
            if frame_targets is not None:
                frame_targets = frame_targets.to(device, non_blocking=True)
                valid_frame_targets = frame_targets >= 0
                frame_targets[valid_frame_targets] = target_id_map[
                    frame_targets[valid_frame_targets]
                ]
        if training and architecture in {"fusion", "gated-fusion", "tongue-gated-fusion",
                                         "large-gated-fusion", "large-transformer-fusion",
                                         "autoavsr-fusion"}:
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
            if architecture in {"fusion", "gated-fusion", "tongue-gated-fusion",
                                "large-gated-fusion", "large-transformer-fusion",
                                "autoavsr-fusion"}:
                logits = model(video, landmarks, landmark_mask)
            elif architecture == "coordinates":
                logits = model(landmarks, landmark_mask)
            else:
                logits = model(video)
            frame_loss = None
            if frame_targets is not None:
                frame_loss = F.cross_entropy(
                    logits.permute(1, 2, 0), frame_targets.to(device, non_blocking=True),
                    ignore_index=-100,
                )
            logits, output_lengths = upsample_ctc_logits(
                logits, lengths, ctc_upsample_factor
            )
            ctc_loss = loss_fn(logits.log_softmax(-1), targets, output_lengths, target_lengths)
            loss = ctc_loss_weight * ctc_loss
            if frame_loss is not None:
                loss = loss + aligned_frame_loss_weight * frame_loss
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        hypotheses = greedy_decode(logits.detach(), output_lengths)
        beam_candidates = None
        if not training and top_n > 1:
            log_probabilities = logits.detach().log_softmax(-1).transpose(0, 1).cpu()
            beam_candidates = [
                ctc_prefix_beam_search(scores[:int(length)], beam_width, top_n,
                                       beam_token_top_k, transition_log_probs, lm_weight)
                for scores, length in zip(log_probabilities, output_lengths)
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
    result = {"loss": total_loss / max(clips, 1),
              "per": total_errors / max(total_phones, 1),
              "errors": total_errors, "phones": total_phones, "clips": clips,
              "seconds": time.perf_counter() - started,
              "complete": clips == len(loader.dataset),
              "paused": bool(should_pause is not None and should_pause())}
    if not training and top_n > 1:
        result.update({"top_n": top_n, "beam_width": beam_width,
                       "beam_token_top_k": beam_token_top_k,
                       "top_n_exact_hits": exact_hits,
                       "top_n_exact_accuracy": exact_hits / clips,
                       "oracle_errors_at_n": oracle_errors,
                       "oracle_per_at_n": oracle_errors / total_phones})
    return result


def _main(resources: ExitStack) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("grid", "lrs3"), default="grid")
    parser.add_argument("--data-root", type=Path, default=Path("datasets/grid-pilot"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/visual-phoneme-mouth"))
    parser.add_argument("--crop", choices=("face", "mouth", "full"), default="mouth")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--frontend-learning-rate", type=float,
                        help="optional lower learning rate for frame_encoder parameters")
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--max-minutes", type=float,
                        help="stop cleanly before starting validation once this wall-time budget is reached")
    parser.add_argument("--divergence-factor", type=float, default=5.0,
                        help="abort as a total failure when validation loss exceeds this many "
                             "times its running minimum (zero disables)")
    parser.add_argument("--baseline-failure-epochs", type=int, default=0,
                        help="abort as a total failure after this many epochs without beating "
                             "the checkpoint the run was initialized or resumed from (zero disables)")
    parser.add_argument("--landmark-fusion", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--architecture", choices=("image", "large-image", "coordinates", "fusion",
                                                    "gated-fusion", "tongue-gated-fusion",
                                                    "large-gated-fusion",
                                                    "large-transformer-fusion",
                                                    "autoavsr-fusion"),
                        default="image", help="input modality used by this ablation")
    parser.add_argument("--horizontal-flip", action=argparse.BooleanOptionalAction, default=None,
                        help="randomly reflect training images (default: image model only)")
    parser.add_argument("--coordinate-mode",
                        choices=("eye-normalized", "pose-frontalized",
                                 "clip-centered", "constant"),
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
    parser.add_argument("--initialize-image-checkpoint", type=Path,
                        help="initialize gated fusion image encoders from an image model")
    parser.add_argument("--initialize-checkpoint", type=Path,
                        help="initialize every tensor from a compatible checkpoint")
    parser.add_argument("--initialize-rest-checkpoint", type=Path,
                        help="expand a visual-group checkpoint with a new REST output")
    parser.add_argument("--resume-state", type=Path,
                        help="restore a pausable training state including optimizer and history")
    parser.add_argument("--pause-file", type=Path,
                        help="pause at a batch boundary when this marker exists (default: OUTPUT/PAUSE)")
    parser.add_argument("--freeze-coordinate-epochs", type=int, default=0,
                        help="train image residuals alone for the first N epochs")
    parser.add_argument("--freeze-image-epochs", type=int, default=0,
                        help="freeze a pretrained image frontend for the first N epochs")
    parser.add_argument("--pretrained-visual-frontend", type=Path,
                        help="published Auto-AVSR model.pth used to initialize its frontend")
    parser.add_argument("--image-gate-initial-probability", type=float,
                        default=0.002472623)
    parser.add_argument("--inner-mouth-gate-initial-probability", type=float, default=0.05)
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
    parser.add_argument("--ctc-upsample-factor", type=int, default=1,
                        help="repeat each model emission this many times before CTC")
    parser.add_argument("--lrs3-teacher-labels", type=Path,
                        help="accepted audio-teacher labels used to make LRS3 training chunks")
    parser.add_argument("--max-chunk-phones", type=int, default=0,
                        help="maximum phones per timestamped LRS3 training chunk; zero disables")
    parser.add_argument("--min-teacher-transcript-agreement", type=float, default=0.8)
    parser.add_argument("--min-rest-seconds", type=float, default=0.08,
                        help="minimum forced-aligned inter-word gap labeled REST")
    parser.add_argument("--aligned-frame-loss-weight", type=float, default=0.0,
                        help="auxiliary CE weight on timestamped nonblank LRS3 frames")
    parser.add_argument("--frame-only-epochs", type=int, default=0,
                        help="warm up only on aligned frame labels for this many epochs")
    parser.add_argument("--ctc-loss-weight", type=float, default=1.0,
                        help="CTC weight after any aligned-frame warm-up")
    parser.add_argument("--checkpoint-every", type=int, default=1,
                        help="save epoch weights every N validations; zero disables")
    parser.add_argument("--target-inventory",
                        choices=("phones", "visual-groups", "visual-groups-rest"),
                        default="phones",
                        help="train exact ARPAbet phones or the fixed visual-group partition")
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
            or args.weight_decay < 0 or args.learning_rate <= 0
            or (args.frontend_learning_rate is not None
                and args.frontend_learning_rate <= 0)
            or args.lm_weight < 0 or args.lm_smoothing <= 0
            or args.freeze_coordinate_epochs < 0 or args.freeze_image_epochs < 0
            or args.ctc_upsample_factor < 1
            or args.max_chunk_phones < 0
            or args.min_rest_seconds < 0
            or args.aligned_frame_loss_weight < 0
            or args.frame_only_epochs < 0
            or args.ctc_loss_weight <= 0
            or args.checkpoint_every < 0
            or not 0 <= args.min_teacher_transcript_agreement <= 1
            or not 0 < args.image_gate_initial_probability < 1
            or not 0 < args.inner_mouth_gate_initial_probability < 1
            or args.divergence_factor < 0
            or args.baseline_failure_epochs < 0):
        parser.error("invalid regularization, bottleneck, weight-decay, or LM setting")
    if (args.lrs3_teacher_labels is None) != (args.max_chunk_phones == 0):
        parser.error("--lrs3-teacher-labels requires a positive --max-chunk-phones")
    if args.lrs3_teacher_labels is not None and args.dataset != "lrs3":
        parser.error("audio-teacher chunks are supported only for LRS3")
    if args.coordinate_mode == "pose-frontalized" and args.dataset != "lrs3":
        parser.error("pose-frontalized coordinates require LRS3 68-point landmarks")
    if args.aligned_frame_loss_weight and args.lrs3_teacher_labels is None:
        parser.error("aligned frame loss requires LRS3 teacher chunks")
    if args.target_inventory == "visual-groups-rest" and args.lrs3_teacher_labels is None:
        parser.error("REST targets require forced-aligned LRS3 teacher labels")
    if args.frame_only_epochs and not args.aligned_frame_loss_weight:
        parser.error("frame-only warm-up requires a positive aligned frame loss weight")
    if ((args.architecture == "autoavsr-fusion")
            != (args.pretrained_visual_frontend is not None)):
        parser.error("autoavsr-fusion requires exactly one --pretrained-visual-frontend")
    if (args.pretrained_visual_frontend is not None
            and not args.pretrained_visual_frontend.is_file()):
        parser.error("pretrained visual frontend checkpoint does not exist")
    if args.resume_state and (not args.resume_state.is_file()
                              or args.initialize_checkpoint or args.initialize_rest_checkpoint
                              or args.initialize_coordinate_checkpoint
                              or args.initialize_image_checkpoint):
        parser.error("--resume-state must exist and cannot be combined with initialization")

    if args.landmark_fusion:
        if args.architecture != "image":
            parser.error("--landmark-fusion cannot be combined with --architecture")
        args.architecture = "fusion"
    if args.horizontal_flip is None:
        args.horizontal_flip = args.architecture in {"image", "large-image"}
    if args.horizontal_flip and args.architecture not in {"image", "large-image"}:
        parser.error("horizontal reflection requires an image-only model until landmark permutation exists")
    if args.coordinate_mode != "eye-normalized" and args.architecture in {"image", "large-image"}:
        parser.error("--coordinate-mode applies only to coordinates or fusion")

    configure_logging(args.output_dir)
    pause = PauseController(args.pause_file or args.output_dir / "PAUSE")
    signal.signal(signal.SIGINT, pause.request)
    signal.signal(signal.SIGTERM, pause.request)
    deadline = time.monotonic() + args.max_minutes * 60 if args.max_minutes is not None else None
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resources.enter_context(ThermalGuard(args.output_dir, device.type == "cuda"))
    include_landmarks = args.architecture in {
        "coordinates", "fusion", "gated-fusion", "tongue-gated-fusion",
        "large-gated-fusion", "large-transformer-fusion", "autoavsr-fusion"
    }
    include_video = args.architecture in {
        "image", "large-image", "fusion", "gated-fusion", "tongue-gated-fusion",
        "large-gated-fusion", "large-transformer-fusion", "autoavsr-fusion"
    }
    dataset_class = GridClips if args.dataset == "grid" else Lrs3Clips
    teacher_options = ({
        "teacher_labels_dir": args.lrs3_teacher_labels,
        "max_chunk_phones": args.max_chunk_phones,
        "min_teacher_transcript_agreement": args.min_teacher_transcript_agreement,
        "include_frame_targets": bool(args.aligned_frame_loss_weight),
        "include_rest_targets": args.target_inventory == "visual-groups-rest",
        "min_rest_seconds": args.min_rest_seconds,
    } if args.dataset == "lrs3" else {})
    train = dataset_class(args.data_root, "train", args.image_size, args.crop, args.train_limit,
                          True, include_landmarks, include_video, args.horizontal_flip,
                          args.coordinate_mode, args.frame_cache_dir, args.coordinate_features,
                          args.landmark_jitter, args.point_dropout, args.frame_span_dropout,
                          args.max_frame_span, **teacher_options)
    validation_teacher_options = dict(teacher_options)
    validation_teacher_options["include_frame_targets"] = False
    validation = dataset_class(
        args.data_root, "validation", args.image_size, args.crop,
        args.validation_limit, False, include_landmarks, include_video, False,
        args.coordinate_mode, args.frame_cache_dir, args.coordinate_features,
        **validation_teacher_options,
    )
    collate_functions = {"image": collate_clips, "large-image": collate_clips,
                         "coordinates": collate_landmark_clips,
                         "fusion": collate_fusion_clips,
                         "gated-fusion": collate_fusion_clips,
                         "tongue-gated-fusion": collate_fusion_clips,
                         "large-gated-fusion": collate_fusion_clips,
                         "large-transformer-fusion": collate_fusion_clips,
                         "autoavsr-fusion": collate_fusion_clips}
    aligned_collate_functions = {
        "image": collate_aligned_clips,
        "large-image": collate_aligned_clips,
        "coordinates": collate_aligned_landmark_clips,
        "fusion": collate_aligned_fusion_clips,
        "gated-fusion": collate_aligned_fusion_clips,
        "tongue-gated-fusion": collate_aligned_fusion_clips,
        "large-gated-fusion": collate_aligned_fusion_clips,
        "large-transformer-fusion": collate_aligned_fusion_clips,
        "autoavsr-fusion": collate_aligned_fusion_clips,
    }
    loader_args = {"batch_size": args.batch_size, "num_workers": args.workers,
                   "pin_memory": device.type == "cuda",
                   "persistent_workers": args.workers > 0}
    train_loader = DataLoader(
        train, shuffle=True,
        collate_fn=(aligned_collate_functions[args.architecture]
                    if args.aligned_frame_loss_weight else collate_functions[args.architecture]),
        **loader_args,
    )
    validation_loader = DataLoader(
        validation, shuffle=False, collate_fn=collate_functions[args.architecture],
        **loader_args,
    )
    model_classes = {"image": CompactVisualPhoneme, "large-image": LargeVisualPhoneme,
                     "coordinates": CompactLandmarkPhoneme,
                     "fusion": CompactFusionVisualPhoneme,
                     "gated-fusion": CompactGatedFusionVisualPhoneme,
                     "tongue-gated-fusion": CompactTongueGatedFusionVisualPhoneme,
                     "large-gated-fusion": LargeGatedFusionVisualPhoneme,
                     "large-transformer-fusion": LargeTransformerFusionVisualPhoneme,
                     "autoavsr-fusion": AutoAvsrFusionVisualPhoneme}
    model_class = model_classes[args.architecture]
    output_labels = ({"phones": PHONEMES, "visual-groups": VISUAL_GROUPS,
                      "visual-groups-rest": VISUAL_GROUPS_WITH_REST}[args.target_inventory])
    target_id_map = None
    if args.target_inventory in {"visual-groups", "visual-groups-rest"}:
        mapping = (PHONE_ID_TO_VISUAL_GROUP_ID_WITH_REST
                   if args.target_inventory == "visual-groups-rest"
                   else PHONE_ID_TO_VISUAL_GROUP_ID)
        target_id_map = torch.tensor(mapping, device=device)
    model_args = {"classes": len(output_labels) + 1}
    if include_landmarks:
        model_args["coordinate_dimensions"] = landmark_feature_dimensions(args.coordinate_features)
        model_args["landmark_points"] = train.landmark_points
    if args.architecture in {"coordinates", "gated-fusion", "tongue-gated-fusion",
                              "large-gated-fusion", "large-transformer-fusion",
                              "autoavsr-fusion"}:
        model_args["landmark_bottleneck"] = args.landmark_bottleneck or None
    if args.architecture in {"gated-fusion", "tongue-gated-fusion",
                              "large-gated-fusion", "large-transformer-fusion",
                              "autoavsr-fusion"}:
        model_args["image_gate_probability"] = args.image_gate_initial_probability
    if args.architecture == "tongue-gated-fusion":
        model_args["inner_mouth_gate_probability"] = args.inner_mouth_gate_initial_probability
    model = model_class(**model_args).to(device)
    if args.pretrained_visual_frontend:
        source = torch.load(args.pretrained_visual_frontend, map_location="cpu",
                            weights_only=True)
        prefix = "encoder.frontend."
        frontend_state = {
            name.removeprefix(prefix): value for name, value in source.items()
            if name.startswith(prefix)
        }
        missing, unexpected = model.frame_encoder.load_state_dict(frontend_state, strict=False)
        if missing or unexpected:
            raise ValueError(
                f"Auto-AVSR frontend mismatch: missing={missing[:3]} unexpected={unexpected[:3]}"
            )
        LOGGER.info("initialized %d visual frontend tensors from %s",
                    len(frontend_state), args.pretrained_visual_frontend)
    if args.initialize_checkpoint:
        if (not args.initialize_checkpoint.is_file()
                or args.initialize_rest_checkpoint
                or args.initialize_coordinate_checkpoint
                or args.initialize_image_checkpoint):
            parser.error("full initialization requires one existing checkpoint and cannot be combined with partial initialization")
        initial = torch.load(args.initialize_checkpoint, map_location="cpu", weights_only=True)
        expected = {
            "architecture": args.architecture,
            "coordinate_mode": args.coordinate_mode,
            "coordinate_features": args.coordinate_features,
            "ctc_upsample_factor": args.ctc_upsample_factor,
            "target_inventory": args.target_inventory,
        }
        mismatches = {
            key: (initial.get(key, "phones" if key == "target_inventory" else None), value)
            for key, value in expected.items()
            if initial.get(key, "phones" if key == "target_inventory" else None) != value
        }
        if mismatches:
            raise ValueError(f"initial checkpoint configuration mismatch: {mismatches}")
        model.load_state_dict(initial["model"], strict=True)
        LOGGER.info("initialized complete %s model from epoch %s at %s",
                    args.architecture, initial.get("epoch"), args.initialize_checkpoint)
    if args.initialize_rest_checkpoint:
        if (args.target_inventory != "visual-groups-rest"
                or not args.initialize_rest_checkpoint.is_file()
                or args.initialize_checkpoint or args.initialize_coordinate_checkpoint
                or args.initialize_image_checkpoint):
            parser.error("REST initialization requires one visual-group checkpoint")
        initial = torch.load(args.initialize_rest_checkpoint, map_location="cpu",
                             weights_only=True)
        if (initial.get("architecture") != args.architecture
                or initial.get("target_inventory") != "visual-groups"
                or tuple(initial.get("phones", ())) != VISUAL_GROUPS):
            raise ValueError("REST initializer must be a matching visual-group checkpoint")
        source = initial["model"]
        destination = model.state_dict()
        compatible = {name: value for name, value in source.items()
                      if name in destination and destination[name].shape == value.shape}
        model.load_state_dict(compatible, strict=False)
        with torch.no_grad():
            model.classifier.weight[:len(VISUAL_GROUPS) + 1].copy_(
                source["classifier.weight"])
            model.classifier.bias[:len(VISUAL_GROUPS) + 1].copy_(
                source["classifier.bias"])
        LOGGER.info("expanded %s epoch %s with a randomly initialized REST output",
                    args.initialize_rest_checkpoint, initial.get("epoch"))
    if args.initialize_coordinate_checkpoint:
        if (args.architecture not in {"gated-fusion", "tongue-gated-fusion",
                                     "large-gated-fusion", "large-transformer-fusion",
                                     "autoavsr-fusion"}
                or not args.initialize_coordinate_checkpoint.is_file()):
            parser.error("coordinate initialization requires gated fusion and an existing checkpoint")
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
    if args.initialize_image_checkpoint:
        if (args.architecture not in {"gated-fusion", "tongue-gated-fusion",
                                     "large-gated-fusion", "large-transformer-fusion",
                                     "autoavsr-fusion"}
                or not args.initialize_image_checkpoint.is_file()):
            parser.error("image initialization requires gated fusion and an existing checkpoint")
        initial = torch.load(args.initialize_image_checkpoint, map_location="cpu",
                             weights_only=True)
        source = initial["model"]
        image_state = {name: value for name, value in source.items()
                       if name.startswith(("frame_encoder.", "image_projection."))
                       and name in model.state_dict()
                       and model.state_dict()[name].shape == value.shape}
        if not image_state:
            raise ValueError("image checkpoint has no compatible frame encoder")
        model.load_state_dict(image_state, strict=False)
        copied = len(image_state)
        if args.architecture == "tongue-gated-fusion":
            tongue_state = {
                name.replace("frame_encoder.", "inner_mouth_encoder.", 1): value
                for name, value in source.items() if name.startswith("frame_encoder.")
                and name.replace("frame_encoder.", "inner_mouth_encoder.", 1)
                in model.state_dict()
                and model.state_dict()[name.replace(
                    "frame_encoder.", "inner_mouth_encoder.", 1)].shape == value.shape
            }
            model.load_state_dict(tongue_state, strict=False)
            copied += len(tongue_state)
        LOGGER.info("initialized %d image-path tensors from %s", copied,
                    args.initialize_image_checkpoint)
    if args.frontend_learning_rate is not None:
        frontend_parameters = list(model.frame_encoder.parameters())
        frontend_ids = {id(parameter) for parameter in frontend_parameters}
        other_parameters = [parameter for parameter in model.parameters()
                            if id(parameter) not in frontend_ids]
        optimizer_parameters = [
            {"params": other_parameters, "lr": args.learning_rate, "name": "main"},
            {"params": frontend_parameters, "lr": args.frontend_learning_rate,
             "name": "frontend"},
        ]
    else:
        optimizer_parameters = model.parameters()
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    transition_log_probs = None
    if args.lm_weight:
        target_sequences = train.target_sequences()
        if target_id_map is not None:
            target_sequences = [tuple(mapping[token] for token in sequence)
                                for sequence in target_sequences]
        transition_log_probs = fit_bigram_log_probs(
            target_sequences, len(output_labels) + 1, args.lm_smoothing,
        )
    LOGGER.info("device=%s parameters=%d dataset=%s train_clips=%d validation_clips=%d crop=%s architecture=%s horizontal_flip=%s",
                device, model.parameter_count, args.dataset, len(train), len(validation), args.crop,
                args.architecture, args.horizontal_flip)
    if train.excluded or validation.excluded:
        LOGGER.warning("excluded invalid CTC clips: train=%d validation=%d",
                       getattr(train, "excluded_total", len(train.excluded)),
                       getattr(validation, "excluded_total", len(validation.excluded)))

    history = []
    best_per = selected_checkpoint_per = float("inf")
    best_selection_key = (float("inf"),)
    stale = 0
    start_epoch = 1
    if args.resume_state:
        resume = torch.load(args.resume_state, map_location=device, weights_only=True)
        validate_resume_configuration(resume, args)
        model.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        history = resume["history"]
        best_per = resume["best_per"]
        selected_checkpoint_per = resume["selected_checkpoint_per"]
        best_selection_key = tuple(resume["best_selection_key"])
        stale = resume["stale"]
        start_epoch = resume["resume_epoch"]
        restore_random_state(resume["random_state"])
        LOGGER.info("resumed epoch=%d history=%d stale=%d from %s",
                    start_epoch, len(history), stale, args.resume_state)
    resume_path = args.output_dir / "resume.pt"
    min_valid_loss = min((h["validation"]["loss"] for h in history
                          if math.isfinite(h["validation"]["loss"])),
                         default=float("inf"))
    for epoch in range(start_epoch, args.epochs + 1):
        freeze_coordinate_path = (args.architecture in {
            "gated-fusion", "tongue-gated-fusion", "large-gated-fusion",
            "large-transformer-fusion", "autoavsr-fusion"
        } and epoch <= args.freeze_coordinate_epochs)
        freeze_image_path = epoch <= args.freeze_image_epochs
        for name in ("landmark_encoder", "temporal", "classifier"):
            module = getattr(model, name, None)
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad_(not freeze_coordinate_path)
        if freeze_coordinate_path:
            LOGGER.info("epoch=%d coordinate path frozen for image warm-up", epoch)
        for parameter in model.frame_encoder.parameters():
            parameter.requires_grad_(not freeze_image_path)
        if freeze_image_path:
            LOGGER.info("epoch=%d pretrained image frontend frozen", epoch)
        frame_only = epoch <= args.frame_only_epochs
        if frame_only:
            LOGGER.info("epoch=%d CTC disabled for aligned-frame warm-up", epoch)
        training = run_epoch(model, train_loader, loss_fn, device, args.architecture,
                             optimizer, deadline,
                             image_modality_dropout=args.image_modality_dropout,
                             coordinate_modality_dropout=args.coordinate_modality_dropout,
                             freeze_coordinate_path=freeze_coordinate_path,
                             freeze_image_path=freeze_image_path,
                             ctc_upsample_factor=args.ctc_upsample_factor,
                             aligned_frame_loss_weight=args.aligned_frame_loss_weight,
                             ctc_loss_weight=0.0 if frame_only else args.ctc_loss_weight,
                             target_id_map=target_id_map,
                             should_pause=pause.should_pause)
        if not training["complete"] or (deadline is not None and time.monotonic() >= deadline):
            save_training_state(resume_path, model, optimizer, scheduler, args, epoch,
                                history, best_per, selected_checkpoint_per,
                                best_selection_key, stale)
            reason = "pause requested" if training["paused"] else "time budget exhausted"
            LOGGER.warning("stopping before validation because %s", reason)
            break
        with torch.inference_mode():
            valid = run_epoch(model, validation_loader, loss_fn, device, args.architecture,
                              top_n=args.top_n, beam_width=args.beam_width,
                              beam_token_top_k=args.beam_token_top_k,
                              transition_log_probs=transition_log_probs,
                              lm_weight=args.lm_weight,
                              ctc_upsample_factor=args.ctc_upsample_factor,
                              target_id_map=target_id_map)
        failure_reason = total_failure_reason(training, valid, min_valid_loss,
                                              args.divergence_factor)
        if failure_reason is not None:
            LOGGER.error("TOTAL FAILURE at epoch=%d: %s", epoch, failure_reason)
            save_training_state(resume_path, model, optimizer, scheduler, args, epoch,
                                history, best_per, selected_checkpoint_per,
                                best_selection_key, stale)
            break
        min_valid_loss = min(min_valid_loss, valid["loss"])
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
        if args.frontend_learning_rate is not None:
            row["frontend_learning_rate"] = optimizer.param_groups[1]["lr"]
        if hasattr(model, "image_gate"):
            row["image_gate"] = model.image_gate
        if hasattr(model, "inner_mouth_gate"):
            row["inner_mouth_gate"] = model.inner_mouth_gate
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
        if hasattr(model, "inner_mouth_gate"):
            LOGGER.info("epoch=%d inner_mouth_gate=%.5f", epoch, model.inner_mouth_gate)
        checkpoint = {
            "model": model.state_dict(), "phones": output_labels, "crop": args.crop,
            "image_size": args.image_size, "epoch": epoch,
            "training": training, "validation": valid,
            "selection_key": list(selection_key),
            "architecture": args.architecture,
            "coordinate_mode": args.coordinate_mode,
            "coordinate_features": args.coordinate_features,
            "coordinate_dimensions": landmark_feature_dimensions(
                args.coordinate_features) if include_landmarks else None,
            "landmark_bottleneck": args.landmark_bottleneck or None,
            "image_gate_initial_probability": args.image_gate_initial_probability,
            "image_gate": model.image_gate if hasattr(model, "image_gate") else None,
            "inner_mouth_gate_initial_probability": (
                args.inner_mouth_gate_initial_probability
                if args.architecture == "tongue-gated-fusion" else None),
            "inner_mouth_gate": (
                model.inner_mouth_gate if hasattr(model, "inner_mouth_gate") else None),
            "selection_metric": args.selection_metric,
            "ctc_upsample_factor": args.ctc_upsample_factor,
            "decoding": {"top_n": args.top_n, "beam_width": args.beam_width,
                         "beam_token_top_k": args.beam_token_top_k,
                         "lm_weight": args.lm_weight,
                         "lm_smoothing": args.lm_smoothing},
            "bigram_log_probs": transition_log_probs,
            "landmark_points": train.landmark_points if include_landmarks else None,
            "dataset": args.dataset,
            "target_inventory": args.target_inventory,
        }
        if args.checkpoint_every and epoch % args.checkpoint_every == 0:
            epoch_directory = args.output_dir / "epochs"
            epoch_directory.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, epoch_directory / f"epoch-{epoch:04d}.pt")
        if selection_key < best_selection_key:
            best_selection_key = selection_key
            selected_checkpoint_per = valid["per"]
            stale = 0
            torch.save(checkpoint, args.output_dir / "best.pt")
        else:
            stale += 1
        (args.output_dir / "metrics.json").write_text(json.dumps({
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "task": "visual-only face-video to phoneme sequence using CTC",
            "split": ({"train_speakers": list(range(1, 9)),
                       "validation_speakers": [9, 10], "test_speakers": [11, 12],
                       "test_policy": "not evaluated during development"}
                      if args.dataset == "grid" else
                      {"train": "LRS3 trainval excluding AV-HuBERT validation IDs",
                       "validation": "official AV-HuBERT 1,200-ID validation list",
                       "test": "separate 1,321-utterance parquet; not evaluated during development"}),
            "excluded_invalid_ctc": {
                "train": {"count": getattr(train, "excluded_total", len(train.excluded)),
                          "examples": train.excluded[:100]},
                "validation": {"count": getattr(validation, "excluded_total",
                                                   len(validation.excluded)),
                               "examples": validation.excluded[:100]},
            },
            "model_parameters": model.parameter_count, "device": str(device),
            "arguments": vars(args) | {
                "data_root": str(args.data_root),
                "output_dir": str(args.output_dir),
                "frame_cache_dir": str(args.frame_cache_dir) if args.frame_cache_dir else None,
                "initialize_coordinate_checkpoint": (
                    str(args.initialize_coordinate_checkpoint)
                    if args.initialize_coordinate_checkpoint else None),
                "initialize_image_checkpoint": (
                    str(args.initialize_image_checkpoint)
                    if args.initialize_image_checkpoint else None),
                "initialize_checkpoint": (
                    str(args.initialize_checkpoint) if args.initialize_checkpoint else None),
                "initialize_rest_checkpoint": (
                    str(args.initialize_rest_checkpoint)
                    if args.initialize_rest_checkpoint else None),
                "resume_state": str(args.resume_state) if args.resume_state else None,
                "pause_file": str(args.pause_file) if args.pause_file else None,
                "pretrained_visual_frontend": (
                    str(args.pretrained_visual_frontend)
                    if args.pretrained_visual_frontend else None),
                "lrs3_teacher_labels": (
                    str(args.lrs3_teacher_labels) if args.lrs3_teacher_labels else None),
            },
            "history": history, "best_validation_per": best_per,
            "selected_checkpoint_validation_per": selected_checkpoint_per,
            "selection_metric": args.selection_metric,
            "best_selection_key": list(best_selection_key),
        }, indent=2) + "\n")
        save_training_state(resume_path, model, optimizer, scheduler, args, epoch + 1,
                            history, best_per, selected_checkpoint_per,
                            best_selection_key, stale)
        if args.baseline_failure_epochs > 0 and stale >= args.baseline_failure_epochs:
            LOGGER.error("TOTAL FAILURE at epoch=%d: no improvement over the initialization "
                         "baseline for %d epochs (best selection key %s)",
                         epoch, stale, best_selection_key)
            break
        if stale >= args.patience:
            LOGGER.info("early stopping after %d epochs", epoch)
            break
    for handler in LOGGER.handlers:
        handler.flush()


def main() -> None:
    with ExitStack() as resources:
        _main(resources)


if __name__ == "__main__":
    main()
