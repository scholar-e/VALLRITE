"""Bounded, paired audio-to-visual probability distillation pilot on LRS3."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import tempfile
import time

import torch
import torch.nn.functional as F

from AudioPhonemeLabeler.phones import IPA_TO_ARPABET, Wav2Vec2PhoneTeacher
from VisualPhoneme.data import collate_aligned_clips, greedy_decode
from VisualPhoneme.evaluate_nbest import load_model
from VisualPhoneme.lrs3_data import Lrs3Clips
from VisualPhoneme.thermal import ThermalGuard
from VisualPhoneme.train import edit_totals, upsample_ctc_logits
from VisualPhoneme.visemes import (PHONE_TO_VISUAL_GROUP, REST_GROUP_ID,
    PHONE_ID_TO_VISUAL_GROUP_ID_WITH_REST)

LOGGER = logging.getLogger(__name__)


def project_probabilities(probabilities, vocabulary, labels):
    """Sum IPA probability mass into visual families, leaving blank unmapped.

    Return the conditional distribution over mapped speech families and its
    original probability mass. Neither CTC blank nor unknown tokens mean REST.
    """
    projected = probabilities.new_zeros((len(probabilities), len(labels) + 1))
    for token, index in vocabulary.items():
        phone = IPA_TO_ARPABET.get(token)
        group = PHONE_TO_VISUAL_GROUP.get(phone)
        if group in labels:
            projected[:, labels.index(group) + 1] += probabilities[:, index]
    mass = projected.sum(-1)
    return projected / mass.clamp_min(1e-8).unsqueeze(-1), mass


def soft_loss(logits, targets, mass, frame_labels, threshold=0.2):
    # REST comes from forced inter-word gaps, rather than acoustic CTC blank.
    valid = (mass >= threshold) & (frame_labels >= 0) & (frame_labels != REST_GROUP_ID)
    confidence = mass * valid
    kl = F.kl_div(logits.log_softmax(-1), targets, reduction="none").sum(-1)
    return (kl * confidence).sum() / confidence.sum().clamp_min(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/lrs3"))
    parser.add_argument("--teacher-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=128)
    parser.add_argument("--validation-limit", type=int, default=128)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--max-minutes", type=float, default=10)
    parser.add_argument("--soft-weight", type=float, default=0.2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.train_limit, args.validation_limit, args.steps, args.max_minutes) <= 0:
        parser.error("limits must be positive")
    if args.soft_weight < 0:
        parser.error("soft weight must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s",
        handlers=[logging.FileHandler(args.output / "pilot.log"), logging.StreamHandler()])
    if not torch.cuda.is_available():
        raise RuntimeError("Host CUDA is required")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("target_inventory") != "visual-groups-rest" or checkpoint["architecture"] != "large-image":
        raise ValueError("pilot requires the current large-image visual-groups-rest checkpoint")
    device = torch.device("cuda")
    labels = tuple(checkpoint["phones"])
    mapping = torch.tensor(PHONE_ID_TO_VISUAL_GROUP_ID_WITH_REST, device=device)
    def dataset(split, limit):
        return Lrs3Clips(args.data_root, split, checkpoint["image_size"], checkpoint["crop"],
            limit, False, False, True, False, "eye-normalized",
            args.data_root / "frame-cache", "position", teacher_labels_dir=args.teacher_labels,
            max_chunk_phones=40, include_frame_targets=True, include_rest_targets=True)
    train = dataset("train", args.train_limit)
    validation = dataset("validation", args.validation_limit)
    train_sources = {row["source_clip_id"] for row in train.rows}
    if train_sources & {row["source_clip_id"] for row in validation.rows}:
        raise ValueError("training/validation source overlap")
    manifest = {"checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "train_ids": [row["clip_id"] for row in train.rows],
        "validation_ids": [row["clip_id"] for row in validation.rows],
        "labels": labels, "teacher": "facebook/wav2vec2-lv-60-espeak-cv-ft",
        "soft_weight": args.soft_weight, "steps": args.steps,
        "scope": "development pilot; group PER, not exact-phone PER"}
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != json.loads(json.dumps(manifest)):
        raise ValueError("existing pilot manifest differs")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    cache = args.output / "audio-probabilities.pt"
    if cache.exists():
        soft = torch.load(cache, map_location="cpu", weights_only=True)
    else:
        teacher = Wav2Vec2PhoneTeacher(manifest["teacher"], "cuda", True,
                                      Path("datasets/audio-teacher-smoke/.model-cache"))
        vocabulary = teacher.processor.tokenizer.get_vocab()
        stride, receptive_field = 1, 1
        for kernel, layer_stride in zip(teacher.model.config.conv_kernel,
                                        teacher.model.config.conv_stride, strict=True):
            receptive_field += (kernel - 1) * stride
            stride *= layer_stride
        soft = {}
        with tempfile.TemporaryDirectory(prefix="visual-audio-distill-") as directory:
            wav = Path(directory) / "segment.wav"
            for index, row in enumerate(train.rows):
                if Path("PAUSE").exists():
                    LOGGER.info("PAUSE observed during audio export; restart reuses completed pilot files")
                    return
                start = row["frame_start"] / row["fps"]
                duration = (row["frame_end"] - row["frame_start"]) / row["fps"]
                subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i",
                    str(args.data_root / row["video"]), "-ss", str(start), "-t", str(duration),
                    "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-y", str(wav)],
                    check=True, capture_output=True)
                log_probs, audio_duration = teacher._infer(wav)
                probabilities, mass = project_probabilities(log_probs.exp(), vocabulary, labels)
                # Use Wav2Vec2 receptive-field centers rather than stretching end frames.
                centers = ((torch.arange(row["frames"]) + 0.5) / row["fps"] * 16000
                           - (receptive_field - 1) / 2) / stride
                indexes = centers.round().long().clamp(0, len(mass) - 1)
                soft[row["clip_id"]] = (probabilities[indexes], mass[indexes])
                if index % 16 == 0:
                    LOGGER.info("audio export %d/%d duration=%.3f", index + 1, len(train), audio_duration)
        torch.save(soft, cache)
        del teacher
        torch.cuda.empty_cache()
    def evaluate(model):
        model.eval()
        errors = inclusive_errors = phones = inclusive_tokens = 0
        with torch.inference_mode():
            for index in range(len(validation)):
                video, packed, lengths, target_lengths, frames, ids = collate_aligned_clips([validation[index]])
                logits, output_lengths = upsample_ctc_logits(model(video.to(device)), lengths,
                    checkpoint.get("ctc_upsample_factor", 1))
                predicted = greedy_decode(logits, output_lengths)[0]
                reference = mapping[packed.to(device)].tolist()
                ie, ic = edit_totals(reference, predicted)
                e, c = edit_totals([x for x in reference if x != REST_GROUP_ID],
                                   [x for x in predicted if x != REST_GROUP_ID])
                errors += e; phones += c; inclusive_errors += ie; inclusive_tokens += ic
        return {"stripped_group_per": errors / phones,
                "inclusive_group_per": inclusive_errors / inclusive_tokens,
                "clips": len(validation), "phones": phones}
    report = {"manifest": manifest, "arms": {}}
    deadline = time.monotonic() + args.max_minutes * 60
    for arm, weight in (("control", 0.0), ("distilled", args.soft_weight)):
        torch.manual_seed(42)
        _, model = load_model(checkpoint, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        state_path = args.output / f"{arm}-resume.pt"
        first_step = 0
        if args.resume and state_path.exists():
            state = torch.load(state_path, map_location=device, weights_only=True)
            model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
            first_step = state["step"]
            torch.set_rng_state(state["rng"].cpu())
            torch.cuda.set_rng_state(state["cuda_rng"].cpu())
        if arm == "control":
            report["baseline"] = evaluate(model) if first_step == 0 else None
            LOGGER.info("baseline=%s", report["baseline"])
        ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
        for step in range(first_step, args.steps):
            if Path("PAUSE").exists() or time.monotonic() >= deadline:
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "step": step, "rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state()}, state_path)
                LOGGER.info("stopping at arm=%s step=%d; resumable state retained", arm, step)
                break
            model.train()
            index = step % len(train)
            video, packed, lengths, target_lengths, frames, ids = collate_aligned_clips([train[index]])
            raw = model(video.to(device))
            frame_labels = frames.to(device)
            valid = frame_labels >= 0
            mapped_frames = torch.full_like(frame_labels, -100)
            mapped_frames[valid] = mapping[frame_labels[valid]]
            probs, mass = soft[ids[0]]
            if len(probs) != raw.shape[0]:
                raise ValueError("audio/video frame count mismatch")
            kd = soft_loss(raw[:, 0], probs.to(device), mass.to(device), mapped_frames[0])
            frame_loss = F.cross_entropy(raw.permute(1, 2, 0), mapped_frames, ignore_index=-100)
            logits, output_lengths = upsample_ctc_logits(raw, lengths, checkpoint.get("ctc_upsample_factor", 1))
            loss = ctc(logits.log_softmax(-1), mapping[packed.to(device)], output_lengths,
                       target_lengths) + 0.1 * frame_loss + weight * kd
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            if (step + 1) % 10 == 0 or step == 0:
                LOGGER.info("arm=%s step=%d loss=%.4f soft_KL=%.4f", arm, step + 1, loss.item(), kd.item())
            if (step + 1) % 25 == 0 or step + 1 == args.steps:
                output_checkpoint = dict(checkpoint, model=model.state_dict(),
                    audio_distillation={"arm": arm, "step": step + 1, "weight": weight})
                torch.save(output_checkpoint, args.output / f"{arm}-step-{step+1:04d}.pt")
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "step": step + 1, "rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state()}, state_path)
        else:
            report["arms"][arm] = evaluate(model)
            LOGGER.info("arm=%s validation=%s", arm, report["arms"][arm])
            (args.output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
            del model, optimizer
            torch.cuda.empty_cache()
            continue
        return
    LOGGER.info("paired pilot complete")


if __name__ == "__main__":
    watchdog_parser = argparse.ArgumentParser(add_help=False)
    watchdog_parser.add_argument("--output", type=Path, required=True)
    watchdog_args, _ = watchdog_parser.parse_known_args()
    watchdog_args.output.mkdir(parents=True, exist_ok=True)
    with ThermalGuard(watchdog_args.output, gpu_required=True):
        main()
