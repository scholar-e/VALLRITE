"""Fine-tune the top of the Auto-AVSR encoder for phones (encoder plan, step 1).

Frozen-feature phone heads stop at ~26.9% greedy PER on test-select (head C),
while Auto-AVSR's own words converted to phones reach 18.8%, so the encoder
holds phonetic information the head cannot extract. This trainer starts from
the published encoder and a trained phone head, keeps the visual frontend and
the lower conformer blocks frozen, and trains the top ``--trainable-blocks``
blocks plus the head with phone CTC on mouth crops (not cached features).

Forgetting control: the published word CTC head stays attached, and a KL term
keeps the fine-tuned encoder's word-CTC posteriors close to those of a frozen
copy of the original top blocks (the checkpoint ships only the unit list, not
the sentencepiece model, so transcripts cannot be re-tokenized exactly).
Test-select greedy word-CTC WER is logged as a forgetting monitor.

Selection uses test-select (first 660 test clips) only. Test-report is not
evaluated here; export and decode the selected checkpoint once instead.
Training sources are ``MANIFEST@ROOT`` pairs (``video`` paths are relative to
ROOT); one source is sampled per batch by ``--train-weights``.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from VisualPhoneme.autoavsr_encoder import (AUTOAVSR_CROP, AUTOAVSR_MEAN, AUTOAVSR_STD,
                                            AutoAvsrEncoder, ctc_greedy_text, load_autoavsr_ctc,
                                            load_units)
from VisualPhoneme.data import PHONE_TO_ID, PHONEMES
from VisualPhoneme.evaluate_autoavsr_ctc import edit_counts, normalize_words
from VisualPhoneme.thermal import ThermalGuard
from VisualPhoneme.train_autoavsr_phone_head import PhoneHead, build_head, greedy_phones, time_mask

LOGGER = logging.getLogger(__name__)
HEAD_KEYS = ("upsample", "dropout", "context_layers", "hidden", "input_dropout")


def parse_source(spec: str) -> tuple[Path, Path]:
    """``MANIFEST@ROOT`` -> (manifest, root)."""
    if "@" not in spec:
        raise ValueError(f"expected MANIFEST@ROOT, got {spec!r}")
    manifest, root = spec.rsplit("@", 1)
    return Path(manifest), Path(root)


def read_rows(manifest: Path, root: Path) -> list[dict]:
    rows = []
    for line in manifest.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows.append({"clip_id": row["clip_id"], "video": str(root / row["video"]),
                         "frames": int(row["frames"]), "transcript": row["transcript"],
                         "phone_ids": [PHONE_TO_ID[phone] for phone in row["phonemes"]]})
    return rows


def prepare_frames(frames: np.ndarray, rng: random.Random | None) -> torch.Tensor:
    """uint8 [T,96,96] -> normalized [T,1,88,88]; random crop and flip when ``rng`` is given."""
    height, width = frames.shape[1:]
    if rng is None:
        top, left = (height - AUTOAVSR_CROP) // 2, (width - AUTOAVSR_CROP) // 2
    else:
        top, left = rng.randint(0, height - AUTOAVSR_CROP), rng.randint(0, width - AUTOAVSR_CROP)
    crop = frames[:, top:top + AUTOAVSR_CROP, left:left + AUTOAVSR_CROP]
    if rng is not None and rng.random() < 0.5:
        crop = crop[:, :, ::-1]
    video = torch.from_numpy(np.ascontiguousarray(crop)).float()
    return ((video / 255 - AUTOAVSR_MEAN) / AUTOAVSR_STD).unsqueeze(1)


class Clips(Dataset):
    def __init__(self, rows: list[dict], augment: bool, seed: int = 0):
        self.rows, self.augment, self.seed = rows, augment, seed
        self.rng = None  # created lazily so each loader worker gets its own seed

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        frames = np.load(row["video"], allow_pickle=False)
        if len(frames) != row["frames"]:
            raise ValueError(f"{row['clip_id']}: {len(frames)} frames, manifest says {row['frames']}")
        if self.augment and self.rng is None:
            self.rng = random.Random(self.seed * 1000003 + torch.initial_seed())
        return (prepare_frames(frames, self.rng if self.augment else None),
                torch.tensor(row["phone_ids"]), index)


def collate(items):
    videos, phones, indices = zip(*items)
    lengths = torch.tensor([len(video) for video in videos])
    batch = videos[0].new_zeros(len(videos), int(lengths.max()), *videos[0].shape[1:])
    for slot, video in enumerate(videos):
        batch[slot, :len(video)] = video
    return batch, lengths, torch.cat(phones), torch.tensor([len(p) for p in phones]), list(indices)


class WeightedBatches(Sampler):
    """Endless batches of global indices; each batch comes from one source.

    Sources are drawn by weight; within a source clips are visited in a fresh
    random order each pass, so every clip is seen before any repeats.
    """

    def __init__(self, sizes: list[int], weights: list[float], batch_size: int, seed: int):
        if len(sizes) != len(weights) or min(sizes) < batch_size:
            raise ValueError("one positive weight per source, each source >= batch size")
        self.sizes, self.weights, self.batch_size = sizes, weights, batch_size
        self.offsets = [sum(sizes[:i]) for i in range(len(sizes))]
        self.rng = random.Random(seed)

    def __iter__(self):
        orders = [self.rng.sample(range(size), size) for size in self.sizes]
        positions = [0] * len(self.sizes)
        while True:
            which = self.rng.choices(range(len(self.sizes)), self.weights)[0]
            if positions[which] + self.batch_size > self.sizes[which]:
                orders[which] = self.rng.sample(range(self.sizes[which]), self.sizes[which])
                positions[which] = 0
            start = positions[which]
            positions[which] += self.batch_size
            yield [self.offsets[which] + i for i in orders[which][start:start + self.batch_size]]


class PhoneFinetuner(nn.Module):
    """Frozen frontend and lower blocks, trainable top blocks and phone head.

    ``teacher`` is a frozen copy of the original top blocks for word-CTC
    distillation; the trunk below the split is computed once and shared.
    """

    def __init__(self, encoder: AutoAvsrEncoder, word_ctc: nn.Linear, head: PhoneHead,
                 trainable_blocks: int, distill: bool = True):
        super().__init__()
        if not 1 <= trainable_blocks <= len(encoder.encoders):
            raise ValueError("trainable_blocks must be between 1 and the number of blocks")
        self.encoder, self.word_ctc, self.head = encoder, word_ctc, head
        self.split = len(encoder.encoders) - trainable_blocks
        self.teacher = (nn.ModuleList([*copy.deepcopy(encoder.encoders[self.split:]),
                                       copy.deepcopy(encoder.after_norm)]) if distill else None)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for module in self.top_modules():
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        for parameter in head.parameters():
            parameter.requires_grad_(True)

    def top_modules(self) -> list[nn.Module]:
        return [*self.encoder.encoders[self.split:], self.encoder.after_norm]

    def train(self, mode: bool = True):
        """Frozen parts and every BatchNorm stay in eval mode (padded batches, running stats)."""
        super().train(False)
        if mode:
            for module in self.top_modules():
                module.train()
            self.head.train()
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def trunk(self, video: torch.Tensor, lengths: torch.Tensor):
        valid = torch.arange(video.shape[1], device=video.device)[None, :] < lengths[:, None]
        with torch.no_grad():
            video = video * valid[:, :, None, None, None].to(video.dtype)
            x, positions = self.encoder.embed[1](self.encoder.embed[0](self.encoder.frontend(video)))
            for layer in self.encoder.encoders[:self.split]:
                x = layer(x, positions, valid)
        return x, positions, valid

    def forward(self, video: torch.Tensor, lengths: torch.Tensor, teacher: bool = False):
        """Returns student features [B,T,768] and, if ``teacher``, frozen teacher features."""
        x, positions, valid = self.trunk(video, lengths)
        student = x
        for layer in self.encoder.encoders[self.split:]:
            student = layer(student, positions, valid)
        student = self.encoder.after_norm(student)
        reference = None
        if teacher and self.teacher is not None:
            with torch.no_grad():
                reference = x
                for layer in self.teacher[:-1]:
                    reference = layer(reference, positions, valid)
                reference = self.teacher[-1](reference)
        return student, reference

    def trainable_state(self) -> dict:
        state = {f"encoders.{self.split + i}.{key}": value
                 for i, module in enumerate(self.encoder.encoders[self.split:])
                 for key, value in module.state_dict().items()}
        state.update({f"after_norm.{key}": value
                      for key, value in self.encoder.after_norm.state_dict().items()})
        return state


def word_distillation(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                      valid: torch.Tensor) -> torch.Tensor:
    """Mean over valid frames of KL(teacher || student) on word-CTC posteriors."""
    teacher = teacher_logits.float().log_softmax(-1)
    student = student_logits.float().log_softmax(-1)
    kl = (teacher.exp() * (teacher - student)).sum(-1)
    return (kl * valid).sum() / valid.sum().clamp(min=1)


def load_finetuned_encoder(checkpoint: Path, base: Path | None = None) -> AutoAvsrEncoder:
    """Published encoder with a fine-tuned top loaded over it."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    encoder = load_autoavsr_ctc(Path(base or saved["base_checkpoint"])).encoder
    missing, unexpected = encoder.load_state_dict(saved["encoder_top"], strict=False)
    if unexpected or len(saved["encoder_top"]) == 0:
        raise ValueError(f"unexpected encoder keys: {unexpected}")
    return encoder


@torch.inference_mode()
def evaluate(model: PhoneFinetuner, rows: list[dict], device: torch.device, units: list[str],
             batch_size: int = 32) -> dict:
    model.eval()
    totals = {"phones": 0, "phone_edits": 0, "words": 0, "word_edits": 0,
              "blank_frames": 0, "emission_steps": 0}
    order = sorted(range(len(rows)), key=lambda i: rows[i]["frames"])
    dataset = Clips(rows, augment=False)
    for start in range(0, len(order), batch_size):
        video, lengths, _, _, indices = collate([dataset[i] for i in order[start:start + batch_size]])
        video, lengths = video.to(device), lengths.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            features, _ = model(video, lengths)
        features = features.float()
        phones = model.head(features, lengths).cpu()
        words = model.word_ctc(features).log_softmax(-1).cpu()
        for slot, index in enumerate(indices):
            row, frames = rows[index], int(lengths[slot])
            path = phones[slot, :frames * model.head.upsample]
            totals["phones"] += len(row["phone_ids"])
            totals["phone_edits"] += sum(edit_counts(row["phone_ids"], greedy_phones(path)))
            totals["blank_frames"] += int((path.argmax(-1) == 0).sum())
            totals["emission_steps"] += len(path)
            reference = normalize_words(row["transcript"])
            hypothesis = normalize_words(ctc_greedy_text(words[slot, :frames], units))
            totals["words"] += len(reference)
            totals["word_edits"] += sum(edit_counts(reference, hypothesis))
    model.train()
    return {**totals, "clips": len(rows), "greedy_per": totals["phone_edits"] / totals["phones"],
            "word_ctc_wer": totals["word_edits"] / totals["words"],
            "blank_rate": totals["blank_frames"] / totals["emission_steps"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/pretrained/autoavsr/LRS3_V_WER19.1/model.pth"))
    parser.add_argument("--units", type=Path,
                        default=Path("checkpoints/pretrained/autoavsr/tokens/unigram5000_units.txt"))
    parser.add_argument("--head", type=Path,
                        default=Path("checkpoints/autoavsr-phone-C-masked-20261106/best.pt"))
    parser.add_argument("--train", nargs="+", required=True, help="MANIFEST@ROOT training sources")
    parser.add_argument("--train-weights", type=float, nargs="+",
                        help="per-source sampling weights (default: source sizes)")
    parser.add_argument("--test", default="datasets/lrs3/manifests/test.jsonl@datasets/lrs3",
                        help="official test MANIFEST@ROOT; the first half selects")
    parser.add_argument("--select-clips", type=int, help="evaluate only the first N test-select clips")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trainable-blocks", type=int, default=4)
    parser.add_argument("--encoder-lr", type=float, default=3e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--distill-weight", type=float, default=0.3,
                        help="weight of the word-CTC KL to the frozen original (0 disables)")
    parser.add_argument("--time-masks", type=int, help="feature masks per clip (default: head's)")
    parser.add_argument("--time-mask-width", type=int, help="maximum masked span (default: head's)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--patience", type=int, default=0, help="stop after N evals without gain (0: off)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--max-minutes", type=float, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(filename)s:%(lineno)d %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(args.output / "train.log"),
                                  logging.StreamHandler()], force=True)
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    torch.manual_seed(args.seed)
    numpy_rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sources = [read_rows(*parse_source(spec)) for spec in args.train]
    weights = args.train_weights or [len(rows) for rows in sources]
    test_rows = read_rows(*parse_source(args.test))
    selection = test_rows[:len(test_rows) // 2][:args.select_clips]
    units = load_units(args.units)
    saved_head = torch.load(args.head, map_location="cpu", weights_only=True)
    head_config = {key: saved_head["config"].get(key) for key in HEAD_KEYS}
    masks = saved_head["config"].get("time_masks", 0) if args.time_masks is None else args.time_masks
    mask_width = (saved_head["config"].get("time_mask_width", 10) if args.time_mask_width is None
                  else args.time_mask_width)
    base = load_autoavsr_ctc(args.checkpoint)
    head = build_head(head_config, base.encoder.width)
    head.load_state_dict(saved_head["head"])
    model = PhoneFinetuner(base.encoder, base.ctc_lo, head, args.trainable_blocks,
                           distill=args.distill_weight > 0).to(device).train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config = {**{key: str(value) if isinstance(value, Path) else value
                 for key, value in vars(args).items()},
              **head_config, "time_masks": masks, "time_mask_width": mask_width,
              "train_weights": weights, "train_clips": [len(rows) for rows in sources],
              "test_select_clips": len(selection), "trainable_parameters": trainable,
              "test_split": "official test manifest order: first half selects; report not evaluated"}
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    LOGGER.info("Config: %s", config)

    optimizer = torch.optim.AdamW([
        {"params": [p for m in model.top_modules() for p in m.parameters()], "lr": args.encoder_lr},
        {"params": list(model.head.parameters()), "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    warmup = max(1, args.steps // 20)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(
        (step + 1) / warmup, 0.5 * (1 + math.cos(math.pi * min(step, args.steps) / args.steps))))
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    dataset = Clips([row for rows in sources for row in rows], augment=True, seed=args.seed)
    loader = DataLoader(dataset, batch_sampler=WeightedBatches(
        [len(rows) for rows in sources], weights, args.batch_size, args.seed),
        num_workers=args.workers, collate_fn=collate, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0)

    def save(name: str, step: int) -> None:
        cpu = lambda state: {key: value.detach().cpu() for key, value in state.items()}
        torch.save({"encoder_top": cpu(model.trainable_state()), "head": cpu(model.head.state_dict()),
                    "trainable_blocks": args.trainable_blocks,
                    "base_checkpoint": str(args.checkpoint), "step": step, "config": config,
                    "phones": PHONEMES}, args.output / name)

    history, best, stale, step = [], None, 0, 0
    started = time.monotonic()
    with ThermalGuard(args.output, device.type == "cuda"):
        initial = evaluate(model, selection, device, units)
        history.append({"step": 0, "test_select": initial, "seconds": time.monotonic() - started})
        LOGGER.info("step 0 (initial) test-select greedy PER %.2f%% word-CTC WER %.2f%%",
                    100 * initial["greedy_per"], 100 * initial["word_ctc_wer"])
        best = history[0]
        running = {"phone": 0.0, "distill": 0.0, "count": 0}
        batches = iter(loader)
        for step in range(1, args.steps + 1):
            if (stopping or Path("PAUSE").exists() or (args.output / "PAUSE").exists()
                    or time.monotonic() - started > args.max_minutes * 60):
                LOGGER.warning("Stopped at step %d before the planned %d; run is incomplete",
                               step - 1, args.steps)
                step -= 1
                break
            video, lengths, targets, target_lengths, _ = next(batches)
            video, lengths = video.to(device, non_blocking=True), lengths.to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                features, reference = model(video, lengths, teacher=args.distill_weight > 0)
                masked = time_mask(features, lengths, masks, mask_width, numpy_rng)
                phones = model.head(masked, lengths)
            phone_loss = ctc(phones.float().transpose(0, 1), targets, lengths * model.head.upsample,
                             target_lengths)
            loss = phone_loss
            distill = torch.zeros((), device=device)
            if reference is not None:
                valid = (torch.arange(features.shape[1], device=device)[None, :]
                         < lengths[:, None]).float()
                distill = word_distillation(model.word_ctc(features.float()),
                                            model.word_ctc(reference.float()), valid)
                loss = loss + args.distill_weight * distill
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            optimizer.step()
            scheduler.step()
            running["phone"] += phone_loss.item()
            running["distill"] += distill.item()
            running["count"] += 1
            if step % 50 == 0:
                LOGGER.info("step %d phone CTC %.3f distill %.4f (%.0fs)", step,
                            running["phone"] / running["count"],
                            running["distill"] / running["count"], time.monotonic() - started)
            if step % args.eval_every == 0 or step == args.steps:
                selected = evaluate(model, selection, device, units)
                record = {"step": step, "phone_loss": running["phone"] / max(1, running["count"]),
                          "distill": running["distill"] / max(1, running["count"]),
                          "test_select": selected, "seconds": time.monotonic() - started}
                running = {"phone": 0.0, "distill": 0.0, "count": 0}
                history.append(record)
                LOGGER.info("step %d test-select greedy PER %.2f%% word-CTC WER %.2f%% blank %.1f%%",
                            step, 100 * selected["greedy_per"], 100 * selected["word_ctc_wer"],
                            100 * selected["blank_rate"])
                if selected["greedy_per"] < best["test_select"]["greedy_per"]:
                    best, stale = record, 0
                    save("best.pt", step)
                else:
                    stale += 1
                    if args.patience and stale >= args.patience:
                        LOGGER.info("No test-select gain for %d evaluations; stopping", stale)
                        break
        save("last.pt", step)
    metrics = {"selected_step": best["step"], "steps_completed": step, "steps_planned": args.steps,
               "initial_test_select": initial, "best_test_select": best["test_select"],
               "history": history, "seconds": time.monotonic() - started,
               "finished_utc": datetime.now(timezone.utc).isoformat()}
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    LOGGER.info("Selected step %d: test-select greedy PER %.2f%% (initial %.2f%%), word-CTC WER "
                "%.2f%% (initial %.2f%%)%s", best["step"], 100 * best["test_select"]["greedy_per"],
                100 * initial["greedy_per"], 100 * best["test_select"]["word_ctc_wer"],
                100 * initial["word_ctc_wer"],
                "" if best["step"] else "; no improvement, best.pt not written")


if __name__ == "__main__":
    main()
