"""Train a phone CTC head on cached, frozen Auto-AVSR encoder features.

Phase 2 of FUTURE_TRAINING_PLAN.md. Auto-AVSR trained on all of LRS3 trainval
and pretrain, so checkpoints are selected on the first half of the official
test set (``test-select``) and reported once on the second half
(``test-report``). Validation numbers use clips the encoder has seen and are
optimistic; they are reported only for continuity with the old 1,140-clip
development results.
"""
import argparse
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import signal
import time

import numpy as np
import torch
from torch import nn

from VisualPhoneme.data import PHONEMES, ctc_prefix_beam_search
from VisualPhoneme.evaluate_autoavsr_ctc import edit_counts
from VisualPhoneme.thermal import ThermalGuard

LOGGER = logging.getLogger(__name__)
CLASSES = len(PHONEMES) + 1


class PhoneHead(nn.Module):
    """Head emitting ``upsample`` CTC steps per video frame.

    ``layers=0`` is a per-frame linear head; otherwise a small pre-norm
    transformer (masked over padding) adds temporal context first.
    """

    def __init__(self, width: int = 768, upsample: int = 2, dropout: float = 0.1,
                 layers: int = 0, hidden: int = 256):
        super().__init__()
        self.upsample, self.layers = upsample, layers
        self.dropout = nn.Dropout(dropout)
        if layers:
            self.project = nn.Linear(width, hidden)
            self.context = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(hidden, 4, 4 * hidden, dropout, batch_first=True,
                                           norm_first=True),
                layers, enable_nested_tensor=False)
            width = hidden
        self.linear = nn.Linear(width, upsample * CLASSES)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """[batch,time,width], frame lengths -> [batch,time*upsample,classes] log probs."""
        x = self.dropout(features)
        if self.layers:
            padding = (torch.arange(x.shape[1], device=x.device)[None, :]
                       >= lengths.to(x.device)[:, None])
            x = self.context(self.project(x), src_key_padding_mask=padding)
        logits = self.linear(x)
        return logits.reshape(features.shape[0], -1, CLASSES).log_softmax(-1)


class Split:
    """Cached features of one split, held on the training device."""

    def __init__(self, directory: Path, device: torch.device, rows: slice = slice(None)):
        self.rows = [json.loads(line) for line in
                     (directory / "index.jsonl").read_text().splitlines()][rows]
        self.features = torch.from_numpy(np.load(directory / "features.npy")).to(device)

    def batch(self, indices: list[int], upsample: int):
        rows = [self.rows[index] for index in indices]
        lengths = torch.tensor([row["frames"] for row in rows])
        features = self.features.new_zeros(len(rows), int(lengths.max()), self.features.shape[1])
        for slot, row in enumerate(rows):
            features[slot, :row["frames"]] = self.features[row["offset"]:row["offset"] + row["frames"]]
        targets = torch.tensor([phone for row in rows for phone in row["phone_ids"]])
        target_lengths = torch.tensor([len(row["phone_ids"]) for row in rows])
        return features.float(), lengths * upsample, targets, target_lengths


def greedy_phones(log_probabilities: torch.Tensor) -> list[int]:
    output, previous = [], None
    for index in log_probabilities.argmax(-1).tolist():
        if index != previous and index != 0:
            output.append(index)
        previous = index
    return output


@torch.inference_mode()
def evaluate(head: PhoneHead, split: Split, indices: list[int], beam: bool = False,
             hypotheses_path: Path | None = None) -> dict:
    head.eval()
    totals = {"phones": 0, "substitutions": 0, "deletions": 0, "insertions": 0,
              "predicted_phones": 0, "blank_frames": 0, "emission_steps": 0}
    beam_edits = oracle_edits = 0
    handle = hypotheses_path.open("w") if hypotheses_path else None
    for start in range(0, len(indices), 64):
        chunk = indices[start:start + 64]
        features, lengths, _, _ = split.batch(chunk, head.upsample)
        log_probabilities = head(features, lengths // head.upsample).cpu()
        for slot, index in enumerate(chunk):
            row = split.rows[index]
            path = log_probabilities[slot, :int(lengths[slot])]
            hypothesis = greedy_phones(path)
            substitutions, deletions, insertions = edit_counts(row["phone_ids"], hypothesis)
            totals["phones"] += len(row["phone_ids"])
            totals["substitutions"] += substitutions
            totals["deletions"] += deletions
            totals["insertions"] += insertions
            totals["predicted_phones"] += len(hypothesis)
            totals["blank_frames"] += int((path.argmax(-1) == 0).sum())
            totals["emission_steps"] += len(path)
            if beam:
                candidates = ctc_prefix_beam_search(path.double(), beam_width=64, top_n=5,
                                                    token_top_k=12)
                errors = [sum(edit_counts(row["phone_ids"], sequence)) for sequence, _ in candidates]
                beam_edits += errors[0]
                oracle_edits += min(errors)
                if handle:
                    handle.write(json.dumps({
                        "clip_id": row["clip_id"],
                        "reference_phones": [PHONEMES[i - 1] for i in row["phone_ids"]],
                        "reference_words": row["transcript"].split(),
                        "phone_hypotheses": [
                            {"rank": rank, "phones": [PHONEMES[i - 1] for i in sequence],
                             "ctc_log_score": score}
                            for rank, (sequence, score) in enumerate(candidates, 1)],
                    }) + "\n")
    if handle:
        handle.close()
    head.train()
    edits = totals["substitutions"] + totals["deletions"] + totals["insertions"]
    result = {**totals, "clips": len(indices), "greedy_per": edits / totals["phones"],
              "blank_rate": totals["blank_frames"] / totals["emission_steps"]}
    if beam:
        result.update(beam_per=beam_edits / totals["phones"],
                      oracle_per_at_5=oracle_edits / totals["phones"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("checkpoints/autoavsr-features-20261102"))
    parser.add_argument("--validation-manifest", type=Path,
                        default=Path("datasets/lrs3/manifests/autoavsr/validation.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upsample", type=int, choices=(1, 2), default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--context-layers", type=int, default=0,
                        help="transformer layers before the phone projection (0: linear head)")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--max-minutes", type=float, default=60)
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
    generator = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = Split(args.features / "train", device)
    test = Split(args.features / "test", device)
    validation = Split(args.features / "validation", device)
    half = len(test.rows) // 2
    selection, report = list(range(half)), list(range(half, len(test.rows)))
    coverage = {json.loads(line)["clip_id"]: json.loads(line)["pronunciation_coverage"]
                for line in args.validation_manifest.read_text().splitlines()}
    eligible = [i for i, row in enumerate(validation.rows) if coverage[row["clip_id"]] >= 0.95]
    config = {**{key: str(value) if isinstance(value, Path) else value
                 for key, value in vars(args).items()},
              "train_clips": len(train.rows), "test_select_clips": len(selection),
              "test_report_clips": len(report),
              "test_split": "official test manifest order: first half selects, second half reports",
              "validation_eligible_clips": len(eligible),
              "validation_eligible_phones": sum(len(validation.rows[i]["phone_ids"]) for i in eligible)}
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    LOGGER.info("Config: %s", config)

    head = PhoneHead(train.features.shape[1], args.upsample, args.dropout,
                     args.context_layers).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    warmup = max(1, args.steps // 20)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(
        (step + 1) / warmup, 0.5 * (1 + math.cos(math.pi * min(step, args.steps) / args.steps))))
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    history, best = [], None
    started = time.monotonic()
    with ThermalGuard(args.output, device.type == "cuda"):
        for step in range(1, args.steps + 1):
            if (stopping or Path("PAUSE").exists() or (args.output / "PAUSE").exists()
                    or time.monotonic() - started > args.max_minutes * 60):
                LOGGER.warning("Stopped at step %d before the planned %d; run is incomplete",
                               step - 1, args.steps)
                break
            indices = generator.choice(len(train.rows), args.batch_size, replace=False).tolist()
            features, lengths, targets, target_lengths = train.batch(indices, args.upsample)
            log_probabilities = head(features, lengths // args.upsample)
            loss = ctc(log_probabilities.transpose(0, 1), targets, lengths, target_lengths)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            if step % args.eval_every == 0 or step == args.steps:
                selected = evaluate(head, test, selection)
                record = {"step": step, "loss": loss.item(), "test_select": selected,
                          "seconds": time.monotonic() - started}
                history.append(record)
                LOGGER.info("step %d loss %.3f test-select greedy PER %.2f%% blank %.1f%%",
                            step, loss.item(), 100 * selected["greedy_per"],
                            100 * selected["blank_rate"])
                if best is None or selected["greedy_per"] < best["test_select"]["greedy_per"]:
                    best = record
                    torch.save({"head": head.state_dict(), "step": step, "config": config,
                                "phones": PHONEMES}, args.output / "best.pt")
        torch.save({"head": head.state_dict(), "step": step, "config": config,
                    "phones": PHONEMES}, args.output / "last.pt")
        if best is None:
            raise RuntimeError("no evaluation was completed")
        head.load_state_dict(torch.load(args.output / "best.pt", weights_only=True)["head"])
        LOGGER.info("Selected step %d; evaluating test-report and validation once", best["step"])
        final = {
            "selected_step": best["step"],
            "steps_completed": step, "steps_planned": args.steps,
            "test_select": evaluate(head, test, selection),
            "test_report": evaluate(head, test, report, beam=True,
                                    hypotheses_path=args.output / "test-report-phone-hypotheses.jsonl"),
            "validation_eligible_seen_by_encoder": evaluate(
                head, validation, eligible, beam=True,
                hypotheses_path=args.output / "validation-phone-hypotheses.jsonl"),
            "history": history,
            "seconds": time.monotonic() - started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        }
    (args.output / "metrics.json").write_text(json.dumps(final, indent=2) + "\n")
    for name in ("test_select", "test_report", "validation_eligible_seen_by_encoder"):
        result = final[name]
        LOGGER.info("%s: greedy PER %.2f%%%s", name, 100 * result["greedy_per"],
                    "" if "beam_per" not in result else
                    f", beam PER {100 * result['beam_per']:.2f}%, oracle PER@5 "
                    f"{100 * result['oracle_per_at_5']:.2f}%")


if __name__ == "__main__":
    main()
