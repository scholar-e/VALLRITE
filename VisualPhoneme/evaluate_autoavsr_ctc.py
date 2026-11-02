"""Reproduce published Auto-AVSR visual WER on LRS3 mouth crops.

Phase 1 gate of FUTURE_TRAINING_PLAN.md. ``--decoding greedy`` uses only the
auxiliary CTC head; ``--decoding joint`` uses the attention decoder with CTC
prefix scoring (published setting: beam 40, CTC weight 0.1, plus an external
language model that is not ported). Words are normalized as in the existing
decoder evaluators: lowercase ``[a-z]+(?:'[a-z]+)*``.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
import signal
import time

import numpy as np
import torch

from VisualPhoneme.autoavsr_decoding import joint_beam_search, load_autoavsr_joint
from VisualPhoneme.autoavsr_encoder import (AUTOAVSR_CROP, AUTOAVSR_MEAN, AUTOAVSR_STD,
                                            autoavsr_video_transform, ctc_greedy_text,
                                            load_autoavsr_ctc, load_units)
from VisualPhoneme.thermal import ThermalGuard

LOGGER = logging.getLogger(__name__)
WORD = re.compile(r"[a-z]+(?:'[a-z]+)*")


def normalize_words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def edit_counts(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int]:
    """Return (substitutions, deletions, insertions) of one minimum-edit alignment."""
    previous = [(0, 0, column) for column in range(len(hypothesis) + 1)]
    for row, word in enumerate(reference, 1):
        current = [(0, row, 0)]
        for column, other in enumerate(hypothesis, 1):
            if word == other:
                current.append(previous[column - 1])
                continue
            substitution, deletion, insertion = (previous[column - 1], previous[column],
                                                 current[column - 1])
            current.append(min(
                (substitution[0] + 1, substitution[1], substitution[2]),
                (deletion[0], deletion[1] + 1, deletion[2]),
                (insertion[0], insertion[1], insertion[2] + 1),
                key=lambda counts: (sum(counts), counts[1], counts[2], counts[0]),
            ))
        previous = current
    return previous[-1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/pretrained/autoavsr/LRS3_V_WER19.1/model.pth"))
    parser.add_argument("--units", type=Path,
                        default=Path("checkpoints/pretrained/autoavsr/tokens/unigram5000_units.txt"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("datasets/lrs3/manifests/test.jsonl"))
    parser.add_argument("--data-root", type=Path, default=Path("datasets/lrs3"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decoding", choices=("greedy", "joint"), default="greedy")
    parser.add_argument("--beam", type=int, default=40)
    parser.add_argument("--ctc-weight", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, help="evaluate only the first N manifest rows")
    parser.add_argument("--max-minutes", type=float, default=30)
    args = parser.parse_args()
    if (args.batch_size < 1 or args.beam < 1 or (args.limit is not None and args.limit < 1)
            or not 0 <= args.ctc_weight <= 1):
        parser.error("--batch-size, --beam and --limit must be positive; --ctc-weight in [0, 1]")
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(filename)s:%(lineno)d %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.output / "run.log"), logging.StreamHandler()],
        force=True,
    )
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    started = time.monotonic()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    rows = rows[:args.limit] if args.limit else rows
    config = {
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
        "units": str(args.units), "units_sha256": sha256(args.units),
        "manifest": str(args.manifest), "manifest_sha256": sha256(args.manifest),
        "clips": len(rows),
        "decoding": ("greedy CTC, blank and <eos> removed" if args.decoding == "greedy" else
                     {"method": "joint CTC/attention beam search, no language model",
                      "beam": args.beam, "ctc_weight": args.ctc_weight,
                      "ctc_pre_beam": int(1.5 * args.beam)}),
        "preprocessing": {"centre_crop": AUTOAVSR_CROP, "scale": "1/255",
                          "mean": AUTOAVSR_MEAN, "std": AUTOAVSR_STD},
        "normalization": "lowercase [a-z]+(?:'[a-z]+)*",
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Evaluating %d clips on %s", len(rows), device)
    units = load_units(args.units)
    if args.decoding == "joint":
        model, decoder = load_autoavsr_joint(args.checkpoint)
        decoder = decoder.to(device).eval()
    else:
        model = load_autoavsr_ctc(args.checkpoint)
    model = model.to(device).eval()
    if len(units) != model.ctc_lo.out_features:
        raise ValueError(f"{len(units)} units for {model.ctc_lo.out_features} CTC outputs")
    # Sort by length so padding stays small; outputs are written in manifest order.
    order = sorted(range(len(rows)), key=lambda index: rows[index]["frames"])
    results: dict[int, dict] = {}
    totals = {"words": 0, "substitutions": 0, "deletions": 0, "insertions": 0,
              "exact": 0, "empty_hypotheses": 0}
    with ThermalGuard(args.output, device.type == "cuda"), torch.inference_mode():
        for start in range(0, len(order), args.batch_size):
            if (stopping or Path("PAUSE").exists() or (args.output / "PAUSE").exists()
                    or time.monotonic() - started > args.max_minutes * 60):
                LOGGER.warning("Stopped after %d/%d clips; metrics are incomplete",
                               len(results), len(rows))
                break
            indices = order[start:start + args.batch_size]
            videos = [autoavsr_video_transform(torch.from_numpy(
                np.load(args.data_root / rows[index]["video"], allow_pickle=False)))
                for index in indices]
            lengths = torch.tensor([len(video) for video in videos])
            batch = videos[0].new_zeros(len(videos), int(lengths.max()), *videos[0].shape[1:])
            for slot, video in enumerate(videos):
                batch[slot, :len(video)] = video
            # Full precision: this is a reproduction check, not a speed benchmark.
            memory = model.encoder(batch.to(device), lengths.to(device))
            log_probabilities = model.ctc_lo(memory).log_softmax(-1)
            for slot, index in enumerate(indices):
                length = int(lengths[slot])
                if args.decoding == "greedy":
                    hypothesis_text = ctc_greedy_text(log_probabilities[slot, :length].cpu(), units)
                else:
                    ids = joint_beam_search(log_probabilities[slot, :length], memory[slot, :length],
                                            decoder, args.beam, args.ctc_weight)
                    hypothesis_text = "".join(units[i] for i in ids).replace("\u2581", " ")
                reference = normalize_words(rows[index]["transcript"])
                hypothesis = normalize_words(hypothesis_text)
                substitutions, deletions, insertions = edit_counts(reference, hypothesis)
                results[index] = {"clip_id": rows[index]["clip_id"],
                                  "reference": " ".join(reference),
                                  "hypothesis": " ".join(hypothesis),
                                  "substitutions": substitutions, "deletions": deletions,
                                  "insertions": insertions}
                totals["words"] += len(reference)
                totals["substitutions"] += substitutions
                totals["deletions"] += deletions
                totals["insertions"] += insertions
                totals["exact"] += reference == hypothesis
                totals["empty_hypotheses"] += not hypothesis
            if start // args.batch_size % 20 == 0:
                LOGGER.info("%d/%d clips", len(results), len(rows))

    with (args.output / "hypotheses.jsonl").open("w") as handle:
        for index in sorted(results):
            handle.write(json.dumps(results[index]) + "\n")
    edits = totals["substitutions"] + totals["deletions"] + totals["insertions"]
    metrics = {
        **totals, "clips_evaluated": len(results), "clips_requested": len(rows),
        "complete": len(results) == len(rows), "edits": edits,
        "wer": edits / totals["words"] if totals["words"] else None,
        "seconds": time.monotonic() - started, "device": str(device),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    LOGGER.info("WER %.2f%% over %d words (%d/%d clips, S=%d D=%d I=%d)",
                100 * (metrics["wer"] or 0), totals["words"], len(results), len(rows),
                totals["substitutions"], totals["deletions"], totals["insertions"])


if __name__ == "__main__":
    main()
