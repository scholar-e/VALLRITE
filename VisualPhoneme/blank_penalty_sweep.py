"""Sweep a CTC blank penalty for a cached-feature phone head (head C by default).

Subtracting ``b`` from the blank log probability at decode time makes the head
emit borderline phones it would otherwise skip (head C deletes ~3.7 phones per
insertion). Greedy PER is swept on test-select, the best penalties are checked
with the same beam search as the export, and test-report is scored once at the
penalty selected on test-select beam PER. No training.
"""
import argparse
import json
import logging
from pathlib import Path

import torch

from VisualPhoneme.data import ctc_prefix_beam_search
from VisualPhoneme.evaluate_autoavsr_ctc import edit_counts
from VisualPhoneme.thermal import ThermalGuard
from VisualPhoneme.train_autoavsr_phone_head import Split, build_head, greedy_phones

LOGGER = logging.getLogger(__name__)


def score(paths: list[torch.Tensor], references: list[list[int]], penalty: float, beam: bool) -> dict:
    totals = {"phones": 0, "substitutions": 0, "deletions": 0, "insertions": 0}
    for path, reference in zip(paths, references):
        path = path.clone()
        path[:, 0] -= penalty
        hypothesis = (ctc_prefix_beam_search(path.double(), beam_width=64, top_n=1, token_top_k=12)[0][0]
                      if beam else greedy_phones(path))
        s, d, i = edit_counts(reference, hypothesis)
        totals["phones"] += len(reference)
        totals.update(substitutions=totals["substitutions"] + s, deletions=totals["deletions"] + d,
                      insertions=totals["insertions"] + i)
    errors = totals["substitutions"] + totals["deletions"] + totals["insertions"]
    return {**totals, "penalty": penalty, "per": errors / totals["phones"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", type=Path,
                        default=Path("checkpoints/autoavsr-phone-C-masked-20261106/best.pt"))
    parser.add_argument("--features", type=Path, default=Path("checkpoints/autoavsr-features-20261102/test"))
    parser.add_argument("--penalties", type=float, nargs="+",
                        default=[0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0])
    parser.add_argument("--beam-top", type=int, default=3, help="best greedy penalties checked with beam")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(filename)s:%(lineno)d %(message)s",
                        handlers=[logging.FileHandler(args.output / "sweep.log"), logging.StreamHandler()],
                        force=True)
    with ThermalGuard(args.output, False), torch.inference_mode():
        saved = torch.load(args.head, map_location="cpu", weights_only=True)
        test = Split(args.features, torch.device("cpu"))
        head = build_head(saved["config"], test.features.shape[1])
        head.load_state_dict(saved["head"])
        head.eval()
        paths, references = [], []
        for start in range(0, len(test.rows), 64):
            chunk = list(range(start, min(start + 64, len(test.rows))))
            features, lengths, _, _ = test.batch(chunk, head.upsample)
            log_probabilities = head(features, lengths // head.upsample)
            paths += [log_probabilities[slot, :int(lengths[slot])] for slot in range(len(chunk))]
            references += [test.rows[index]["phone_ids"] for index in chunk]
        half = len(paths) // 2
        select, report = (paths[:half], references[:half]), (paths[half:], references[half:])
        greedy = [score(*select, penalty, beam=False) for penalty in args.penalties]
        for result in greedy:
            LOGGER.info("test-select greedy  b=%.2f PER %.2f%% (S %d D %d I %d)", result["penalty"],
                        100 * result["per"], result["substitutions"], result["deletions"], result["insertions"])
        candidates = sorted({0.0, *[r["penalty"] for r in sorted(greedy, key=lambda r: r["per"])[:args.beam_top]]})
        beam = [score(*select, penalty, beam=True) for penalty in candidates]
        for result in beam:
            LOGGER.info("test-select beam    b=%.2f PER %.2f%% (S %d D %d I %d)", result["penalty"],
                        100 * result["per"], result["substitutions"], result["deletions"], result["insertions"])
        chosen = min(beam, key=lambda r: (r["per"], r["penalty"]))["penalty"]
        final = {f"{kind}_{penalty}": score(*report, penalty, beam=kind == "beam")
                 for kind in ("greedy", "beam") for penalty in sorted({0.0, chosen})}
        for name, result in final.items():
            LOGGER.info("test-report %-10s PER %.2f%% (S %d D %d I %d)", name, 100 * result["per"],
                        result["substitutions"], result["deletions"], result["insertions"])
    (args.output / "metrics.json").write_text(json.dumps(
        {"head": str(args.head), "selected_penalty": chosen, "test_select_greedy": greedy,
         "test_select_beam": beam, "test_report": final}, indent=2) + "\n")


if __name__ == "__main__":
    main()
