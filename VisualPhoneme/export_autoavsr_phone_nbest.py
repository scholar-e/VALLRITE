"""Export five-best phone hypotheses from a trained Auto-AVSR phone head.

Writes the word-decoder input format (``phone-hypotheses.jsonl``) and PER
metrics for one cached feature split, optionally restricted to a row range
(the official test set is split into a selection half and a report half).
"""
import argparse
import json
from pathlib import Path

import torch

from VisualPhoneme.train_autoavsr_phone_head import PhoneHead, Split, evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", type=Path, required=True, help="best.pt from the trainer")
    parser.add_argument("--features", type=Path, required=True, help="cached split directory")
    parser.add_argument("--rows", default=":", help="python slice over split rows, e.g. :660")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.head, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    split = Split(args.features, device)
    start, stop = (int(part) if part else None for part in args.rows.split(":"))
    indices = list(range(len(split.rows)))[start:stop]
    head = PhoneHead(split.features.shape[1], config["upsample"], config["dropout"],
                     config.get("context_layers", 0)).to(device)
    head.load_state_dict(checkpoint["head"])
    metrics = evaluate(head, split, indices, beam=True,
                       hypotheses_path=args.output / "phone-hypotheses.jsonl")
    metrics.update(head=str(args.head), head_step=checkpoint["step"], features=str(args.features),
                   rows=args.rows)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({key: metrics[key] for key in ("clips", "greedy_per", "beam_per",
                                                    "oracle_per_at_5")}))


if __name__ == "__main__":
    main()
