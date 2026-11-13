"""Add real phone errors from encoder-unseen video to the decoder's training rows.

The matched-noise decoder (``matched_phone_noise.py``) learned from LRS3 phone
strings with synthetic errors. Seamless Interaction clips were never seen by
Auto-AVSR, so head C's greedy phones on them carry real visual errors
(~38% PER). This writes a ``--prepared-rows`` directory for
``train_vallr_clean.py``: the base rows plus one row per Seamless clip
(prompt = greedy phone string, target = automatic transcript).
"""
import argparse
import hashlib
import json
from pathlib import Path

from CompetitiveWordDecoder.evaluate_vallr_qwen import vallr_prompt


def seamless_rows(posteriors: Path, min_phones: int) -> list[dict]:
    rows = []
    for line in (posteriors / "index.jsonl").read_text().splitlines():
        item = json.loads(line)
        if len(item["greedy_phones"]) < min_phones:
            continue
        rows.append({"prompt": vallr_prompt(item["greedy_phones"]), "target": item["transcript"],
                     "unknown_words": 0, "clip_id": item["clip_id"], "input_kind": "seamless_real_head_c"})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("datasets/decoder-training/matched-noise-20261102"))
    parser.add_argument("--posteriors", type=Path, nargs="+", required=True,
                        help="export_phone_posteriors output directories (head C on Seamless)")
    parser.add_argument("--min-phones", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base = (args.base / "train.jsonl").read_text().splitlines()
    added = [row for path in args.posteriors for row in seamless_rows(path, args.min_phones)]
    train = base + [json.dumps(row) for row in added]
    (args.output / "train.jsonl").write_text("\n".join(train) + "\n")
    (args.output / "validation.jsonl").write_text((args.base / "validation.jsonl").read_text())
    metadata = {"version": "matched-noise-plus-seamless-v1", "base": str(args.base),
                "base_metadata": json.loads((args.base / "metadata.json").read_text()),
                "posteriors": [str(path) for path in args.posteriors], "min_phones": args.min_phones,
                "train": {"base_rows": len(base), "seamless_rows": len(added), "rows": len(train)},
                "train_sha256": hashlib.sha256((args.output / "train.jsonl").read_bytes()).hexdigest()}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata["train"]))


if __name__ == "__main__":
    main()
