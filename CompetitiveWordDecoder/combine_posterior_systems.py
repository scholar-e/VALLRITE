"""Combine posterior-fused decoding with 20-best posterior rescoring.

Per clip, choose between the fused decoder's top sentence and the rescored
20-best winner using only terms both share: CTC log-likelihood of the
pronunciation, word count and unknown words, with a margin favouring one
system. The rule is fitted on test-select and applied once to test-report.
A paired clip bootstrap compares the result with the previous selected system
(matched-noise decoder + existing ranker over five candidates).
"""
import argparse
import itertools
import json
from pathlib import Path
import random

from CompetitiveWordDecoder.posterior_rescoring import attach, load_clips, phone_posteriors, select
from CompetitiveWordDecoder.train_vallr_clean import edit_distance
from PhonemeDecoder.evaluate_qwen_direct import words

RANKER = {"ctc": 0, "ranker": 1, "words": 0, "unknown": 0}


def half_records(name: str, args, posteriors, weights) -> list[dict]:
    first = Path(args.first_block.format(name=name))
    hypotheses = args.pool_root / name / "phone-hypotheses.jsonl"
    pool = attach(load_clips([first] + [args.pool_root / name / f"decode{i}/ranked/scores.jsonl"
                                        for i in (1, 2, 3)]), hypotheses, posteriors)
    five = {clip["clip_id"]: clip for clip in attach(load_clips([first]), hypotheses, posteriors)}
    fused = {}
    for line in (args.fused_root / f"{name}-w0.1/nbest.jsonl").read_text().splitlines():
        row = json.loads(line)
        fused[row["clip_id"]] = row["nbest"][0]
    records = []
    for clip in pool:
        winner = select(clip["groups"], weights)
        top = fused[clip["clip_id"]]
        records.append({
            "clip_id": clip["clip_id"], "reference_words": len(clip["reference"]),
            "previous_errors": select(five[clip["clip_id"]]["groups"], RANKER)["errors"],
            "pool": {"text": " ".join(winner["words"]), "ctc": winner["ctc"],
                     "words": winner["words_count"], "unknown": winner["unknown"],
                     "errors": winner["errors"]},
            "fused": {"text": top["text"], "ctc": top["ctc"], "words": top["words"],
                      "unknown": top["unknown"],
                      "errors": edit_distance(clip["reference"], words(top["text"]))},
        })
    return records


def choose(record: dict, rule: tuple) -> dict:
    word_weight, unknown_weight, margin = rule
    def score(item):
        return item["ctc"] + word_weight * item["words"] + unknown_weight * item["unknown"]
    return record["fused"] if score(record["fused"]) > score(record["pool"]) + margin else record["pool"]


GRID = ((-2, -1, 0, 1, 2), (-20, -5, 0), (-4, -2, -1, 0, 1, 2, 4))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", type=Path,
                        default=Path("checkpoints/autoavsr-phone-context2-20261102/best.pt"))
    parser.add_argument("--features", type=Path,
                        default=Path("checkpoints/autoavsr-features-20261102/test"))
    parser.add_argument("--first-block",
                        default="checkpoints/phase3-autoavsr-phones-20261102/{name}-matched-noise/ranked/scores.jsonl",
                        help="ranker scores of phone paths 1-5; {name} is select or report")
    parser.add_argument("--pool-root", type=Path,
                        default=Path("checkpoints/posterior-rescoring-20261105/phones20"))
    parser.add_argument("--rescoring-metrics", type=Path,
                        default=Path("checkpoints/posterior-rescoring-20261105/matched-noise-20best/metrics.json"))
    parser.add_argument("--fused-root", type=Path, default=Path("checkpoints/posterior-fused-20261105"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    weights = json.loads(args.rescoring_metrics.read_text())["weights"]
    posteriors = phone_posteriors(args.head, args.features)
    halves = {name: half_records(name, args, posteriors, weights) for name in ("select", "report")}
    rule = min(itertools.product(*GRID),
               key=lambda r: sum(choose(x, r)["errors"] for x in halves["select"]))
    result = {"rule": dict(zip(("word_weight", "unknown_weight", "margin"), rule)),
              "rescoring_weights": weights}
    for name, records in halves.items():
        count = sum(r["reference_words"] for r in records)
        totals = {"previous_selected": sum(r["previous_errors"] for r in records),
                  "rescored_20_best": sum(r["pool"]["errors"] for r in records),
                  "fused": sum(r["fused"]["errors"] for r in records),
                  "combined": sum(choose(r, rule)["errors"] for r in records),
                  "oracle_of_two": sum(min(r["pool"]["errors"], r["fused"]["errors"]) for r in records)}
        result[f"test_{name}"] = {"clips": len(records), "reference_words": count,
                                  **{k: {"word_errors": v, "wer": v / count} for k, v in totals.items()}}
    report = halves["report"]
    rng = random.Random(20261105)
    differences = []
    for _ in range(10000):
        sample = [report[rng.randrange(len(report))] for _ in report]
        words_total = sum(r["reference_words"] for r in sample)
        differences.append(sum(r["previous_errors"] - choose(r, rule)["errors"] for r in sample)
                           / words_total)
    differences.sort()
    result["test_report_bootstrap_wer_reduction"] = {
        "resamples": 10000, "unit": "clip", "interval_95": [differences[249], differences[9749]],
        "fraction_not_better": sum(d <= 0 for d in differences) / len(differences)}
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output / "test-report-selected.jsonl").open("w") as handle:
        for record in report:
            chosen = choose(record, rule)
            handle.write(json.dumps({"clip_id": record["clip_id"], "text": chosen["text"],
                                     "system": "fused" if chosen is record["fused"] else "rescored"}) + "\n")
    for name in ("test_select", "test_report"):
        print(name, {k: round(100 * v["wer"], 2) for k, v in result[name].items() if isinstance(v, dict)})
    print("rule", result["rule"], "bootstrap", result["test_report_bootstrap_wer_reduction"])


if __name__ == "__main__":
    main()
