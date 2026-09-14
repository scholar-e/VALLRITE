"""Compute decoder WER from reusable candidate-model score records.

Any phoneme-to-word reranker can be evaluated by emitting JSONL objects with
``clip_id``, ``candidate_index``, ``candidate_text``, and finite ``score``.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, expected in enumerate(reference, 1):
        current = [i]
        for j, actual in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (expected != actual)))
        previous = current
    return previous[-1]


def load_scores(path: Path):
    scores = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        row = json.loads(line)
        key = (row["clip_id"], row["candidate_index"])
        if key in scores or not math.isfinite(row["score"]):
            raise ValueError(f"Invalid or duplicate score at {path}:{number}")
        scores[key] = (row["candidate_text"], row["score"])
    return scores


def evaluate(records, scores, weights, visual_score_window):
    total_words = sum(len(row["reference_words"]) for row in records)
    baseline_errors = oracle_errors = 0
    selected = []
    for row in records:
        candidates = row["candidates"]
        if not candidates:
            baseline_errors += len(row["reference_words"])
            oracle_errors += len(row["reference_words"])
            selected.append([])
            continue
        errors = [edit_distance(row["reference_words"], c["words"])
                  for c in candidates]
        baseline_errors += errors[0]
        oracle_errors += min(errors)
        cutoff = candidates[0]["log_score"] - visual_score_window
        choices = [i for i, candidate in enumerate(candidates)
                   if candidate["log_score"] >= cutoff]
        for index in choices:
            key = (row["clip_id"], index)
            if key not in scores:
                raise ValueError(f"Missing score for {key}")
            text, _ = scores[key]
            if text != candidates[index]["text"]:
                raise ValueError(f"Candidate mismatch for {key}")
        selected.append(choices)
    results = {}
    for weight in weights:
        errors = changes = 0
        for row, choices in zip(records, selected, strict=True):
            if not choices:
                errors += len(row["reference_words"])
                continue
            winner = max(choices, key=lambda index: (
                row["candidates"][index]["log_score"]
                + weight * scores[(row["clip_id"], index)][1], -index))
            changes += winner != 0
            errors += edit_distance(row["reference_words"],
                                    row["candidates"][winner]["words"])
        results[str(weight)] = {"word_errors": errors, "wer": errors / total_words,
                                "winner_changes": changes}
    return {"records": len(records), "reference_words": total_words,
            "baseline": {"word_errors": baseline_errors,
                         "wer": baseline_errors / total_words},
            "oracle_at_n": {"word_errors": oracle_errors,
                            "wer": oracle_errors / total_words},
            "visual_score_window": visual_score_window,
            "results_by_reranker_weight": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visual-score-window", type=float, default=5.0)
    parser.add_argument("--weights", type=float, nargs="+",
                        default=[0.0, 0.1, 0.25, 0.5, 1.0, 2.0])
    args = parser.parse_args()
    records = [json.loads(line) for line in args.examples.read_text().splitlines()]
    result = evaluate(records, load_scores(args.scores), args.weights,
                      args.visual_score_window)
    result.update({"examples": str(args.examples.resolve()),
                   "scores": str(args.scores.resolve())})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
