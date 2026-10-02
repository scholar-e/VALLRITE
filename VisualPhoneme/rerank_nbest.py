"""Fit and cross-validate a lightweight linear ranker for phoneme N-best lists."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys

import torch

LOGGER = logging.getLogger("visual_phoneme.rerank")


def configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.handlers.clear()
    for handler in (logging.FileHandler(path, delay=False), logging.StreamHandler(sys.stdout)):
        handler.setLevel(logging.DEBUG if isinstance(handler, logging.FileHandler)
                         else logging.INFO)
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    row = list(range(len(hypothesis) + 1))
    for ref in reference:
        following = [row[0] + 1]
        for index, hyp in enumerate(hypothesis, 1):
            following.append(min(following[-1] + 1, row[index] + 1,
                                 row[index - 1] + (ref != hyp)))
        row = following
    return row[-1]


def source_fold(clip_id: str, folds: int) -> int:
    source = clip_id.rsplit("/", 1)[0]
    digest = hashlib.sha256(source.encode()).digest()
    return int.from_bytes(digest[:8], "big") % folds


def features(phones: list[str], score: float, vocabulary: tuple[str, ...]) -> list[float]:
    count = max(len(phones), 1)
    ids = [vocabulary.index(phone) for phone in phones]
    result = [score, score / count, len(phones), len(set(phones)) / count,
              sum(a == b for a, b in zip(ids, ids[1:])) / count]
    result.extend(ids.count(token) / count for token in range(len(vocabulary)))
    pairs = list(zip(ids, ids[1:]))
    pair_count = max(len(pairs), 1)
    result.extend(pairs.count((left, right)) / pair_count
                  for left in range(len(vocabulary)) for right in range(len(vocabulary)))
    return result


def fit_ridge(inputs: torch.Tensor, targets: torch.Tensor, ridge: float):
    mean = inputs.mean(0)
    scale = inputs.std(0).clamp_min(1e-5)
    normalized = (inputs - mean) / scale
    augmented = torch.cat((normalized, torch.ones(len(normalized), 1)), dim=1)
    penalty = torch.eye(augmented.shape[1], dtype=torch.float64) * ridge
    penalty[-1, -1] = 0
    weights = torch.linalg.solve(augmented.T @ augmented + penalty,
                                 augmented.T @ targets)
    return mean, scale, weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()
    if (not args.input.is_file() or args.folds < 2 or args.ridge <= 0
            or args.top_n < 1):
        parser.error("invalid input, folds, ridge, or top-n")
    configure_logging(args.output.with_suffix(".log"))
    rows = [json.loads(line) for line in args.input.read_text().splitlines()]
    vocabulary = tuple(sorted({phone for row in rows
                               for candidate in row["phone_hypotheses"]
                               for phone in candidate["phones"]}))
    examples = []
    for row in rows:
        reference = row["reference_phones"]
        candidates = row["phone_hypotheses"]
        if len(candidates) < args.top_n:
            continue
        vectors = [features(item["phones"], item["ctc_log_score"], vocabulary)
                   for item in candidates]
        errors = [edit_distance(reference, item["phones"]) for item in candidates]
        examples.append({"clip_id": row["clip_id"], "reference": reference,
                         "vectors": vectors, "errors": errors})
    all_predictions: dict[str, list[float]] = {}
    for fold in range(args.folds):
        train = [item for item in examples if source_fold(item["clip_id"], args.folds) != fold]
        held_out = [item for item in examples if source_fold(item["clip_id"], args.folds) == fold]
        x = torch.tensor([vector for item in train for vector in item["vectors"]],
                         dtype=torch.float64)
        y = torch.tensor([-error / max(len(item["reference"]), 1)
                          for item in train for error in item["errors"]], dtype=torch.float64)
        mean, scale, weights = fit_ridge(x, y, args.ridge)
        for item in held_out:
            held_x = torch.tensor(item["vectors"], dtype=torch.float64)
            augmented = torch.cat(((held_x - mean) / scale,
                                   torch.ones(len(held_x), 1)), dim=1)
            all_predictions[item["clip_id"]] = (augmented @ weights).tolist()
        LOGGER.info("fold=%d train_clips=%d held_out_clips=%d", fold, len(train), len(held_out))
    phones = baseline_errors = ranked_errors = ranked_top1_errors = 0
    for item in examples:
        predictions = all_predictions[item["clip_id"]]
        order = sorted(range(len(predictions)), key=lambda index: predictions[index], reverse=True)
        phones += len(item["reference"])
        baseline_errors += min(item["errors"][:args.top_n])
        ranked_errors += min(item["errors"][index] for index in order[:args.top_n])
        ranked_top1_errors += item["errors"][order[0]]
    report = {
        "format": "visual-phoneme-linear-reranker-cv-0.1",
        "input": str(args.input.resolve()), "clips": len(examples), "folds": args.folds,
        "ridge": args.ridge, "top_n": args.top_n, "vocabulary": list(vocabulary),
        "feature_dimensions": len(examples[0]["vectors"][0]), "reference_phones": phones,
        "baseline_oracle_per_at_n": baseline_errors / phones,
        "reranked_oracle_per_at_n": ranked_errors / phones,
        "reranked_top1_per": ranked_top1_errors / phones,
        "evaluation": "source-disjoint cross-validation; no reference-derived inference features",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    LOGGER.info("complete baseline_PER@%d=%.4f%% reranked_PER@%d=%.4f%% top1=%.4f%%",
                args.top_n, 100 * report["baseline_oracle_per_at_n"], args.top_n,
                100 * report["reranked_oracle_per_at_n"],
                100 * report["reranked_top1_per"])
    for handler in LOGGER.handlers:
        handler.flush()


if __name__ == "__main__":
    main()
