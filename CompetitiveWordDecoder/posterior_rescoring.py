"""Rescore decoder sentences by how well they explain the phone posteriors.

For each candidate sentence, its CMUdict pronunciation (the same lookup that
produced the training labels) is scored by the CTC forward likelihood under the
phone head's frame posteriors, i.e. log P(phones(W) | video). This is combined
with the existing ranker score (phone-path beam score + decoder token log
probability, merged over duplicate sentences) and word-count / unknown-word
terms. Weights are fitted on the test-select half only and applied once to
test-report.
"""
import argparse
from datetime import datetime, timezone
import itertools
import json
import logging
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from CompetitiveWordDecoder.train_vallr_clean import edit_distance
from PhonemeDecoder.evaluate_qwen_direct import words
from VisualPhoneme.data import PHONE_TO_ID
from VisualPhoneme.prepare_lrs3 import transcript_phones
from VisualPhoneme.train_autoavsr_phone_head import Split, build_head

LOGGER = logging.getLogger(__name__)
FEATURES = ("ctc", "ranker", "words", "unknown")


def logsumexp(values):
    largest = max(values)
    return largest + math.log(sum(math.exp(value - largest) for value in values))


@torch.inference_mode()
def phone_posteriors(head_path: Path, features: Path) -> dict[str, torch.Tensor]:
    """clip_id -> [steps, classes] log posteriors of every row in a cached split."""
    checkpoint = torch.load(head_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    split = Split(features, torch.device("cpu"))
    head = build_head(config, split.features.shape[1])
    head.load_state_dict(checkpoint["head"])
    head.eval()
    output = {}
    for start in range(0, len(split.rows), 64):
        indices = list(range(start, min(start + 64, len(split.rows))))
        batch, lengths, _, _ = split.batch(indices, head.upsample)
        log_probabilities = head(batch, lengths // head.upsample)
        for slot, index in enumerate(indices):
            output[split.rows[index]["clip_id"]] = log_probabilities[slot, :int(lengths[slot])].clone()
    return output


def ctc_log_likelihood(log_probabilities: torch.Tensor, phone_ids: list[int]) -> float:
    """log P(phone sequence | frames) summed over CTC alignments (-inf if too long)."""
    if not phone_ids:
        return float(log_probabilities[:, 0].sum())
    loss = F.ctc_loss(log_probabilities[:, None].double(), torch.tensor([phone_ids]),
                      torch.tensor([len(log_probabilities)]), torch.tensor([len(phone_ids)]),
                      blank=0, reduction="sum", zero_infinity=False)
    return -float(loss)


def sentence_groups(candidates: list[dict], log_probabilities: torch.Tensor) -> list[dict]:
    """Merge duplicate normalized sentences and attach every scoring feature."""
    groups = {}
    for index, item in enumerate(candidates):
        key = tuple(words(item["text"]))
        groups.setdefault(key, []).append(
            (index, item["phone_path_log_score"] + item["decoder_log_probability"]))
    result = []
    for key, entries in groups.items():
        phones, unknown, _ = transcript_phones(" ".join(key))
        result.append({
            "words": list(key), "first_index": entries[0][0],
            "ctc": ctc_log_likelihood(log_probabilities, [PHONE_TO_ID[p] for p in phones]),
            "ranker": logsumexp([value for _, value in entries]),
            "words_count": len(key), "unknown": len(unknown),
        })
    return result


def select(groups: list[dict], weights: dict) -> dict:
    def score(group):
        value = (weights["ctc"] * group["ctc"] + weights["ranker"] * group["ranker"]
                 + weights["words"] * group["words_count"] + weights["unknown"] * group["unknown"])
        return value if math.isfinite(value) else -math.inf
    return max(groups, key=lambda group: (score(group), -group["first_index"]))


def word_errors(clips: list[dict], weights: dict) -> int:
    return sum(select(clip["groups"], weights)["errors"] for clip in clips)


def load_clips(scores_paths: list[Path], paths_per_file: int = 5) -> dict[str, list[dict]]:
    """Read the ranker's per-candidate scores (exact generated tokens) for one half.

    File ``i`` holds phone paths ``i*paths_per_file+1 ..``; ``phone_rank`` becomes global.
    """
    by_clip = {}
    for offset, scores_path in enumerate(scores_paths):
        for line in scores_path.read_text().splitlines():
            item = json.loads(line)
            item["phone_rank"] += offset * paths_per_file
            by_clip.setdefault(item["clip_id"], []).append(item)
    return by_clip


def attach(by_clip: dict, hypotheses: Path, posteriors: dict) -> list[dict]:
    clips = []
    for line in hypotheses.read_text().splitlines():
        row = json.loads(line)
        candidates = sorted(by_clip[row["clip_id"]], key=lambda item: item["phone_rank"])
        reference = words(" ".join(row["reference_words"]))
        groups = sentence_groups(candidates, posteriors[row["clip_id"]])
        for group in groups:  # references enter only after features are fixed
            group["errors"] = edit_distance(reference, group["words"])
        clips.append({"clip_id": row["clip_id"], "reference": reference, "groups": groups,
                      "first": words(candidates[0]["text"])})
    return clips


GRID = {
    "ctc": (0.0, 1.0),
    "ranker": (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0),
    "words": tuple(x / 2 for x in range(-4, 13)),
    "unknown": (-20.0, -10.0, -5.0, -2.0, 0.0),
}


def fit(clips: list[dict]) -> tuple[dict, int]:
    best = None
    for values in itertools.product(*(GRID[name] for name in FEATURES)):
        weights = dict(zip(FEATURES, values))
        if not weights["ctc"] and not weights["ranker"]:
            continue
        errors = word_errors(clips, weights)
        if best is None or errors < best[1]:
            best = (weights, errors)
    return best


def summary(clips: list[dict], weights: dict) -> dict:
    count = sum(len(clip["reference"]) for clip in clips)
    errors = {
        "top_phone_path": sum(edit_distance(c["reference"], c["first"]) for c in clips),
        "existing_ranker": word_errors(clips, {"ctc": 0, "ranker": 1, "words": 0, "unknown": 0}),
        "ctc_only": word_errors(clips, {"ctc": 1, "ranker": 0, "words": 0, "unknown": 0}),
        "fitted": word_errors(clips, weights),
        "oracle": sum(min(g["errors"] for g in c["groups"]) for c in clips),
    }
    return {"clips": len(clips), "reference_words": count,
            **{name: {"word_errors": value, "wer": value / count} for name, value in errors.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", type=Path,
                        default=Path("checkpoints/autoavsr-phone-context2-20261102/best.pt"))
    parser.add_argument("--features", type=Path,
                        default=Path("checkpoints/autoavsr-features-20261102/test"))
    parser.add_argument("--select-scores", type=Path, nargs="+", required=True,
                        help="ranker scores.jsonl files covering successive blocks of five paths")
    parser.add_argument("--select-hypotheses", type=Path, required=True)
    parser.add_argument("--report-scores", type=Path, nargs="+", required=True,
                        help="ranker scores.jsonl files covering successive blocks of five paths")
    parser.add_argument("--report-hypotheses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(filename)s:%(lineno)d %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(args.output / "rescore.log"),
                                  logging.StreamHandler()], force=True)
    posteriors = phone_posteriors(args.head, args.features)
    LOGGER.info("Phone posteriors for %d clips", len(posteriors))
    selection = attach(load_clips(args.select_scores), args.select_hypotheses, posteriors)
    weights, errors = fit(selection)
    LOGGER.info("Fitted on test-select: %s (%d word errors)", weights, errors)
    report = attach(load_clips(args.report_scores), args.report_hypotheses, posteriors)
    result = {
        "weights": weights, "grid": GRID,
        "test_select": summary(selection, weights),
        "test_report": summary(report, weights),
        "inputs": {key: str(value) for key, value in vars(args).items()},
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output / "test-report-selected.jsonl").open("w") as handle:
        for clip in report:
            chosen = select(clip["groups"], weights)
            handle.write(json.dumps({"clip_id": clip["clip_id"], "text": " ".join(chosen["words"]),
                                     "reference": " ".join(clip["reference"])}) + "\n")
    for name in ("test_select", "test_report"):
        LOGGER.info("%s: %s", name, {key: round(100 * value["wer"], 2)
                                     for key, value in result[name].items() if isinstance(value, dict)})


if __name__ == "__main__":
    main()
