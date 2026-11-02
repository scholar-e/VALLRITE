"""Synthetic phone errors matched to a measured visual phone recognizer.

The profile is estimated from rank-one hypotheses on a selection split: for
each reference phone, probabilities of being kept, deleted or substituted (by
which phone), plus the probability and identity of insertions after each
position. Real errors are bursty, so each sampled sentence draws an error scale
from the measured per-clip phone error rates.

The CLI writes decoder training rows (``prompt``/``target``) for
``train_vallr_clean.py --prepared-rows``.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

from CompetitiveWordDecoder.evaluate_vallr_qwen import vallr_prompt, words
from VisualPhoneme.evaluate_autoavsr_ctc import edit_counts

VERSION = "matched-phone-noise-v1"


def align(reference: list[str], hypothesis: list[str]) -> list[tuple[str, str | None, list[str]]]:
    """Minimum-edit alignment as (reference phone, output or None, inserted after).

    Insertions before the first reference phone are attached to a "<s>" entry.
    """
    rows, columns = len(reference), len(hypothesis)
    cost = [[0] * (columns + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        cost[i][0] = i
    for j in range(1, columns + 1):
        cost[0][j] = j
    for i in range(1, rows + 1):
        for j in range(1, columns + 1):
            cost[i][j] = min(cost[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]),
                             cost[i - 1][j] + 1, cost[i][j - 1] + 1)
    operations = []
    i, j = rows, columns
    while i or j:
        if i and j and cost[i][j] == cost[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]):
            operations.append(("align", reference[i - 1], hypothesis[j - 1]))
            i, j = i - 1, j - 1
        elif i and cost[i][j] == cost[i - 1][j] + 1:
            operations.append(("delete", reference[i - 1], None))
            i -= 1
        else:
            operations.append(("insert", None, hypothesis[j - 1]))
            j -= 1
    result: list[tuple[str, str | None, list[str]]] = [("<s>", "<s>", [])]
    for kind, ref, out in reversed(operations):
        if kind == "insert":
            result[-1][2].append(out)
        else:
            result.append((ref, out, []))
    return result


def estimate_profile(records: list[dict]) -> dict:
    outcomes: dict[str, Counter] = defaultdict(Counter)
    insert_counts: Counter = Counter()
    inserted: Counter = Counter()
    clip_rates = []
    for record in records:
        reference = record["reference_phones"]
        hypothesis = record["phone_hypotheses"][0]["phones"]
        edits = 0
        for ref, out, extra in align(reference, hypothesis):
            insert_counts[len(extra)] += 1
            inserted.update(extra)
            edits += len(extra)
            if ref == "<s>":
                continue
            outcomes[ref]["<del>" if out is None else out] += 1
            edits += out != ref
        if reference:
            clip_rates.append(edits / len(reference))
    total_phones = sum(sum(counter.values()) for counter in outcomes.values())
    total_edits = sum(rate * len(r["reference_phones"]) for rate, r in zip(clip_rates, records))
    return {
        "version": VERSION, "clips": len(records), "reference_phones": total_phones,
        "phone_error_rate": total_edits / total_phones,
        "outcomes": {phone: dict(counter) for phone, counter in sorted(outcomes.items())},
        "insertions_per_position": {str(k): v for k, v in sorted(insert_counts.items())},
        "inserted_phones": dict(inserted),
        "clip_error_rates": clip_rates,
    }


class NoiseModel:
    def __init__(self, profile: dict):
        self.profile = profile
        self.mean_rate = profile["phone_error_rate"]
        self.outcomes = {phone: (list(counts), list(counts.values()))
                         for phone, counts in profile["outcomes"].items()}
        counts = profile["insertions_per_position"]
        self.insert_sizes = [int(k) for k in counts]
        self.insert_weights = list(counts.values())
        self.inserted = (list(profile["inserted_phones"]), list(profile["inserted_phones"].values()))

    def corrupt(self, phones: list[str], rng: random.Random) -> tuple[list[str], float]:
        """Return noisy phones and the sentence error scale used."""
        # Scale relative to the mean so sentence-level burstiness matches the measurement.
        scale = rng.choice(self.profile["clip_error_rates"]) / self.mean_rate
        output = self._insert(rng, scale)
        for phone in phones:
            if phone not in self.outcomes:
                output.append(phone)
            else:
                choices, weights = self.outcomes[phone]
                kept = weights[choices.index(phone)] if phone in choices else 0
                errors = sum(weights) - kept
                if kept + errors and rng.random() < min(1.0, scale * errors / (kept + errors)):
                    error_choices = [c for c in choices if c != phone]
                    error_weights = [w for c, w in zip(choices, weights) if c != phone]
                    outcome = rng.choices(error_choices, error_weights)[0]
                    if outcome != "<del>":
                        output.append(outcome)
                else:
                    output.append(phone)
            output.extend(self._insert(rng, scale))
        return output, scale

    def _insert(self, rng: random.Random, scale: float) -> list[str]:
        size = rng.choices(self.insert_sizes, self.insert_weights)[0]
        if size and rng.random() >= min(1.0, scale):
            size = 0
        return rng.choices(*self.inserted, k=size) if size else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-source", type=Path, required=True,
                        help="phone-hypotheses.jsonl on a selection split (never the report split)")
    parser.add_argument("--train-manifest", type=Path,
                        default=Path("datasets/lrs3/manifests/autoavsr/train.jsonl"))
    parser.add_argument("--exclude-manifest", type=Path, action="append",
                        default=[Path("datasets/lrs3/manifests/test.jsonl")],
                        help="transcripts to exclude from training text")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--noisy", type=int, default=9000)
    parser.add_argument("--clean", type=int, default=3000)
    parser.add_argument("--validation", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in args.profile_source.read_text().splitlines()]
    profile = estimate_profile(records)
    model = NoiseModel(profile)
    excluded = {" ".join(words(json.loads(line)["transcript"]))
                for path in args.exclude_manifest for line in path.read_text().splitlines()}
    candidates, seen = [], set()
    for line in args.train_manifest.read_text().splitlines():
        row = json.loads(line)
        text = " ".join(words(row["transcript"]))
        if (not text or text in excluded or text in seen or row.get("unknown_words")
                or len(text.split()) < 2):
            continue
        seen.add(text)
        candidates.append(row)
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    needed = args.noisy + args.clean + 2 * args.validation
    if len(candidates) < needed:
        raise ValueError(f"need {needed} unique transcripts, found {len(candidates)}")
    stats = {"candidates": len(candidates)}
    splits = {
        "train": [("noisy", row) for row in candidates[:args.noisy]]
        + [("clean", row) for row in candidates[args.noisy:args.noisy + args.clean]],
        "validation": [("noisy", row) for row in
                       candidates[needed - 2 * args.validation:needed - args.validation]]
        + [("clean", row) for row in candidates[needed - args.validation:needed]],
    }
    for split, items in splits.items():
        rng.shuffle(items)
        edits = phones = 0
        with (args.output / f"{split}.jsonl").open("w") as handle:
            for kind, row in items:
                reference = row["phonemes"]
                noisy = model.corrupt(reference, rng)[0] if kind == "noisy" else reference
                handle.write(json.dumps({
                    "prompt": vallr_prompt(noisy), "target": " ".join(words(row["transcript"])),
                    "unknown_words": 0, "clip_id": row["clip_id"], "input_kind": kind,
                }) + "\n")
                if kind == "noisy":
                    edits += sum(edit_counts(reference, noisy))
                    phones += len(reference)
        stats[split] = {"rows": len(items), "noisy_rows": sum(k == "noisy" for k, _ in items),
                        "realized_noisy_phone_error_rate": edits / phones if phones else 0}
    profile_path = args.output / "profile.json"
    profile_path.write_text(json.dumps(profile, indent=2) + "\n")
    metadata = {
        "version": VERSION, "seed": args.seed, "profile_source": str(args.profile_source),
        "profile_source_sha256": hashlib.sha256(args.profile_source.read_bytes()).hexdigest(),
        "train_manifest_sha256": hashlib.sha256(args.train_manifest.read_bytes()).hexdigest(),
        "measured_phone_error_rate": profile["phone_error_rate"], **stats,
        "note": "trainval transcripts, test transcripts excluded; targets normalized lowercase",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
