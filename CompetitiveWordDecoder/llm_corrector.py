"""Anchored LLM correction on top of 20-candidate rescoring.

``generate``: rank each clip's candidates with rescoring weights fitted on
test-select (references never enter), show the top five to an instruction LLM
through an OpenAI-compatible completions server (LM Studio), and save one
corrected sentence per clip.

``evaluate``: refit the rescoring on test-select with a wider LM-weight grid,
then let the corrected sentence replace the chosen one only when its evidence
score (head C CTC fit + LM + word-count + unknown-word terms, same weights;
the decoder never produced it, so it has no ranker score) beats the chosen
sentence's by a margin fitted on test-select ("never" is allowed).
Test-report is scored once for the wider grid alone and with the corrector.
"""
import argparse
import json
import logging
import math
from pathlib import Path
import random
import urllib.request

from CompetitiveWordDecoder import posterior_rescoring as rescoring
from CompetitiveWordDecoder.train_vallr_clean import edit_distance
from PhonemeDecoder.evaluate_qwen_direct import words
from VisualPhoneme.data import PHONE_TO_ID
from VisualPhoneme.prepare_lrs3 import transcript_phones

LOGGER = logging.getLogger(__name__)
PIPELINE = Path("checkpoints/posterior-headC-20261106/phones20")
PROMPT = (
    "<|im_start|>user\nA lip-reading system produced these candidate transcriptions of one short "
    "clip of spoken English, best first. They are noisy: words that look alike on the lips "
    "(p/b/m, f/v, t/d/n, k/g, s/z, sh/ch/j) are often confused, and words can be split or merged "
    "wrongly. Write the single most likely sentence actually spoken. Stay close to the sounds of the "
    "candidates and do not add content. Reply with the sentence only, in lowercase, without "
    "punctuation.\n\n{candidates}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
WIDE_LM = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
MARGINS = (-math.inf, -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0, math.inf)  # inf = never replace


def load_half(half: str, posteriors: dict, lm_scores: dict) -> list[dict]:
    scores = [PIPELINE / half / f"decode{i}" / "ranked" / "scores.jsonl" for i in range(4)]
    return rescoring.attach(rescoring.load_clips(scores), PIPELINE / half / "phone-hypotheses.jsonl",
                            posteriors, lm_scores)


def read_lm(paths: list[Path]) -> dict:
    return {json.loads(line)["text"]: json.loads(line)["lm_logprob"]
            for path in paths for line in path.read_text().splitlines() if line.strip()}


def evidence(group: dict, weights: dict) -> float:
    value = (weights["ctc"] * group["ctc"] + weights["words"] * group["words_count"]
             + weights["unknown"] * group["unknown"] + weights.get("lm", 0.0) * group["lm"])
    return value if math.isfinite(value) else -math.inf


def corrected_group(text: str, posterior, lm_scores: dict) -> dict | None:
    key = tuple(words(text))
    if not key or " ".join(key) not in lm_scores:
        return None
    phones, unknown, _ = transcript_phones(" ".join(key))
    return {"words": list(key), "ctc": rescoring.ctc_log_likelihood(posterior, [PHONE_TO_ID[p] for p in phones]),
            "words_count": len(key), "unknown": len(unknown), "lm": lm_scores[" ".join(key)]}


def final_choice(clip: dict, weights: dict, margin: float, correction: dict | None) -> list[str]:
    chosen = rescoring.select(clip["groups"], weights)
    if (correction is not None and margin != math.inf
            and evidence(correction, weights) - evidence(chosen, weights) > margin):
        return correction["words"]
    return chosen["words"]


def errors(clips: list[dict], weights: dict, margin: float, corrections: dict) -> int:
    return sum(edit_distance(c["reference"], final_choice(c, weights, margin, corrections.get(c["clip_id"])))
               for c in clips)


def generate(args) -> None:
    posteriors = rescoring.phone_posteriors(args.head, args.features)
    lm_scores = read_lm([args.lm_scores])
    weights = json.loads(args.weights.read_text())["weights"]
    with args.output.open("w") as handle:
        for half in ("select", "report"):
            for clip in load_half(half, posteriors, lm_scores):
                ranked = sorted(clip["groups"], key=lambda g: (
                    -(weights["ctc"] * g["ctc"] + weights["ranker"] * g["ranker"]
                      + weights["words"] * g["words_count"] + weights["unknown"] * g["unknown"]
                      + weights.get("lm", 0.0) * g["lm"]), g["first_index"]))[:5]
                listing = "\n".join(f"{i}. {' '.join(g['words'])}" for i, g in enumerate(ranked, 1))
                body = json.dumps({"model": args.model, "prompt": PROMPT.format(candidates=listing),
                                   "max_tokens": 64, "temperature": 0, "stop": ["\n", "<|im_end|>"]}).encode()
                request = urllib.request.Request(args.url, body, {"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    text = json.loads(response.read())["choices"][0]["text"].strip()
                handle.write(json.dumps({"clip_id": clip["clip_id"], "half": half, "text": text,
                                         "shown": [" ".join(g["words"]) for g in ranked]}) + "\n")
    LOGGER.info("wrote %s", args.output)


def evaluate(args) -> None:
    posteriors = rescoring.phone_posteriors(args.head, args.features)
    lm_scores = read_lm([args.lm_scores, args.correction_lm_scores])
    selection, report = (load_half(half, posteriors, lm_scores) for half in ("select", "report"))
    rescoring.GRID["lm"] = WIDE_LM
    weights, select_errors = rescoring.fit(selection, rescoring.FEATURES + ("lm",))
    LOGGER.info("wider-grid weights fitted on test-select: %s (%d errors)", weights, select_errors)
    corrections = {}
    for line in args.corrections.read_text().splitlines():
        row = json.loads(line)
        group = corrected_group(row["text"], posteriors[row["clip_id"]], lm_scores)
        if group:
            corrections[row["clip_id"]] = group
    margin = min(MARGINS, key=lambda m: (errors(selection, weights, m, corrections), -m))
    LOGGER.info("margin fitted on test-select: %s", margin)
    count = lambda clips: sum(len(c["reference"]) for c in clips)
    previous = json.loads(args.previous_weights.read_text())["weights"]
    result = {"weights": weights, "margin": margin, "corrections_usable": len(corrections)}
    for name, clips in (("test_select", selection), ("test_report", report)):
        result[name] = {label: errors(clips, w, m, corrections) / count(clips) for label, w, m in (
            ("previous_lm_grid", previous, math.inf), ("wider_lm_grid", weights, math.inf),
            ("wider_lm_grid_plus_corrector", weights, margin))}
        result[name]["replaced_clips"] = sum(
            final_choice(c, weights, margin, corrections.get(c["clip_id"]))
            != rescoring.select(c["groups"], weights)["words"] for c in clips)
        LOGGER.info("%s: %s", name, {k: round(100 * v, 2) if isinstance(v, float) else v
                                     for k, v in result[name].items()})
    diffs = [edit_distance(c["reference"], final_choice(c, weights, margin, corrections.get(c["clip_id"])))
             - edit_distance(c["reference"], rescoring.select(c["groups"], previous)["words"]) for c in report]
    rng = random.Random(0)
    boots = sorted(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / count(report) for _ in range(10000))
    result["report_vs_previous_best"] = {"difference": sum(diffs) / count(report), "ci95": [boots[250], boots[9750]],
                                         "better": sum(d < 0 for d in diffs), "worse": sum(d > 0 for d in diffs)}
    LOGGER.info("test-report vs previous best: %s", result["report_vs_previous_best"])
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("generate", "evaluate"))
    parser.add_argument("--head", type=Path, default=Path("checkpoints/autoavsr-phone-C-masked-20261106/best.pt"))
    parser.add_argument("--features", type=Path, default=Path("checkpoints/autoavsr-features-20261102/test"))
    parser.add_argument("--lm-scores", type=Path, default=Path("checkpoints/lm-rescoring-20261126/lm-scores.jsonl"))
    parser.add_argument("--weights", type=Path, default=Path("checkpoints/lm-rescoring-20261126/rescored/metrics.json"),
                        help="generate: rescoring weights used to rank the candidates shown")
    parser.add_argument("--previous-weights", type=Path,
                        default=Path("checkpoints/lm-rescoring-20261126/rescored/metrics.json"))
    parser.add_argument("--url", default="http://localhost:1234/v1/completions")
    parser.add_argument("--model", default="corrector")
    parser.add_argument("--corrections", type=Path)
    parser.add_argument("--correction-lm-scores", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(filename)s:%(lineno)d %(message)s", force=True)
    (generate if args.mode == "generate" else evaluate)(args)


if __name__ == "__main__":
    main()
