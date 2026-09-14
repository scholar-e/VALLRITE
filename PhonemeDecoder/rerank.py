"""Candidate-only bridge from VisualPhoneme.predict output to local Qwen LoRA."""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
from pathlib import Path

from PhonemeDecoder.train_qwen import candidate_prompt


class QwenScorer:
    def __init__(self, model, adapter, device="cuda"):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        base = AutoModelForImageTextToText.from_pretrained(
            model, local_files_only=True, dtype=torch.bfloat16).to(device)
        self.model = PeftModel.from_pretrained(base, adapter, local_files_only=True).eval()

    def __call__(self, record, indices):
        import torch
        prefix = self.tokenizer.encode(candidate_prompt(record), add_special_tokens=False)
        scores = []
        with torch.inference_mode():
            for index in indices:
                target = self.tokenizer.encode(" " + record["candidates"][index]["text"],
                                               add_special_tokens=False)
                target.append(self.tokenizer.eos_token_id)
                ids = torch.tensor([prefix + target], device=self.device)
                logits = self.model(input_ids=ids, use_cache=False).logits
                # Only target positions need the expensive vocabulary normalization.
                selected = logits[0, len(prefix) - 1:-1].float()
                loss = torch.nn.functional.cross_entropy(
                    selected, torch.tensor(target, device=self.device))
                scores.append(-loss.item())
        return scores


def rerank(result, phones, scorer, weight=1.0, score_window=5.0):
    """Return a copy; a scorer failure preserves the original candidate ranking.

    The current runtime exports aggregated lexical log_score, not per-path CTC
    scores. This is explicitly a base-score window until that contract expands.
    """
    if not all(math.isfinite(v) and 0 <= v <= 100 for v in (weight, score_window)):
        raise ValueError("weight and score window must be finite and in [0,100]")
    output = copy.deepcopy(result)
    candidates = output["candidates"]
    if not candidates or weight == 0:
        output["reranker"] = {"status": "skipped"}
        return output
    indices = [i for i, c in enumerate(candidates)
               if c["log_score"] >= candidates[0]["log_score"] - score_window]
    try:
        scores = scorer({"greedy_visual_phones": phones, "candidates": candidates}, indices)
        if len(scores) != len(indices) or not all(math.isfinite(s) for s in scores):
            raise ValueError("Scorer returned missing or nonfinite scores")
        ranked = []
        for index, score in zip(indices, scores, strict=True):
            candidate = candidates[index]
            candidate.update(base_rank=index, reranker_score=score,
                             final_score=candidate["log_score"] + weight * score)
            ranked.append(candidate)
        output["candidates"] = sorted(ranked, key=lambda c: (-c["final_score"], c["base_rank"]))
        output["reranker"] = {"status": "ok", "weight": weight,
            "base_score_window": score_window, "filtered": len(candidates) - len(ranked)}
    except Exception as error:
        output = copy.deepcopy(result)
        output["reranker"] = {"status": "fallback", "warning": str(error)}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--score-window", type=float, default=5.0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    if args.prediction.resolve() == args.output.resolve():
        parser.error("Use a separate output path to preserve the visual prediction")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s",
        handlers=[logging.FileHandler(args.output.with_suffix(".log")), logging.StreamHandler()])
    prediction = json.loads(args.prediction.read_text())
    # Lazy loading is inside the fallback boundary and is skipped for empty results.
    def score(record, indices):
        return QwenScorer(args.model, args.adapter, args.device)(record, indices)
    prediction["word_decoding_reranked"] = rerank(prediction["word_decoding"],
        prediction["phonemes"], score, args.weight, args.score_window)
    prediction["word_reranker_model"] = str(args.model.resolve())
    prediction["word_reranker_adapter"] = str(args.adapter.resolve())
    args.output.write_text(json.dumps(prediction, indent=2, allow_nan=False) + "\n")
    logging.info("reranker=%s output=%s", prediction["word_decoding_reranked"]["reranker"],
                 args.output)


if __name__ == "__main__":
    main()
