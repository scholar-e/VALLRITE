"""Evaluate a Qwen LoRA checkpoint as a bounded decoder-candidate reranker."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("checkpoints/Qwen3.5-2B"))
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--visual-score-window", type=float, default=5.0)
    parser.add_argument("--weights", type=float, nargs="+",
                        default=[0.0, 0.1, 0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s",
        handlers=[logging.FileHandler(args.output.with_suffix(".log")),
                  logging.StreamHandler()])
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from PhonemeDecoder.train_qwen import candidate_prompt
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but is unavailable")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForImageTextToText.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        low_cpu_mem_usage=True).to(device)
    model = PeftModel.from_pretrained(base, args.adapter, local_files_only=True)
    model.eval()
    records = [json.loads(line) for line in args.examples.read_text().splitlines()]
    if args.limit:
        records = records[:args.limit]
    items = []
    score_path = args.output.with_name(args.output.stem + ".scores.jsonl")
    for ri, record in enumerate(records):
        candidates = record["candidates"]
        if not candidates:
            continue
        cutoff = candidates[0]["log_score"] - args.visual_score_window
        selected = [i for i, candidate in enumerate(candidates)
                    if candidate["log_score"] >= cutoff]
        if 0 not in selected:
            selected.insert(0, 0)
        prompt_ids = tokenizer.encode(candidate_prompt(record), add_special_tokens=False)
        for ci in selected:
            target = tokenizer.encode(" " + candidates[ci]["text"], add_special_tokens=False)
            target.append(tokenizer.eos_token_id)
            items.append((ri, ci, prompt_ids + target,
                          [-100] * len(prompt_ids) + target))
    LOGGER.info("device=%s records=%d candidate_sequences=%d adapter=%s",
                device, len(records), len(items), args.adapter)
    score_output = score_path.open("w", buffering=1)
    with torch.inference_mode(), score_output:
        for start in range(0, len(items), args.batch_size):
            batch = items[start:start + args.batch_size]
            width = max(len(item[2]) for item in batch)
            input_ids, attention, labels = [], [], []
            for _, _, ids, labs in batch:
                padding = width - len(ids)
                input_ids.append(ids + [tokenizer.pad_token_id] * padding)
                attention.append([1] * len(ids) + [0] * padding)
                labels.append(labs + [-100] * padding)
            input_tensor = torch.tensor(input_ids, device=device)
            logits = model(input_ids=input_tensor,
                           attention_mask=torch.tensor(attention, device=device)).logits
            log_probs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
            target_tensor = torch.tensor(labels, device=device)[:, 1:]
            mask = target_tensor.ne(-100)
            gathered = log_probs.gather(-1, target_tensor.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            means = (gathered * mask).sum(1) / mask.sum(1)
            for item, score in zip(batch, means.tolist(), strict=True):
                ri, ci = item[:2]
                score_output.write(json.dumps({"clip_id": records[ri]["clip_id"],
                    "candidate_index": ci,
                    "candidate_text": records[ri]["candidates"][ci]["text"],
                    "score": score}) + "\n")
            if start == 0 or (start // args.batch_size) % 100 == 0:
                LOGGER.info("scored=%d/%d", min(start + len(batch), len(items)), len(items))
    from PhonemeDecoder.evaluate_candidate_scores import evaluate, load_scores
    output = evaluate(records, load_scores(score_path), args.weights,
                      args.visual_score_window)
    output.update({"scope": "use the examples manifest to determine evaluation scope",
              "adapter": str(args.adapter.resolve()),
              "candidate_sequences_scored": len(items),
              "score": "mean target-token log likelihood including EOS",
              "score_records": str(score_path.resolve())})
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    LOGGER.info("complete results=%s", output["results_by_reranker_weight"])


if __name__ == "__main__":
    main()
