"""Score candidate sentences with a general language model for rescoring.

Collects every distinct normalized sentence from ranker ``scores.jsonl`` files
and writes ``{"text", "lm_logprob", "tokens"}`` rows: the summed token log
probability of the lowercase sentence after a newline, under a causal LM read
from the local Hugging Face cache. References are never read.
"""
import argparse
import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from PhonemeDecoder.evaluate_qwen_direct import words
from VisualPhoneme.thermal import ThermalGuard

LOGGER = logging.getLogger(__name__)


@torch.inference_mode()
def sentence_log_probs(model, tokenizer, sentences: list[str], batch_size: int) -> list[tuple[float, int]]:
    """Sum of log P(token | newline, earlier tokens) over each sentence's tokens."""
    prefix = tokenizer("\n", add_special_tokens=False)["input_ids"]
    results = []
    for start in range(0, len(sentences), batch_size):
        chunk = [tokenizer(text, add_special_tokens=False)["input_ids"] for text in sentences[start:start + batch_size]]
        ids = [prefix + tokens for tokens in chunk]
        width = max(len(row) for row in ids)
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        batch = torch.tensor([row + [pad] * (width - len(row)) for row in ids], device=model.device)
        mask = torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in ids], device=model.device)
        log_probs = model(input_ids=batch, attention_mask=mask).logits[:, :-1].float().log_softmax(-1)
        picked = log_probs.gather(-1, batch[:, 1:, None])[..., 0]
        for slot, tokens in enumerate(chunk):
            span = picked[slot, len(prefix) - 1:len(prefix) - 1 + len(tokens)]
            results.append((float(span.sum()), len(tokens)))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B-Base")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(filename)s:%(lineno)d %(message)s",
                        force=True)
    sentences = sorted({" ".join(words(json.loads(line)["text"])) for path in args.scores
                        for line in path.read_text().splitlines() if line.strip()})
    sentences = [s for s in sentences if s]
    LOGGER.info("%d distinct sentences", len(sentences))
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=".cache/huggingface/hub", local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir=".cache/huggingface/hub",
                                                 local_files_only=True, dtype=torch.bfloat16).to("cuda").eval()
    order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]))  # similar lengths per batch
    with ThermalGuard(args.output.parent, True):
        scored = sentence_log_probs(model, tokenizer, [sentences[i] for i in order], args.batch_size)
    with args.output.open("w") as handle:
        for index, (logprob, count) in zip(order, scored):
            handle.write(json.dumps({"text": sentences[index], "lm_logprob": logprob, "tokens": count}) + "\n")
    LOGGER.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
