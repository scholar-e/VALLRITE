"""Probe whether an LLM has memorized LRS3 test transcripts (TED talks are public text).

Each sentence of 8+ words is cut in half; the model continues the first half
greedily from raw text (no chat template). Memorization shows as verbatim
continuations: exact rest of sentence, or exact next four words. The control is
Seamless Interaction transcripts (recent conversations, not public text) with
the same count and length filter. Conversational speech is more formulaic than
TED, so the control errs toward a higher baseline, not a lower one.
Backends: ``hf`` (local transformers model) or ``http`` (OpenAI-compatible
/v1/completions server such as LM Studio).
"""
import argparse
import json
import logging
from pathlib import Path
import random
import re
import urllib.request

LOGGER = logging.getLogger(__name__)
WORD = re.compile(r"[a-z]+(?:'[a-z]+)*")


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def sentences(manifest: Path, limit: int | None, seed: int) -> list[list[str]]:
    rows = [words(json.loads(line)["transcript"]) for line in manifest.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if len(row) >= 8]
    if limit is not None and len(rows) > limit:
        rows = random.Random(seed).sample(rows, limit)
    return rows


def score(sentence: list[str], continuation: str) -> dict:
    half = len(sentence) // 2
    target, produced = sentence[half:], words(continuation)
    matched = 0
    for expected, got in zip(target, produced):
        if expected != got:
            break
        matched += 1
    return {"exact_rest": matched == len(target), "next4": matched >= min(4, len(target)),
            "matched_fraction": matched / len(target)}


def http_generate(url: str, model: str, prompt: str, max_tokens: int) -> str:
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0, "stream": False}).encode()
    request = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())["choices"][0]["text"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("hf", "http"), required=True)
    parser.add_argument("--model", required=True, help="HF id/path, or the server's model key")
    parser.add_argument("--revision", help="HF revision")
    parser.add_argument("--url", default="http://localhost:1234/v1/completions")
    parser.add_argument("--lrs3", type=Path, default=Path("datasets/lrs3/manifests/test.jsonl"))
    parser.add_argument("--control", type=Path,
                        default=Path("datasets/seamless-interaction/prepared/batch0000/manifest.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261126)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(filename)s:%(lineno)d %(message)s",
                        handlers=[logging.FileHandler(args.output / "probe.log"), logging.StreamHandler()],
                        force=True)
    test_rows = [line for line in args.lrs3.read_text().splitlines() if line.strip()]
    select_manifest = args.output / "lrs3-test-select.jsonl"  # first half only, as everywhere else
    select_manifest.write_text("\n".join(test_rows[:len(test_rows) // 2]) + "\n")
    lrs3 = sentences(select_manifest, None, args.seed)
    sets = {"lrs3_test_select": lrs3, "seamless_control": sentences(args.control, len(lrs3), args.seed)}

    if args.backend == "hf":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from VisualPhoneme.thermal import ThermalGuard
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision,
                                                  cache_dir=".cache/huggingface/hub", local_files_only=True)
        tokenizer.padding_side, tokenizer.pad_token = "left", tokenizer.pad_token or tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.revision, cache_dir=".cache/huggingface/hub", local_files_only=True,
            dtype=torch.bfloat16).to("cuda").eval()
        guard = ThermalGuard(args.output, True)

        def generate(prompts: list[str], max_tokens: int) -> list[str]:
            batch = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
            with torch.inference_mode():
                out = model.generate(**batch, max_new_tokens=max_tokens, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id)
            return tokenizer.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
    else:
        from contextlib import nullcontext
        guard = nullcontext()  # the server process is guarded separately

        def generate(prompts: list[str], max_tokens: int) -> list[str]:
            return [http_generate(args.url, args.model, prompt, max_tokens) for prompt in prompts]

    results, examples = {}, {}
    with guard:
        for name, rows in sets.items():
            scored = []
            for start in range(0, len(rows), 16):
                chunk = rows[start:start + 16]
                prompts = [" ".join(row[:len(row) // 2]) for row in chunk]
                outputs = generate(prompts, 3 * max(len(row) - len(row) // 2 for row in chunk) + 8)
                scored += [score(row, text) for row, text in zip(chunk, outputs)]
                examples.setdefault(name, []).extend(
                    {"prefix": p, "expected": " ".join(r[len(r) // 2:]), "produced": " ".join(words(t))}
                    for p, r, t in zip(prompts, chunk, outputs))
            results[name] = {"sentences": len(scored),
                             "exact_rest": sum(s["exact_rest"] for s in scored) / len(scored),
                             "next4": sum(s["next4"] for s in scored) / len(scored),
                             "mean_matched_fraction": sum(s["matched_fraction"] for s in scored) / len(scored)}
            LOGGER.info("%s %s: %d sentences, exact rest %.1f%%, next 4 words %.1f%%, matched %.1f%%",
                        args.model, name, len(scored), 100 * results[name]["exact_rest"],
                        100 * results[name]["next4"], 100 * results[name]["mean_matched_fraction"])
    (args.output / "metrics.json").write_text(json.dumps({"model": args.model, **results}, indent=2) + "\n")
    with (args.output / "examples.jsonl").open("w") as handle:
        for name, rows in examples.items():
            for row in rows:
                handle.write(json.dumps({"set": name, **row}) + "\n")


if __name__ == "__main__":
    main()
