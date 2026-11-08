"""Rescore pooled candidates with the posterior-reading decoder.

The posterior decoder (``posterior_decoder.py``, encoder trained with the LoRA
frozen) assigns much higher probability to reference sentences than the plain
decoder (test-select token loss 1.69 -> 1.49), but its greedy output inserts
fluent, unsupported words. Here it is used as a scorer instead. Per clip the
candidate pool is the union of:

- the 20 decoded phone paths (head C, matched-noise decoder),
- the posterior-fused decoder's finished hypotheses,
- five beam-search outputs of the posterior decoder itself;

each unique sentence is scored by

    log P_posterior-decoder(W | phone path, posteriors) + ctc_weight * log P_CTC(phones(W) | video)
        + word_weight * |W| + unknown_weight * |unknown words|

with weights fitted on test-select and applied once to test-report.
"""
import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import random

import torch

from CompetitiveWordDecoder.posterior_decoder import Batcher, PosteriorEncoder, PosteriorSet, load_model
from CompetitiveWordDecoder.posterior_rescoring import ctc_log_likelihood
from CompetitiveWordDecoder.train_vallr_clean import edit_distance
from PhonemeDecoder.evaluate_qwen_direct import words
from VisualPhoneme.data import PHONE_TO_ID
from VisualPhoneme.prepare_lrs3 import transcript_phones


def pooled_sentences(pool_root: Path, fused_root: Path, name: str) -> dict[str, set[tuple]]:
    pool: dict[str, set[tuple]] = {}
    for block in range(4):
        path = pool_root / name / f"decode{block}" / "visual" / "generations.jsonl"
        for line in path.read_text().splitlines():
            item = json.loads(line)
            pool.setdefault(item["clip_id"], set()).add(tuple(words(item["text"])))
    for line in (fused_root / f"{name}-w0.1" / "nbest.jsonl").read_text().splitlines():
        row = json.loads(line)
        for hypothesis in row["nbest"]:
            pool[row["clip_id"]].add(tuple(words(hypothesis["text"])))
    return pool


@torch.inference_mode()
def beam_candidates(batcher: Batcher, data: PosteriorSet, rows: list[dict], beams: int = 5) -> list[list[str]]:
    output = []
    for start in range(0, len(rows), 8):
        chunk = rows[start:start + 8]
        sequences = batcher.pieces(data, chunk, False)
        longest = max(len(s) for s, _ in sequences)
        pad = batcher.embed(torch.tensor([batcher.eos], device=batcher.embed.weight.device))[0]
        embeds = pad.expand(len(chunk), longest, pad.shape[0]).clone()
        mask = torch.zeros(len(chunk), longest, dtype=torch.long, device=embeds.device)
        for slot, (sequence, _) in enumerate(sequences):
            embeds[slot, longest - len(sequence):] = sequence
            mask[slot, longest - len(sequence):] = 1
        generated = batcher.model.generate(inputs_embeds=embeds, attention_mask=mask, do_sample=False,
                                           num_beams=beams, num_return_sequences=beams, max_new_tokens=64,
                                           eos_token_id=batcher.eos, pad_token_id=batcher.eos, use_cache=True)
        texts = [batcher.tokenizer.decode(ids, skip_special_tokens=True) for ids in generated.tolist()]
        output += [texts[i * beams:(i + 1) * beams] for i in range(len(chunk))]
    return output


@torch.inference_mode()
def sentence_log_probabilities(batcher: Batcher, data: PosteriorSet, row: dict,
                               sentences: list[tuple], batch_size: int = 32) -> list[float]:
    """Sum of target-token (and EOS) log probabilities of each sentence for one clip."""
    scores = []
    for start in range(0, len(sentences), batch_size):
        chunk = [{**row, "transcript": " ".join(s)} for s in sentences[start:start + batch_size]]
        embeds, mask, labels = batcher.training_batch(data, chunk)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = batcher.model(inputs_embeds=embeds, attention_mask=mask).logits.float()
        losses = torch.nn.functional.cross_entropy(logits[:, :-1].transpose(1, 2), labels[:, 1:],
                                                   ignore_index=-100, reduction="none")
        scores += (-losses.sum(-1)).tolist()
    return scores


def score(candidate: dict, weights: tuple) -> float:
    ctc_weight, word_weight, unknown_weight = weights
    return (candidate["decoder"] + ctc_weight * candidate["ctc"] + word_weight * candidate["words"]
            + unknown_weight * candidate["unknown"])


def errors(clips: list[dict], weights: tuple) -> int:
    return sum(max(clip["candidates"], key=lambda c: score(c, weights))["errors"] for clip in clips)


GRID = ((0.0, 0.25, 0.5, 1.0, 2.0, 4.0), (-2.0, -1.0, 0.0, 1.0, 2.0), (-20.0, -5.0, 0.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="directory with adapter/ and encoder.pt")
    parser.add_argument("--test", type=Path,
                        default=Path("checkpoints/posterior-decoder-20261107/posteriors-test"))
    parser.add_argument("--pool-root", type=Path,
                        default=Path("checkpoints/posterior-headC-20261106/phones20"))
    parser.add_argument("--fused-root", type=Path, default=Path("checkpoints/posterior-headC-20261106/fused"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zero-soft", action="store_true",
                        help="control: zero the posterior tokens (plain decoder, same positions)")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    test = PosteriorSet(args.test)
    half = len(test.rows) // 2
    model, tokenizer = load_model(args.run / "adapter", trainable=False)
    model.eval()
    encoder = PosteriorEncoder(model.config.hidden_size).cuda().eval()
    encoder.load_state_dict(torch.load(args.run / "encoder.pt", weights_only=True))
    batcher = Batcher(model, tokenizer, encoder)
    batcher.zero_soft = args.zero_soft
    halves = {}
    for name, rows in (("select", test.rows[:half]), ("report", test.rows[half:])):
        pool = pooled_sentences(args.pool_root, args.fused_root, name)
        beams = beam_candidates(batcher, test, rows)
        clips = []
        for row, generated in zip(rows, beams, strict=True):
            sentences = pool[row["clip_id"]] | {tuple(words(text)) for text in generated}
            sentences = sorted(s for s in sentences if s)
            decoder = sentence_log_probabilities(batcher, test, row, sentences)
            posterior = test.posterior(row)
            reference = words(row["transcript"])
            candidates = []
            for sentence, value in zip(sentences, decoder, strict=True):
                phones, unknown, _ = transcript_phones(" ".join(sentence))
                candidates.append({"words_list": list(sentence), "decoder": value,
                                   "ctc": ctc_log_likelihood(posterior, [PHONE_TO_ID[p] for p in phones]),
                                   "words": len(sentence), "unknown": len(unknown),
                                   "beam": sentence in {tuple(words(t)) for t in generated},
                                   "errors": edit_distance(reference, list(sentence))})
            clips.append({"clip_id": row["clip_id"], "reference_words": len(reference),
                          "beam_top_errors": edit_distance(reference, words(generated[0])),
                          "candidates": candidates})
        halves[name] = clips
    weights = min(itertools.product(*GRID), key=lambda w: errors(halves["select"], w))
    result = {"weights": dict(zip(("ctc_weight", "word_weight", "unknown_weight"), weights)),
              "grid": GRID, "run": str(args.run), "zero_soft": args.zero_soft}
    for name, clips in halves.items():
        count = sum(c["reference_words"] for c in clips)
        totals = {"posterior_decoder_beam_top": sum(c["beam_top_errors"] for c in clips),
                  "decoder_score_only": errors(clips, (0.0, 0.0, 0.0)),
                  "fitted": errors(clips, weights),
                  "oracle": sum(min(x["errors"] for x in c["candidates"]) for c in clips)}
        result[f"test_{name}"] = {"clips": len(clips), "reference_words": count,
                                  "mean_candidates": sum(len(c["candidates"]) for c in clips) / len(clips),
                                  **{k: {"word_errors": v, "wer": v / count} for k, v in totals.items()}}
    previous = {json.loads(l)["clip_id"]: words(json.loads(l)["text"]) for l in
                Path("checkpoints/posterior-headC-20261106/combined/test-report-selected.jsonl").read_text().splitlines()}
    references = {row["clip_id"]: words(row["transcript"]) for row in test.rows[half:]}
    paired = []
    for clip in halves["report"]:
        chosen = max(clip["candidates"], key=lambda c: score(c, weights))
        paired.append((edit_distance(references[clip["clip_id"]], previous[clip["clip_id"]]) - chosen["errors"],
                       clip["reference_words"]))
    rng = random.Random(20261107)
    samples = sorted(sum(d for d, _ in s) / sum(n for _, n in s)
                     for s in ([paired[rng.randrange(len(paired))] for _ in paired] for _ in range(10000)))
    result["test_report_vs_head_c_combined"] = {"interval_95": [samples[249], samples[9749]],
                                                "fraction_not_better": sum(x <= 0 for x in samples) / len(samples)}
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output / "test-report-selected.jsonl").open("w") as handle:
        for clip in halves["report"]:
            chosen = max(clip["candidates"], key=lambda c: score(c, weights))
            handle.write(json.dumps({"clip_id": clip["clip_id"], "text": " ".join(chosen["words_list"])}) + "\n")
    for name in ("test_select", "test_report"):
        print(name, {k: round(100 * v["wer"], 2) for k, v in result[name].items() if isinstance(v, dict)},
              "candidates/clip %.1f" % result[name]["mean_candidates"])
    print("weights", result["weights"], "vs head-C combined", result["test_report_vs_head_c_combined"])


if __name__ == "__main__":
    main()
