"""Word decoding steered by phone posteriors (acoustic shallow fusion).

The LoRA word decoder is prompted with the top phone path as usual, but its
beam search is scored jointly with the phone head's frame posteriors. Whenever
a hypothesis completes a word, the CMUdict phones of all completed words are
scored by their CTC prefix probability; at end of sentence the exact CTC
sequence probability is used:

    score = log P_LLM(tokens | phone prompt) + ctc_weight * log P_CTC(phones(W) | video)
            + word_bonus * |W| + unknown_penalty * |unknown words|

Unlike reranking a fixed candidate list, the decoder can reach sentences no
phone path spelled out, as long as their pronunciation fits the posteriors.
Every finished hypothesis is saved with its separate score terms so weights
can be refitted afterwards without decoding again.
"""
import argparse
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import signal
import time

import torch

from CompetitiveWordDecoder.evaluate_vallr_qwen import vallr_prompt
from CompetitiveWordDecoder.posterior_rescoring import phone_posteriors
from CompetitiveWordDecoder.train_vallr_clean import MODEL, REVISION, edit_distance
from PhonemeDecoder.evaluate_qwen_direct import words
from VisualPhoneme.autoavsr_decoding import _ctc_extend
from VisualPhoneme.data import PHONE_TO_ID
from VisualPhoneme.prepare_lrs3 import transcript_phones

LOGGER = logging.getLogger(__name__)


class Pronunciations:
    """Cached word -> phone ids (empty for words CMUdict does not know)."""

    def __init__(self):
        self.cache: dict[str, list[int]] = {}

    def __call__(self, word: str) -> list[int]:
        if word not in self.cache:
            phones, _, _ = transcript_phones(word)
            self.cache[word] = [PHONE_TO_ID[phone] for phone in phones]
        return self.cache[word]


def initial_ctc_state(log_ctc: torch.Tensor) -> torch.Tensor:
    """[T,2] (log r^n, log r^b) of the empty prefix."""
    return torch.stack((torch.full((len(log_ctc),), -math.inf), log_ctc[:, 0].cumsum(0)), dim=1)


def extend_many(log_ctc: torch.Tensor, states: torch.Tensor, last: torch.Tensor,
                prefix: torch.Tensor, sequences: list[list[int]]):
    """Extend N prefixes by their own phone sequences (batched one phone at a time).

    states [N,T,2], last [N], prefix [N] (current prefix log probabilities).
    Returns updated (states, last, prefix); empty sequences are left unchanged.
    """
    states, last, prefix = states.clone(), last.clone(), prefix.clone()
    for position in range(max((len(s) for s in sequences), default=0)):
        active = [i for i, sequence in enumerate(sequences) if len(sequence) > position]
        index = torch.tensor(active)
        phones = torch.tensor([sequences[i][position] for i in active])[:, None]
        new_states, new_prefix = _ctc_extend(log_ctc, states[index], last[index], phones)
        states[index] = new_states[:, 0]
        prefix[index] = new_prefix[:, 0]
        last[index] = phones[:, 0]
    return states, last, prefix


def sequence_log_probability(states: torch.Tensor) -> torch.Tensor:
    """Exact CTC log P(prefix is the whole sequence) from [N,T,2] states."""
    return torch.logaddexp(states[:, -1, 0], states[:, -1, 1])


class Hypothesis:
    __slots__ = ("tokens", "text", "llm", "words", "unknown", "ctc", "state", "last")

    def __init__(self, tokens, text, llm, words_done, unknown, ctc, state, last):
        self.tokens, self.text, self.llm = tokens, text, llm
        self.words, self.unknown, self.ctc, self.state, self.last = (
            words_done, unknown, ctc, state, last)

    def score(self, weights: dict) -> float:
        return (self.llm + weights["ctc"] * self.ctc + weights["words"] * len(self.words)
                + weights["unknown"] * self.unknown)


@torch.inference_mode()
def fused_beam_search(model, tokenizer, prompt: str, log_ctc: torch.Tensor,
                      pronounce: Pronunciations, weights: dict, beam: int = 8,
                      token_candidates: int = 8, max_new_tokens: int = 64) -> list[dict]:
    eos = tokenizer.eos_token_id
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    alive = [Hypothesis([], "", 0.0, [], 0, 0.0, initial_ctc_state(log_ctc), -1)]
    finished: list[Hypothesis] = []
    pieces: dict[int, str] = {}
    # Encode the prompt once; afterwards feed one token per beam and reorder the cache.
    output = model(input_ids=torch.tensor([prompt_ids], device=model.device), use_cache=True)
    cache = output.past_key_values
    logits = output.logits[:, -1].float().log_softmax(-1)
    for _ in range(max_new_tokens):
        top_values, top_tokens = logits.topk(token_candidates, dim=-1)
        expansions = []
        for slot, parent in enumerate(alive):
            base = parent.score(weights)
            for value, token in zip(top_values[slot].tolist(), top_tokens[slot].tolist()):
                expansions.append((base + value, slot, token, value))
        expansions.sort(key=lambda item: -item[0])
        expansions = expansions[:2 * beam]
        # Tokens that start a new word (or end the sentence) complete the previous word.
        completing, children = [], []
        for _, slot, token, value in expansions:
            parent = alive[slot]
            if token not in pieces:
                pieces[token] = tokenizer.decode([token])
            piece = "" if token == eos else pieces[token]
            text = parent.text + piece
            done = words(parent.text) if (token == eos or piece.startswith(" ")) else parent.words
            new_words = done[len(parent.words):] if len(done) >= len(parent.words) else []
            child = Hypothesis(parent.tokens + [token], text, parent.llm + value,
                               parent.words + new_words, parent.unknown, parent.ctc,
                               parent.state, parent.last)
            if new_words or token == eos:
                phones = [p for word in new_words for p in pronounce(word)]
                child.unknown += sum(not pronounce(word) for word in new_words)
                completing.append((child, phones))
            children.append((child, token == eos, slot))
        if completing:
            states = torch.stack([child.state for child, _ in completing])
            last = torch.tensor([child.last for child, _ in completing])
            prefix = torch.tensor([child.ctc for child, _ in completing])
            states, last, prefix = extend_many(log_ctc, states, last, prefix,
                                               [phones for _, phones in completing])
            final = sequence_log_probability(states)
            for row, (child, _) in enumerate(completing):
                child.state, child.last = states[row], int(last[row])
                ended = child.tokens[-1] == eos
                child.ctc = float(final[row] if ended else prefix[row])
        next_alive, parents = [], []
        for child, ended, slot in sorted(children, key=lambda item: -item[0].score(weights)):
            if ended:
                finished.append(child)
            elif len(next_alive) < beam:
                next_alive.append(child)
                parents.append(slot)
        alive = next_alive
        best_finished = max((h.score(weights) for h in finished), default=-math.inf)
        if (len(finished) >= beam or not alive
                or (weights["words"] <= 0 and alive[0].score(weights) < best_finished)):
            break
        cache.reorder_cache(torch.tensor(parents, device=model.device))
        output = model(input_ids=torch.tensor([[h.tokens[-1]] for h in alive], device=model.device),
                       past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        logits = output.logits[:, -1].float().log_softmax(-1)
    if not finished:  # generation cap: score the best open hypothesis as a whole sentence
        for hypothesis in alive[:1]:
            done = words(hypothesis.text)
            new_words = done[len(hypothesis.words):]
            states, _, _ = extend_many(log_ctc, hypothesis.state[None], torch.tensor([hypothesis.last]),
                                       torch.tensor([hypothesis.ctc]),
                                       [[p for word in new_words for p in pronounce(word)]])
            hypothesis.words = done
            hypothesis.ctc = float(sequence_log_probability(states)[0])
            finished.append(hypothesis)
    finished.sort(key=lambda h: -h.score(weights))
    return [{"text": " ".join(h.words), "llm": h.llm, "ctc": h.ctc, "words": len(h.words),
             "unknown": h.unknown, "ended": bool(h.tokens and h.tokens[-1] == eos),
             "score": h.score(weights)} for h in finished]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypotheses", type=Path, required=True,
                        help="phone-hypotheses.jsonl; the rank-1 path is the prompt")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--head", type=Path,
                        default=Path("checkpoints/autoavsr-phone-context2-20261102/best.pt"))
    parser.add_argument("--features", type=Path,
                        default=Path("checkpoints/autoavsr-features-20261102/test"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ctc-weight", type=float, default=1.0)
    parser.add_argument("--word-bonus", type=float, default=0.0)
    parser.add_argument("--unknown-penalty", type=float, default=-5.0)
    parser.add_argument("--beam", type=int, default=8)
    parser.add_argument("--token-candidates", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(filename)s:%(lineno)d %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(args.output / "decode.log"),
                                  logging.StreamHandler()], force=True)
    weights = {"ctc": args.ctc_weight, "words": args.word_bonus, "unknown": args.unknown_penalty}
    (args.output / "config.json").write_text(json.dumps(
        {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
         "weights": weights, "model": MODEL, "revision": REVISION}, indent=2) + "\n")
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    records = [json.loads(line) for line in args.hypotheses.read_text().splitlines()][:args.limit]
    posteriors = phone_posteriors(args.head, args.features)
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
                                                 cache_dir=".cache/huggingface/hub",
                                                 local_files_only=True, dtype=torch.float16,
                                                 attn_implementation="sdpa")
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True).cuda().eval()
    pronounce = Pronunciations()
    output_path = args.output / "nbest.jsonl"
    done = set()
    if output_path.exists():
        done = {json.loads(line)["clip_id"] for line in output_path.read_text().splitlines()}
    started = time.monotonic()
    with output_path.open("a") as handle:
        for count, row in enumerate(records, 1):
            if row["clip_id"] in done:
                continue
            if stopping or (args.output / "PAUSE").exists() or Path("PAUSE").exists():
                LOGGER.warning("Stopped after %d clips; resumable", count - 1)
                return
            nbest = fused_beam_search(model, tokenizer, vallr_prompt(row["phone_hypotheses"][0]["phones"]),
                                      posteriors[row["clip_id"]], pronounce, weights, args.beam,
                                      args.token_candidates)
            handle.write(json.dumps({"clip_id": row["clip_id"], "reference_words": row["reference_words"],
                                     "nbest": nbest}) + "\n")
            handle.flush()
            if count % 50 == 0:
                LOGGER.info("decoded %d/%d clips in %.0fs", count, len(records),
                            time.monotonic() - started)
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    reference_count = errors = 0
    for row in rows:
        reference = words(" ".join(row["reference_words"]))
        reference_count += len(reference)
        errors += edit_distance(reference, words(row["nbest"][0]["text"]))
    metrics = {"clips": len(rows), "reference_words": reference_count, "word_errors": errors,
               "wer": errors / reference_count, "seconds": time.monotonic() - started,
               "finished_utc": datetime.now(timezone.utc).isoformat()}
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    LOGGER.info("Complete: WER %.2f%% on %d clips", 100 * metrics["wer"], len(rows))


if __name__ == "__main__":
    main()
