"""Word decoder that reads phone posteriors directly (soft posterior tokens).

The LoRA Llama decoder keeps its hard phone-path prompt and additionally sees
the phone head's frame log posteriors, compressed by a small convolutional
encoder into soft input embeddings placed between ``</PHONEMES>`` and
``<TEXT>``:

    <S2S>\\n<PHONEMES>\\n{greedy phones}\\n</PHONEMES>\\n [posterior tokens] <TEXT>\\n {words}<eos>

The encoder's output projection starts at zero, so training starts from the
existing matched-noise decoder plus inert extra positions. Training inputs are
cross-fitted, degraded train-clip posteriors (``export_phone_posteriors.py``);
checkpoints are selected by greedy WER on test-select; test-report is decoded
once with ``--evaluate-only``.
"""
import argparse
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from torch import nn

from CompetitiveWordDecoder.evaluate_vallr_qwen import vallr_prompt
from CompetitiveWordDecoder.train_vallr_clean import MODEL, REVISION, edit_distance
from PhonemeDecoder.evaluate_qwen_direct import words

LOGGER = logging.getLogger(__name__)
CLASSES = 40


class PosteriorEncoder(nn.Module):
    """[batch, steps, 40] log posteriors -> [batch, ~steps/4, hidden_size] embeddings."""

    def __init__(self, hidden_size: int, width: int = 512, stride: int = 4):
        super().__init__()
        self.stride = stride
        self.inp = nn.Linear(CLASSES, width)
        self.convs = nn.ModuleList([nn.Conv1d(width, width, 5, stride=2, padding=2)
                                    for _ in range(int(math.log2(stride)))])
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, hidden_size)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        for _ in self.convs:
            lengths = (lengths - 1) // 2 + 1
        return lengths

    def forward(self, log_posteriors: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.gelu(self.inp(log_posteriors.clamp(min=-20) / 10))
        x = x.transpose(1, 2)
        for conv in self.convs:
            x = torch.nn.functional.gelu(conv(x))
        return self.out(self.norm(x.transpose(1, 2)))


class PosteriorSet:
    """Exported posteriors (``export_phone_posteriors.py``) with transcripts."""

    def __init__(self, directory: Path, rows: slice = slice(None)):
        self.rows = [json.loads(line) for line in (directory / "index.jsonl").read_text().splitlines()][rows]
        self.posteriors = np.load(directory / "log_posteriors.npy", mmap_mode="r")

    def posterior(self, row: dict) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.posteriors[row["offset"]:row["offset"] + row["steps"]],
                                           dtype=np.float32))


def prompt_parts(row: dict) -> tuple[str, str]:
    prompt = vallr_prompt(row["greedy_phones"])
    head, tail = prompt.split("<TEXT>")
    return head, "<TEXT>" + tail


class Batcher:
    """Builds padded input embeddings, attention masks and labels."""

    def __init__(self, model, tokenizer, encoder: PosteriorEncoder):
        self.model, self.tokenizer, self.encoder = model, tokenizer, encoder
        self.zero_soft = False  # control: same positions, no posterior information
        self.embed = model.get_input_embeddings()
        self.eos = tokenizer.eos_token_id

    def pieces(self, data: PosteriorSet, rows: list[dict], with_targets: bool):
        device = self.embed.weight.device
        posteriors = [data.posterior(row) for row in rows]
        lengths = torch.tensor([len(p) for p in posteriors])
        padded = torch.full((len(rows), int(lengths.max()), CLASSES), -20.0)
        for slot, p in enumerate(posteriors):
            padded[slot, :len(p)] = p
        soft = self.encoder(padded.to(device)).to(self.embed.weight.dtype)
        if self.zero_soft:
            soft = torch.zeros_like(soft)
        soft_lengths = self.encoder.output_lengths(lengths).tolist()
        sequences = []
        for slot, row in enumerate(rows):
            head, tail = prompt_parts(row)
            head_ids = self.tokenizer.encode(head, add_special_tokens=False)
            tail_ids = self.tokenizer.encode(tail, add_special_tokens=False)
            target = (self.tokenizer.encode(" ".join(words(row["transcript"])), add_special_tokens=False)
                      + [self.eos]) if with_targets else []
            ids = torch.tensor(head_ids + tail_ids + target, device=device)
            embedded = self.embed(ids)
            parts = [embedded[:len(head_ids)], soft[slot, :soft_lengths[slot]], embedded[len(head_ids):]]
            labels = [-100] * (len(head_ids) + soft_lengths[slot] + len(tail_ids)) + target
            sequences.append((torch.cat(parts), labels))
        return sequences

    def training_batch(self, data, rows):
        sequences = self.pieces(data, rows, True)
        longest = max(len(s) for s, _ in sequences)
        hidden = sequences[0][0].shape[-1]
        embeds = sequences[0][0].new_zeros(len(rows), longest, hidden)
        mask = torch.zeros(len(rows), longest, dtype=torch.long, device=embeds.device)
        labels = torch.full((len(rows), longest), -100, dtype=torch.long, device=embeds.device)
        for slot, (sequence, label) in enumerate(sequences):
            embeds[slot, :len(sequence)] = sequence
            mask[slot, :len(sequence)] = 1
            labels[slot, :len(label)] = torch.tensor(label, device=embeds.device)
        return embeds, mask, labels

    @torch.inference_mode()
    def generate(self, data, rows, max_new_tokens: int = 64) -> list[str]:
        sequences = self.pieces(data, rows, False)
        longest = max(len(s) for s, _ in sequences)
        hidden = sequences[0][0].shape[-1]
        pad = self.embed(torch.tensor([self.eos], device=self.embed.weight.device))[0]
        embeds = pad.expand(len(rows), longest, hidden).clone()
        mask = torch.zeros(len(rows), longest, dtype=torch.long, device=embeds.device)
        for slot, (sequence, _) in enumerate(sequences):  # left padding
            embeds[slot, longest - len(sequence):] = sequence
            mask[slot, longest - len(sequence):] = 1
        output = self.model.generate(inputs_embeds=embeds, attention_mask=mask, do_sample=False,
                                     max_new_tokens=max_new_tokens, eos_token_id=self.eos,
                                     pad_token_id=self.eos, use_cache=True)
        return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output.tolist()]


@torch.inference_mode()
def teacher_forced_loss(batcher: Batcher, data: PosteriorSet, rows: list[dict], batch_size: int = 16) -> dict:
    """Mean target-token NLL with real posteriors and with soft tokens zeroed."""
    batcher.model.eval()
    batcher.encoder.eval()
    result = {}
    for name, zero in (("posteriors", False), ("zeroed", True)):
        batcher.zero_soft = zero
        total = tokens = 0.0
        for start in range(0, len(rows), batch_size):
            embeds, mask, labels = batcher.training_batch(data, rows[start:start + batch_size])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = batcher.model(inputs_embeds=embeds, attention_mask=mask).logits.float()
            shifted = labels[:, 1:]
            losses = torch.nn.functional.cross_entropy(logits[:, :-1].transpose(1, 2), shifted,
                                                       ignore_index=-100, reduction="sum")
            total += float(losses)
            tokens += int((shifted != -100).sum())
        result[name] = total / tokens
    batcher.zero_soft = False
    batcher.model.train()
    batcher.encoder.train()
    return result


def evaluate(batcher: Batcher, data: PosteriorSet, rows: list[dict], path: Path | None = None,
             batch_size: int = 32) -> dict:
    batcher.model.eval()
    batcher.encoder.eval()
    errors = count = 0
    handle = path.open("w") if path else None
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        for row, text in zip(chunk, batcher.generate(data, chunk), strict=True):
            reference = words(row["transcript"])
            edits = edit_distance(reference, words(text))
            errors += edits
            count += len(reference)
            if handle:
                handle.write(json.dumps({"clip_id": row["clip_id"], "text": text,
                                         "reference": " ".join(reference), "errors": edits}) + "\n")
    if handle:
        handle.close()
    batcher.model.train()
    batcher.encoder.train()
    return {"clips": len(rows), "reference_words": count, "word_errors": errors, "wer": errors / count}


def load_model(adapter: Path, trainable: bool):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, cache_dir=".cache/huggingface/hub",
                                                 local_files_only=True, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa")
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)  # adapter stores embeddings
    model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable, local_files_only=True).cuda()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, nargs="+", help="exported train-clip posteriors (one or more)")
    parser.add_argument("--train-weights", type=float, nargs="+",
                        help="sampling weight per --train set (default: proportional to size)")
    parser.add_argument("--test", type=Path, required=True, help="exported test posteriors (head C)")
    parser.add_argument("--adapter", type=Path, required=True,
                        help="initial LoRA adapter, or a saved run's adapter with --evaluate-only")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--encoder-learning-rate", type=float, default=5e-4)
    parser.add_argument("--lora-learning-rate", type=float, default=2e-5)
    parser.add_argument("--steps", type=int, default=3800)
    parser.add_argument("--eval-every", type=int, default=400)
    parser.add_argument("--patience", type=int, default=3, help="evaluations without improvement")
    parser.add_argument("--max-minutes", type=float, default=120)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--freeze-lora", action="store_true", help="train only the posterior encoder")
    parser.add_argument("--select-limit", type=int, help="smoke tests: evaluate only the first N select clips")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="decode test-report once with --adapter and its encoder.pt")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(filename)s:%(lineno)d %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(args.output / "train.log"), logging.StreamHandler()],
                        force=True)
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    torch.manual_seed(args.seed)
    test = PosteriorSet(args.test)
    half = len(test.rows) // 2
    selection, report = test.rows[:half][:args.select_limit], test.rows[half:]
    model, tokenizer = load_model(args.adapter, trainable=not args.evaluate_only)
    encoder = PosteriorEncoder(model.config.hidden_size).cuda()
    if args.evaluate_only:
        encoder.load_state_dict(torch.load(args.adapter.parent / "encoder.pt", weights_only=True))
        batcher = Batcher(model, tokenizer, encoder)
        result = {"test_select": evaluate(batcher, test, selection, args.output / "test-select.jsonl"),
                  "test_report": evaluate(batcher, test, report, args.output / "test-report.jsonl"),
                  "adapter": str(args.adapter), "finished_utc": datetime.now(timezone.utc).isoformat()}
        (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        LOGGER.info("select WER %.2f%%, report WER %.2f%%", 100 * result["test_select"]["wer"],
                    100 * result["test_report"]["wer"])
        return
    sources = [PosteriorSet(path) for path in args.train]
    weights = args.train_weights or [len(source.rows) for source in sources]
    if len(weights) != len(sources):
        parser.error("--train-weights needs one value per --train set")
    model.config.use_cache = False
    batcher = Batcher(model, tokenizer, encoder)
    lora = [p for p in model.parameters() if p.requires_grad]
    if args.freeze_lora:
        for parameter in lora:
            parameter.requires_grad_(False)
        lora = []
    for parameter in lora:  # keep trainable LoRA weights in fp32 so small updates survive
        parameter.data = parameter.data.float()
    groups = [{"params": list(encoder.parameters()), "lr": args.encoder_learning_rate}]
    if lora:
        groups.append({"params": lora, "lr": args.lora_learning_rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    warmup = max(1, args.steps // 20)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(
        (step + 1) / warmup, 0.5 * (1 + math.cos(math.pi * min(step, args.steps) / args.steps))))
    config = {**{k: str(v) if isinstance(v, Path) else [str(x) for x in v] if k == "train" else v
                 for k, v in vars(args).items()},
              "train_clips": [len(source.rows) for source in sources], "test_select_clips": len(selection),
              "lora_parameters": sum(p.numel() for p in lora),
              "encoder_parameters": sum(p.numel() for p in encoder.parameters())}
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    LOGGER.info("Config: %s", config)
    sampler = random.Random(args.seed)
    orders = [sampler.sample(range(len(source.rows)), len(source.rows)) for source in sources]
    positions = [0] * len(sources)
    model.train()
    history, best = [], None
    started = time.monotonic()
    baseline = evaluate(batcher, test, selection)
    history.append({"step": 0, "test_select": baseline,
                    "test_select_loss": teacher_forced_loss(batcher, test, selection)})
    LOGGER.info("step 0 test-select WER %.2f%% loss %s", 100 * baseline["wer"], history[-1]["test_select_loss"])
    best = (baseline["wer"], 0)
    stale, running = 0, []
    step = 0
    for step in range(1, args.steps + 1):
        if (stopping or (args.output / "PAUSE").exists() or Path("PAUSE").exists()
                or time.monotonic() - started > args.max_minutes * 60):
            LOGGER.warning("Stopped at step %d of %d; run is incomplete", step - 1, args.steps)
            break
        # One source per batch (rows of a batch share one posterior file); sources drawn by weight.
        which = sampler.choices(range(len(sources)), weights)[0]
        train, order = sources[which], orders[which]
        rows = [train.rows[order[(positions[which] + i) % len(order)]] for i in range(args.batch_size)]
        positions[which] += args.batch_size
        embeds, mask, labels = batcher.training_batch(train, rows)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(inputs_embeds=embeds, attention_mask=mask, labels=labels).loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(list(encoder.parameters()) + lora, 1.0)
        optimizer.step()
        scheduler.step()
        running.append(loss.item())
        if step % 50 == 0:
            LOGGER.info("step %d loss %.4f (%.0fs)", step, sum(running) / len(running),
                        time.monotonic() - started)
            running = []
        if step % args.eval_every == 0 or step == args.steps:
            result = evaluate(batcher, test, selection)
            losses = teacher_forced_loss(batcher, test, selection)
            history.append({"step": step, "test_select": result, "test_select_loss": losses,
                            "seconds": time.monotonic() - started})
            LOGGER.info("step %d test-select WER %.2f%% loss with posteriors %.4f, zeroed %.4f",
                        step, 100 * result["wer"], losses["posteriors"], losses["zeroed"])
            if result["wer"] < best[0]:
                best, stale = (result["wer"], step), 0
                model.save_pretrained(args.output / "best" / "adapter")
                tokenizer.save_pretrained(args.output / "best" / "adapter")
                torch.save(encoder.state_dict(), args.output / "best" / "encoder.pt")
            else:
                stale += 1
                if stale >= args.patience:
                    LOGGER.info("No test-select improvement in %d evaluations; stopping", stale)
                    break
    model.save_pretrained(args.output / "last" / "adapter")
    tokenizer.save_pretrained(args.output / "last" / "adapter")
    torch.save(encoder.state_dict(), args.output / "last" / "encoder.pt")
    summary = {"best_step": best[1], "best_test_select_wer": best[0], "steps_completed": step,
               "steps_planned": args.steps, "history": history, "seconds": time.monotonic() - started,
               "finished_utc": datetime.now(timezone.utc).isoformat()}
    (args.output / "training-metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    LOGGER.info("Training finished: best step %d, test-select WER %.2f%%", best[1], 100 * best[0])


if __name__ == "__main__":
    main()
