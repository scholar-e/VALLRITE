"""Cross-validate a compact Transformer ranker over phoneme N-best candidates."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import random
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from VisualPhoneme.rerank_nbest import configure_logging, edit_distance, source_fold

LOGGER = logging.getLogger("visual_phoneme.context_reranker")


class CandidateLists(Dataset):
    def __init__(self, rows: list[dict], vocabulary: tuple[str, ...]):
        token_id = {token: index + 1 for index, token in enumerate(vocabulary)}
        self.items = []
        for row in rows:
            candidates = row["phone_hypotheses"]
            reference = row["reference_phones"]
            self.items.append({
                "clip_id": row["clip_id"], "reference_length": len(reference),
                "tokens": [[token_id[token] for token in item["phones"]]
                           for item in candidates],
                "scores": [item["ctc_log_score"] for item in candidates],
                "errors": [edit_distance(reference, item["phones"]) for item in candidates],
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate_lists(items: list[dict]):
    batch, nbest = len(items), len(items[0]["tokens"])
    maximum = max(len(tokens) for item in items for tokens in item["tokens"])
    tokens = torch.zeros(batch, nbest, maximum, dtype=torch.long)
    mask = torch.zeros(batch, nbest, maximum, dtype=torch.bool)
    scores = torch.tensor([item["scores"] for item in items], dtype=torch.float32)
    lengths = torch.zeros(batch, nbest)
    for row, item in enumerate(items):
        for column, sequence in enumerate(item["tokens"]):
            tokens[row, column, :len(sequence)] = torch.tensor(sequence)
            mask[row, column, :len(sequence)] = True
            lengths[row, column] = len(sequence)
    def standardize(values):
        return (values - values.mean(1, keepdim=True)) / values.std(1, keepdim=True).clamp_min(1e-5)
    score_per_token = scores / lengths.clamp_min(1)
    side = torch.stack((standardize(scores), standardize(score_per_token),
                        standardize(lengths)), dim=-1)
    utility = torch.tensor([[-error / max(item["reference_length"], 1)
                             for error in item["errors"]] for item in items])
    return tokens, mask, side, utility, items


class ContextRanker(nn.Module):
    def __init__(self, vocabulary: int, dimensions: int = 64, layers: int = 2,
                 heads: int = 4, dropout: float = 0.1, max_length: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary + 1, dimensions, padding_idx=0)
        self.position = nn.Embedding(max_length, dimensions)
        layer = nn.TransformerEncoderLayer(dimensions, heads, dimensions * 2, dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.output = nn.Sequential(nn.LayerNorm(dimensions + 3),
                                    nn.Linear(dimensions + 3, dimensions), nn.SiLU(),
                                    nn.Dropout(dropout), nn.Linear(dimensions, 1))

    def forward(self, tokens, mask, side):
        batch, nbest, length = tokens.shape
        flat = tokens.reshape(batch * nbest, length)
        valid = mask.reshape(batch * nbest, length)
        positions = torch.arange(length, device=tokens.device)
        encoded = self.encoder(self.embedding(flat) + self.position(positions),
                               src_key_padding_mask=~valid)
        pooled = (encoded * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)
        joined = torch.cat((pooled.reshape(batch, nbest, -1), side), dim=-1)
        return self.output(joined).squeeze(-1)


def rank_loss(prediction, utility):
    regression = nn.functional.smooth_l1_loss(prediction, utility)
    target_difference = utility.unsqueeze(2) - utility.unsqueeze(1)
    prediction_difference = prediction.unsqueeze(2) - prediction.unsqueeze(1)
    selected = target_difference != 0
    pairwise = nn.functional.binary_cross_entropy_with_logits(
        prediction_difference[selected], (target_difference[selected] > 0).float())
    return regression + pairwise


def evaluate(model, loader, device, top_n):
    phones = baseline = ranked = top1 = 0
    model.eval()
    with torch.inference_mode():
        for tokens, mask, side, _, items in loader:
            prediction = model(tokens.to(device), mask.to(device), side.to(device)).cpu()
            for row, item in enumerate(items):
                order = prediction[row].argsort(descending=True).tolist()
                phones += item["reference_length"]
                baseline += min(item["errors"][:top_n])
                ranked += min(item["errors"][index] for index in order[:top_n])
                top1 += item["errors"][order[0]]
    return baseline, ranked, top1, phones


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if not args.input.is_file() or args.folds < 2 or args.epochs < 1:
        parser.error("invalid input, folds, or epochs")
    configure_logging(args.output.with_suffix(".log"))
    global LOGGER
    LOGGER = logging.getLogger("visual_phoneme.rerank")
    random.seed(17)
    torch.manual_seed(17)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device("cpu" if device_name == "auto" else device_name)
    rows = [json.loads(line) for line in args.input.read_text().splitlines()]
    vocabulary = tuple(sorted({phone for row in rows for item in row["phone_hypotheses"]
                               for phone in item["phones"]}))
    dataset = CandidateLists(rows, vocabulary)
    totals = torch.zeros(4, dtype=torch.long)
    for fold in range(args.folds):
        train_rows = [item for item in dataset.items
                      if source_fold(item["clip_id"], args.folds) != fold]
        held_rows = [item for item in dataset.items
                     if source_fold(item["clip_id"], args.folds) == fold]
        train_data = CandidateLists.__new__(CandidateLists); train_data.items = train_rows
        held_data = CandidateLists.__new__(CandidateLists); held_data.items = held_rows
        generator = torch.Generator().manual_seed(17 + fold)
        train_loader = DataLoader(train_data, args.batch_size, shuffle=True,
                                  collate_fn=collate_lists, generator=generator)
        held_loader = DataLoader(held_data, args.batch_size, collate_fn=collate_lists)
        model = ContextRanker(len(vocabulary)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), args.learning_rate,
                                      weight_decay=args.weight_decay)
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_total = 0.0
            for tokens, mask, side, utility, _ in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = rank_loss(model(tokens.to(device), mask.to(device), side.to(device)),
                                 utility.to(device))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_total += float(loss.detach())
            if epoch == 1 or epoch % 5 == 0:
                LOGGER.info("fold=%d epoch=%d loss=%.5f", fold, epoch,
                            loss_total / len(train_loader))
        totals += torch.tensor(evaluate(model, held_loader, device, args.top_n))
    baseline, ranked, top1, phones = totals.tolist()
    report = {"format": "visual-phoneme-context-reranker-cv-0.1",
              "clips": len(dataset), "folds": args.folds, "epochs": args.epochs,
              "parameters": sum(p.numel() for p in model.parameters()),
              "reference_phones": phones, "baseline_oracle_per_at_n": baseline / phones,
              "reranked_oracle_per_at_n": ranked / phones,
              "reranked_top1_per": top1 / phones,
              "evaluation": "source-disjoint cross-validation"}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    LOGGER.info("complete baseline=%.4f%% reranked=%.4f%% top1=%.4f%%",
                100 * baseline / phones, 100 * ranked / phones, 100 * top1 / phones)
    for handler in LOGGER.handlers: handler.flush()


if __name__ == "__main__":
    main()
