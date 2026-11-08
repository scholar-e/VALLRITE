"""Export frame phone posteriors (and greedy phone paths) for the word decoder.

Training clips were seen by the Auto-AVSR encoder, so their features are much
cleaner than test features. To give the decoder realistic training inputs:

- cross-fitting: with ``--heads`` trained by ``--held-out-fold`` (one head per
  source-video fold), each clip is scored by the head that never saw it;
- degradation: Gaussian noise (``--noise`` × per-dimension feature std) and
  time masks are applied to the encoder features before the head. The strength
  is calibrated so phone error and confidence match test-select (``--calibrate``
  prints statistics for a subset without writing anything).

Writes ``log_posteriors.npy`` (fp16 [steps, classes]) and ``index.jsonl`` with
clip id, offset, steps, greedy phones, reference phones and transcript.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from VisualPhoneme.data import PHONEMES
from VisualPhoneme.evaluate_autoavsr_ctc import edit_counts
from VisualPhoneme.train_autoavsr_phone_head import (Split, build_head, greedy_phones, source_fold,
                                                     time_mask)


def load_heads(paths: list[Path], width: int, device: torch.device):
    heads = []
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        head = build_head(checkpoint["config"], width).to(device).eval()
        head.load_state_dict(checkpoint["head"])
        heads.append(head)
    return heads


@torch.inference_mode()
def export(split: Split, heads, indices: list[int], noise: float, masks: int, mask_width: int,
           seed: int, output: Path | None = None) -> dict:
    generator = np.random.default_rng(seed)
    torch_generator = torch.Generator(device=split.features.device).manual_seed(seed)
    std = split.features[:200_000].float().std(0)
    totals = {"phones": 0, "edits": 0, "steps": 0, "blank_steps": 0, "emitted_max_probability": 0.0,
              "emitted_steps": 0, "entropy": 0.0}
    chunks, offset = [], 0
    handle = (output / "index.jsonl").open("w") if output else None
    for start in range(0, len(indices), 64):
        chunk = indices[start:start + 64]
        features, lengths, _, _ = split.batch(chunk, heads[0].upsample)
        frames = lengths // heads[0].upsample
        if noise:
            features = features + noise * std * torch.randn(features.shape, generator=torch_generator,
                                                            device=features.device)
        features = time_mask(features, frames, masks, mask_width, generator)
        fold_of = [source_fold(split.rows[i]["clip_id"], len(heads)) if len(heads) > 1 else 0
                   for i in chunk]
        outputs = {fold: heads[fold](features, frames) for fold in set(fold_of)}
        for slot, index in enumerate(chunk):
            row = split.rows[index]
            path = outputs[fold_of[slot]][slot, :int(lengths[slot])].float()
            hypothesis = greedy_phones(path)
            probabilities = path.exp()
            emitted = path.argmax(-1) != 0
            totals["phones"] += len(row["phone_ids"])
            totals["edits"] += sum(edit_counts(row["phone_ids"], hypothesis))
            totals["steps"] += len(path)
            totals["blank_steps"] += int((~emitted).sum())
            totals["emitted_steps"] += int(emitted.sum())
            totals["emitted_max_probability"] += float(probabilities[emitted].max(-1).values.sum())
            totals["entropy"] += float(-(probabilities * path).sum())
            if handle:
                chunks.append(path.half().cpu().numpy())
                handle.write(json.dumps({
                    "clip_id": row["clip_id"], "offset": offset, "steps": len(path),
                    "greedy_phones": [PHONEMES[i - 1] for i in hypothesis],
                    "reference_phones": [PHONEMES[i - 1] for i in row["phone_ids"]],
                    "transcript": row["transcript"], "head_fold": fold_of[slot]}) + "\n")
                offset += len(path)
    if handle:
        handle.close()
        np.save(output / "log_posteriors.npy", np.concatenate(chunks))
    return {"clips": len(indices), "greedy_per": totals["edits"] / totals["phones"],
            "blank_rate": totals["blank_steps"] / totals["steps"],
            "emitted_mean_max_probability": totals["emitted_max_probability"] / max(1, totals["emitted_steps"]),
            "mean_entropy_nats": totals["entropy"] / totals["steps"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True, help="cached split directory")
    parser.add_argument("--heads", type=Path, nargs="+", required=True,
                        help="one head, or one per fold ordered by held-out fold")
    parser.add_argument("--rows", default=":", help="python slice over split rows")
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--time-masks", type=int, default=0)
    parser.add_argument("--time-mask-width", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--calibrate", type=float, nargs="*",
                        help="noise levels to report on the selected rows (writes nothing)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = Split(args.features, device)
    heads = load_heads(args.heads, split.features.shape[1], device)
    start, stop = (int(part) if part else None for part in args.rows.split(":"))
    indices = list(range(len(split.rows)))[start:stop]
    if args.calibrate is not None:
        for noise in args.calibrate:
            stats = export(split, heads, indices, noise, args.time_masks, args.time_mask_width, args.seed)
            print(json.dumps({"noise": noise, "time_masks": args.time_masks, **stats}))
        return
    if args.output is None:
        parser.error("--output is required unless --calibrate is given")
    args.output.mkdir(parents=True, exist_ok=True)
    stats = export(split, heads, indices, args.noise, args.time_masks, args.time_mask_width,
                   args.seed, args.output)
    stats.update({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                 heads=[str(path) for path in args.heads])
    (args.output / "metrics.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps({key: stats[key] for key in ("clips", "greedy_per", "blank_rate",
                                                  "emitted_mean_max_probability", "mean_entropy_nats")}))


if __name__ == "__main__":
    main()
