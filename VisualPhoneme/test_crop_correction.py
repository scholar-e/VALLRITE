"""Undo the geometry mismatch of the third-party LRS3 test crops at test time.

The supplied 96x96 test crops look ~15% more zoomed in and shifted down than
the Auto-AVSR training crops. Each crop is resampled with zoom ``s`` (< 1
shrinks the content, i.e. zooms out) and moved up by ``u`` pixels; missing
borders repeat edge pixels. The setting is chosen on test-select by the
published encoder's greedy word-CTC WER (independent of our phone head), then
test-report is scored once for identity and the chosen setting. Head C's greedy
phone PER is logged alongside. No training.
"""
import argparse
import itertools
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from VisualPhoneme.autoavsr_encoder import (AUTOAVSR_CROP, AUTOAVSR_MEAN, AUTOAVSR_STD,
                                            ctc_greedy_text, load_autoavsr_ctc, load_units)
from VisualPhoneme.evaluate_autoavsr_ctc import edit_counts, normalize_words
from VisualPhoneme.finetune_autoavsr_encoder import parse_source, read_rows
from VisualPhoneme.thermal import ThermalGuard
from VisualPhoneme.train_autoavsr_phone_head import build_head, greedy_phones

LOGGER = logging.getLogger(__name__)


def correct(frames: torch.Tensor, zoom: float, up: float) -> torch.Tensor:
    """uint8 [T,96,96] -> normalized [T,1,88,88] after zooming by ``zoom`` and moving content up ``up`` px."""
    video = frames.float().unsqueeze(1) / 255
    if zoom != 1 or up:
        size = video.shape[-1]
        theta = torch.tensor([[1 / zoom, 0, 0], [0, 1 / zoom, 2 * up / size]], dtype=video.dtype)
        grid = F.affine_grid(theta.expand(len(video), 2, 3), list(video.shape), align_corners=False)
        video = F.grid_sample(video, grid, padding_mode="border", align_corners=False)
    margin = (video.shape[-1] - AUTOAVSR_CROP) // 2
    video = video[..., margin:margin + AUTOAVSR_CROP, margin:margin + AUTOAVSR_CROP]
    return (video - AUTOAVSR_MEAN) / AUTOAVSR_STD


@torch.inference_mode()
def evaluate(model, head, rows, units, device, zoom, up) -> dict:
    totals = {"words": 0, "word_edits": 0, "phones": 0, "phone_edits": 0}
    order = sorted(range(len(rows)), key=lambda i: rows[i]["frames"])
    for start in range(0, len(order), 32):
        chunk = [rows[i] for i in order[start:start + 32]]
        videos = [correct(torch.from_numpy(np.load(row["video"])), zoom, up) for row in chunk]
        lengths = torch.tensor([len(v) for v in videos])
        batch = videos[0].new_zeros(len(videos), int(lengths.max()), *videos[0].shape[1:])
        for slot, video in enumerate(videos):
            batch[slot, :len(video)] = video
        features = model.encoder(batch.to(device), lengths.to(device))
        words = model.ctc_lo(features).log_softmax(-1).cpu()
        phones = head(features, lengths.to(device)).cpu()
        for slot, row in enumerate(chunk):
            frames = int(lengths[slot])
            reference = normalize_words(row["transcript"])
            totals["words"] += len(reference)
            totals["word_edits"] += sum(edit_counts(reference, normalize_words(
                ctc_greedy_text(words[slot, :frames], units))))
            totals["phones"] += len(row["phone_ids"])
            totals["phone_edits"] += sum(edit_counts(
                row["phone_ids"], greedy_phones(phones[slot, :frames * head.upsample])))
    return {"zoom": zoom, "up": up, "word_ctc_wer": totals["word_edits"] / totals["words"],
            "head_greedy_per": totals["phone_edits"] / totals["phones"], **totals}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/pretrained/autoavsr/LRS3_V_WER19.1/model.pth"))
    parser.add_argument("--units", type=Path,
                        default=Path("checkpoints/pretrained/autoavsr/tokens/unigram5000_units.txt"))
    parser.add_argument("--head", type=Path,
                        default=Path("checkpoints/autoavsr-phone-C-masked-20261106/best.pt"))
    parser.add_argument("--test", default="datasets/lrs3/manifests/test.jsonl@datasets/lrs3")
    parser.add_argument("--zooms", type=float, nargs="+", default=[1.0, 0.95, 0.9, 0.87, 0.83, 0.8])
    parser.add_argument("--ups", type=float, nargs="+", default=[0, 2, 4, 6, 8])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(filename)s:%(lineno)d %(message)s",
                        handlers=[logging.FileHandler(args.output / "sweep.log"), logging.StreamHandler()],
                        force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = read_rows(*parse_source(args.test))
    select, report = rows[:len(rows) // 2], rows[len(rows) // 2:]
    units = load_units(args.units)
    model = load_autoavsr_ctc(args.checkpoint).to(device).eval()
    saved = torch.load(args.head, map_location="cpu", weights_only=True)
    head = build_head(saved["config"], model.encoder.width)
    head.load_state_dict(saved["head"])
    head = head.to(device).eval()
    with ThermalGuard(args.output, device.type == "cuda"):
        grid = []
        for zoom, up in itertools.product(args.zooms, args.ups):
            result = evaluate(model, head, select, units, device, zoom, up)
            grid.append(result)
            LOGGER.info("test-select zoom %.2f up %.0f: word-CTC WER %.2f%% head C PER %.2f%%",
                        zoom, up, 100 * result["word_ctc_wer"], 100 * result["head_greedy_per"])
        best = min(grid, key=lambda r: (r["word_ctc_wer"], abs(1 - r["zoom"]) + r["up"]))
        final = {name: evaluate(model, head, report, units, device, zoom, up)
                 for name, zoom, up in (("identity", 1.0, 0), ("chosen", best["zoom"], best["up"]))}
        for name, result in final.items():
            LOGGER.info("test-report %s (zoom %.2f up %.0f): word-CTC WER %.2f%% head C PER %.2f%%",
                        name, result["zoom"], result["up"], 100 * result["word_ctc_wer"],
                        100 * result["head_greedy_per"])
    (args.output / "metrics.json").write_text(json.dumps(
        {"test_select_grid": grid, "chosen": {"zoom": best["zoom"], "up": best["up"]},
         "test_report": final}, indent=2) + "\n")


if __name__ == "__main__":
    main()
