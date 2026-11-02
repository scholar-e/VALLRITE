"""Cache frozen Auto-AVSR encoder features for phone-head training.

Writes ``<output>/<name>/features.npy`` (float16 [frames,768], clips
concatenated in manifest order) and ``index.jsonl`` (clip id, offset, frames,
phone ids). Inputs are 96x96 Auto-AVSR-style mouth crops (``npy-mouth`` rows).
"""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import time

import numpy as np
import torch

from VisualPhoneme.autoavsr_encoder import autoavsr_video_transform, load_autoavsr_ctc
from VisualPhoneme.data import PHONE_TO_ID
from VisualPhoneme.thermal import ThermalGuard

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/pretrained/autoavsr/LRS3_V_WER19.1/model.pth"))
    parser.add_argument("--manifest", action="append", required=True,
                        help="NAME=PATH; repeat for several splits")
    parser.add_argument("--data-root", type=Path, default=Path("datasets/lrs3"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(filename)s:%(lineno)d %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(args.output / "cache.log"),
                                  logging.StreamHandler()], force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_autoavsr_ctc(args.checkpoint).to(device).eval()
    encoder = model.encoder
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    with ThermalGuard(args.output, device.type == "cuda"), torch.inference_mode():
        for spec in args.manifest:
            name, path = spec.split("=", 1)
            path = Path(path)
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            destination = args.output / name
            destination.mkdir(exist_ok=True)
            total = sum(int(row["frames"]) for row in rows)
            features = np.lib.format.open_memmap(destination / "features.tmp.npy", mode="w+",
                                                 dtype=np.float16, shape=(total, encoder.width))
            offsets, offset = [], 0
            for row in rows:
                offsets.append(offset)
                offset += int(row["frames"])
            order = sorted(range(len(rows)), key=lambda index: rows[index]["frames"])
            started = time.monotonic()
            for start in range(0, len(order), args.batch_size):
                indices = order[start:start + args.batch_size]
                videos = [autoavsr_video_transform(torch.from_numpy(
                    np.load(args.data_root / rows[index]["video"], allow_pickle=False)))
                    for index in indices]
                for index, video in zip(indices, videos):
                    if len(video) != int(rows[index]["frames"]):
                        raise ValueError(f"{rows[index]['clip_id']}: {len(video)} frames, "
                                         f"manifest says {rows[index]['frames']}")
                lengths = torch.tensor([len(video) for video in videos])
                batch = videos[0].new_zeros(len(videos), int(lengths.max()), *videos[0].shape[1:])
                for slot, video in enumerate(videos):
                    batch[slot, :len(video)] = video
                encoded = encoder(batch.to(device), lengths.to(device)).half().cpu().numpy()
                for slot, index in enumerate(indices):
                    features[offsets[index]:offsets[index] + int(lengths[slot])] = \
                        encoded[slot, :int(lengths[slot])]
            features.flush()
            del features
            (destination / "features.tmp.npy").rename(destination / "features.npy")
            with (destination / "index.jsonl").open("w") as handle:
                for row, start in zip(rows, offsets):
                    handle.write(json.dumps({
                        "clip_id": row["clip_id"], "offset": start, "frames": int(row["frames"]),
                        "phone_ids": [PHONE_TO_ID[phone] for phone in row["phonemes"]],
                        "transcript": row["transcript"],
                    }) + "\n")
            (destination / "source.json").write_text(json.dumps({
                "manifest": str(path), "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "checkpoint_sha256": checkpoint_hash, "clips": len(rows), "frames": total,
                "dtype": "float16", "width": encoder.width,
            }, indent=2) + "\n")
            LOGGER.info("%s: %d clips, %d frames in %.0fs", name, len(rows), total,
                        time.monotonic() - started)


if __name__ == "__main__":
    main()
