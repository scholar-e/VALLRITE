"""Utterance-level Auto-AVSR mouth crops from Seamless Interaction tarballs.

Seamless Interaction (Meta, CC-BY-NC 4.0, huggingface.co/datasets/facebook/
seamless-interaction) ships, per participant and interaction, a 1080x1920 30 fps
face video, word-timed automatic transcripts and COCO-WholeBody keypoints.
Auto-AVSR never saw it, so its phone posteriors on these clips have realistic
(unseen-video) errors.

Steps per participant file, read straight from the tar:

- split transcript segments into utterances at pauses longer than ``--max-gap``
  seconds or when longer than ``--max-seconds``;
- keep utterances of ``--min-words``+ words whose words are all alphabetic,
  have CMUdict pronunciations and alignment score >= ``--min-word-score``, and
  whose frames all have a valid face box and confident mouth keypoints;
- resample to 25 fps (nearest frame) and cut 96x96 mouth crops with the same
  alignment as ``autoavsr_crops`` (keypoints 23-90 are the 68 iBUG face points).

Writes ``<output>/crops/<participant file>/<n>.npy`` and a manifest
``<output>/manifest.jsonl`` compatible with ``cache_autoavsr_features``.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import io
import json
import logging
from pathlib import Path
import re
import tarfile

import av
import numpy as np

from VisualPhoneme.autoavsr_crops import MEAN_FACE, crop_mouths
from VisualPhoneme.prepare_lrs3 import transcript_phones

LOGGER = logging.getLogger(__name__)
FACE = slice(23, 91)
MOUTH_CONFIDENCE = slice(23 + 48, 23 + 68)
SOURCE_FPS = 30
TARGET_FPS = 25


def utterances(segments: list[dict], max_gap: float, max_seconds: float) -> list[list[dict]]:
    """Group timed words into utterances, cutting at long pauses or a length cap."""
    groups = []
    for segment in segments:
        current = []
        for word in segment.get("words", []):
            if word.get("start") is None or word.get("end") is None:
                current = []  # untimed word: drop the partial utterance around it
                continue
            if current and (word["start"] - current[-1]["end"] > max_gap
                            or word["end"] - current[0]["start"] > max_seconds):
                groups.append(current)
                current = []
            current.append(word)
        if current:
            groups.append(current)
    return groups


def clean_word(word: str) -> str | None:
    """Lowercase and strip edge punctuation; None if the word is not alphabetic."""
    value = word.lower().replace("’", "'").strip(".,!?;:\"()-–—…")
    return value if re.fullmatch(r"[a-z]+(?:'[a-z]+)*", value) else None


def check_words(words: list[dict], min_words: int, min_score: float) -> tuple[str, list[str]] | str:
    """Return (transcript, phones) or a rejection reason."""
    if len(words) < min_words:
        return "too_few_words"
    cleaned = [clean_word(word["word"]) for word in words]
    if any(value is None for value in cleaned):
        return "non_alphabetic"
    if min(word.get("score", 0.0) for word in words) < min_score:
        return "low_word_score"
    transcript = " ".join(cleaned)
    phones, unknown, _ = transcript_phones(transcript)
    if unknown:
        return "unknown_word"
    return transcript, phones


def _process(job: tuple[str, str, str, dict]) -> list[dict]:
    tar_path, stem, output, options = job
    output = Path(output)
    with tarfile.open(tar_path) as archive:
        meta = json.load(archive.extractfile(stem + ".json"))
        npz = np.load(io.BytesIO(archive.extractfile(stem + ".npz").read()))
        video_bytes = archive.extractfile(stem + ".mp4").read()
    keypoints = npz["boxes_and_keypoints:keypoints"]
    valid_box = npz["boxes_and_keypoints:is_valid_box"]
    candidates, rows = [], []
    for words in utterances(meta["metadata:transcript"], options["max_gap"], options["max_seconds"]):
        checked = check_words(words, options["min_words"], options["min_word_score"])
        if isinstance(checked, str):
            rows.append({"stem": stem, "status": checked})
            continue
        start = max(0.0, words[0]["start"] - options["pad"])
        end = words[-1]["end"] + options["pad"]
        indices = np.round(np.arange(start, end, 1 / TARGET_FPS) * SOURCE_FPS).astype(int)
        if len(indices) < options["min_frames"] or indices[-1] >= len(keypoints):
            rows.append({"stem": stem, "status": "bad_length"})
            continue
        if not valid_box[indices].all() or \
                np.median(keypoints[indices][:, MOUTH_CONFIDENCE, 2]) < options["min_mouth_confidence"]:
            rows.append({"stem": stem, "status": "bad_face"})
            continue
        candidates.append((words, checked, indices))
    if not candidates:
        return rows
    needed = {int(i) for _, _, indices in candidates for i in indices}
    last, frames = max(needed), {}
    with av.open(io.BytesIO(video_bytes)) as container:
        for number, frame in enumerate(container.decode(video=0)):
            if number in needed:
                frames[number] = frame.to_ndarray(format="gray")
            if number >= last:
                break
    reference = np.load(MEAN_FACE, allow_pickle=False)
    for count, (words, (transcript, phones), indices) in enumerate(candidates):
        if any(int(i) not in frames for i in indices):
            rows.append({"stem": stem, "status": "missing_frames"})
            continue
        try:
            patches = crop_mouths([frames[int(i)] for i in indices],
                                  keypoints[indices][:, FACE, :2].astype(np.float64), reference)
        except ValueError as error:
            rows.append({"stem": stem, "status": "crop_failed", "error": str(error)})
            continue
        relative = Path("crops") / stem / f"{count:04d}.npy"
        (output / relative).parent.mkdir(parents=True, exist_ok=True)
        np.save(output / relative, patches, allow_pickle=False)
        rows.append({"stem": stem, "status": "ok", "row": {
            "clip_id": f"seamless:{options['label']}/{stem}/{count:04d}", "dataset": "SeamlessInteraction",
            "split": options["label"], "source_type": "npy-mouth", "video": str(relative),
            "frames": len(patches), "fps": TARGET_FPS, "transcript": transcript, "phonemes": phones,
            "participant": stem.rsplit("_", 1)[1], "session": stem.split("_")[1],
            "start": round(float(words[0]["start"]), 3), "end": round(float(words[-1]["end"]), 3),
            "min_word_score": min(word["score"] for word in words), "tar": tar_path,
            "label_source": "seamless_interaction_automatic_transcript"}})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tars", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="naturalistic/train")
    parser.add_argument("--max-gap", type=float, default=0.5)
    parser.add_argument("--max-seconds", type=float, default=6.0)
    parser.add_argument("--min-words", type=int, default=3)
    parser.add_argument("--min-word-score", type=float, default=0.5)
    parser.add_argument("--min-mouth-confidence", type=float, default=0.8)
    parser.add_argument("--pad", type=float, default=0.1, help="seconds of video kept around the words")
    parser.add_argument("--min-frames", type=int, default=25)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(filename)s:%(lineno)d %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(args.output / "prepare.log"), logging.StreamHandler()],
                        force=True)
    options = {key: getattr(args, key) for key in ("label", "max_gap", "max_seconds", "min_words",
                                                    "min_word_score", "min_mouth_confidence", "pad", "min_frames")}
    jobs = []
    for tar_path in args.tars:
        with tarfile.open(tar_path) as archive:
            stems = sorted({name.rsplit(".", 1)[0] for name in archive.getnames() if name.endswith(".mp4")})
        jobs += [(str(tar_path), stem, str(args.output), options) for stem in stems]
    statuses: dict[str, int] = {}
    manifest = []
    with ProcessPoolExecutor(args.workers) as pool:
        for done, rows in enumerate(pool.map(_process, jobs), 1):
            for row in rows:
                statuses[row["status"]] = statuses.get(row["status"], 0) + 1
                if row["status"] == "ok":
                    manifest.append(row["row"])
            LOGGER.info("%d/%d files, %d utterances kept", done, len(jobs), len(manifest))
    manifest.sort(key=lambda row: row["clip_id"])
    with (args.output / "manifest.jsonl").open("w") as handle:
        for row in manifest:
            handle.write(json.dumps(row) + "\n")
    seconds = sum(row["frames"] for row in manifest) / TARGET_FPS
    summary = {"files": len(jobs), "utterances": len(manifest), "hours": seconds / 3600,
               "words": sum(len(row["transcript"].split()) for row in manifest), "statuses": statuses,
               "options": options, "tars": [str(path) for path in args.tars]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    LOGGER.info("summary %s", json.dumps(summary))


if __name__ == "__main__":
    main()
