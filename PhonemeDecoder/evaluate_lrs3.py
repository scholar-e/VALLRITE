"""Generate LRS3 word candidates from a frozen visual checkpoint and score WER."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import sys
import time

import torch
from torch.utils.data import DataLoader

from PhonemeDecoder.decoder import Lexicon, PHONES, StreamingDecoder, probabilities_from_logits
from PhonemeDecoder.evaluate_grid import edit_distance
from VisualPhoneme.data import (collate_clips, collate_fusion_clips,
                                collate_landmark_clips, greedy_decode)
from VisualPhoneme.lrs3_data import Lrs3Clips
from VisualPhoneme.evaluate_nbest import load_model
from VisualPhoneme.visemes import (PHONE_TO_VISUAL_GROUP, VISUAL_GROUPS,
                                   VISUAL_GROUPS_WITH_REST)

LOGGER = logging.getLogger("phoneme_decoder.evaluate_lrs3")


def configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.handlers.clear()
    for handler in (logging.FileHandler(path, delay=False), logging.StreamHandler(sys.stdout)):
        handler.setLevel(logging.DEBUG if isinstance(handler, logging.FileHandler)
                         else logging.INFO)
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def build_cmudict_lexicon(inventory: tuple[str, ...] = PHONES) -> dict:
    """Create the decoder contract from the installed, versioned CMUdict data."""
    try:
        import pronouncing
    except ImportError as error:
        raise RuntimeError("pronouncing is required for the LRS3 lexicon") from error
    pronunciations: dict[str, set[tuple[str, ...]]] = {}
    for word, raw_phones in pronouncing.cmudict.entries():
        normalized_word = word.lower()
        if not re.fullmatch(r"[a-z]+(?:'[a-z]+)*", normalized_word):
            continue
        phones = tuple(re.sub(r"\d", "", phone).upper() for phone in raw_phones
                       if re.sub(r"\d", "", phone).upper() in PHONES)
        if phones and len(phones) == len(raw_phones):
            if inventory in {VISUAL_GROUPS, VISUAL_GROUPS_WITH_REST}:
                phones = tuple(PHONE_TO_VISUAL_GROUP[phone] for phone in phones)
            pronunciations.setdefault(normalized_word, set()).add(phones)
    words = []
    for word, variants_set in sorted(pronunciations.items()):
        variants = sorted(variants_set)
        prior = 1.0 / len(variants)
        words.append({
            "id": word,
            "text": word,
            "pronunciations": [
                {"id": f"{word}-{index}", "phones": list(phones), "prior": prior}
                for index, phones in enumerate(variants, 1)
            ],
        })
    document = {"phone_inventory": list(inventory), "words": words}
    Lexicon(document)
    return document


def reference_words(transcript: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)*", transcript.lower())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/lrs3"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--lexical-beam", type=int, default=32)
    parser.add_argument("--nbest", type=int, default=5)
    parser.add_argument("--soft-rest-boundaries", action="store_true",
                        help="allow REST emissions to be ignored instead of forcing a word split")
    parser.add_argument("--rest-boundary-bonus", type=float, default=0.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--frame-cache-dir", type=Path,
                        default=Path("datasets/lrs3/frame-cache"))
    args = parser.parse_args()
    if (not args.checkpoint.is_file() or args.batch_size < 1 or args.workers < 0
            or args.beam_width < args.nbest or args.lexical_beam < 1
            or args.nbest < 1 or (args.limit is not None and args.limit < 1)):
        parser.error("invalid checkpoint, limit, batch, worker, beam, or n-best setting")
    return args


def main() -> None:
    args = parse_args()
    configure_logging(args.output_dir / "generate.log")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device("cpu" if device_name == "auto" else device_name)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    architecture, model = load_model(checkpoint, device)
    inventory = tuple(checkpoint["phones"])
    if inventory not in {PHONES, VISUAL_GROUPS, VISUAL_GROUPS_WITH_REST}:
        raise ValueError("checkpoint has an unsupported decoder inventory")
    include_video = architecture in {"image", "large-image", "fusion", "gated-fusion",
                                     "tongue-gated-fusion", "large-gated-fusion",
                                     "large-transformer-fusion", "autoavsr-fusion"}
    include_landmarks = architecture in {"coordinates", "fusion", "gated-fusion",
                                         "tongue-gated-fusion", "large-gated-fusion",
                                         "large-transformer-fusion", "autoavsr-fusion"}
    dataset = Lrs3Clips(
        args.data_root, "validation", int(checkpoint["image_size"]), checkpoint["crop"],
        args.limit, False, include_landmarks, include_video, False,
        checkpoint.get("coordinate_mode", "eye-normalized"), args.frame_cache_dir,
        checkpoint.get("coordinate_features", "position"),
    )
    collators = {"image": collate_clips, "large-image": collate_clips,
                 "coordinates": collate_landmark_clips,
                 "fusion": collate_fusion_clips, "gated-fusion": collate_fusion_clips,
                 "tongue-gated-fusion": collate_fusion_clips,
                 "large-gated-fusion": collate_fusion_clips,
                 "large-transformer-fusion": collate_fusion_clips,
                 "autoavsr-fusion": collate_fusion_clips}
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=collators[architecture],
                        pin_memory=device.type == "cuda",
                        persistent_workers=args.workers > 0)
    lexicon_document = build_cmudict_lexicon(inventory)
    lexicon = Lexicon(lexicon_document)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "lexicon.json").write_text(
        json.dumps(lexicon_document, separators=(",", ":")) + "\n"
    )
    output_path = args.output_dir / "examples.jsonl"
    total_words = top_errors = oracle_errors = exact_hits = processed = 0
    statuses: Counter[str] = Counter()
    factor = int(checkpoint.get("ctc_upsample_factor", 1))
    started = time.perf_counter()
    LOGGER.info("device=%s architecture=%s clips=%d beam=%d nbest=%d ctc_upsample=%d",
                device, architecture, len(dataset), args.beam_width, args.nbest, factor)
    with output_path.open("w", buffering=1) as output, torch.inference_mode():
        for batch_number, batch in enumerate(loader, 1):
            if architecture in {"fusion", "gated-fusion", "tongue-gated-fusion",
                                "large-gated-fusion", "large-transformer-fusion",
                                "autoavsr-fusion"}:
                video, landmarks, mask, _, lengths, _, clip_ids = batch
                logits = model(video.to(device, non_blocking=True),
                               landmarks.to(device, non_blocking=True),
                               mask.to(device, non_blocking=True))
            elif architecture == "coordinates":
                landmarks, mask, _, lengths, _, clip_ids = batch
                logits = model(landmarks.to(device, non_blocking=True),
                               mask.to(device, non_blocking=True))
            else:
                video, _, lengths, _, clip_ids = batch
                logits = model(video.to(device, non_blocking=True))
            if factor > 1:
                logits = logits.repeat_interleave(factor, dim=0)
                lengths = lengths * factor
            greedy = greedy_decode(logits, lengths)
            for index, (clip_id, length) in enumerate(zip(clip_ids, lengths, strict=True)):
                decoder = StreamingDecoder(lexicon, beam_width=args.beam_width,
                                           lexical_beam=args.lexical_beam,
                                           soft_rest_boundaries=args.soft_rest_boundaries,
                                           rest_boundary_bonus=args.rest_boundary_bonus)
                decoder.accept(probabilities_from_logits(
                    logits, checkpoint["phones"], int(length), index
                ))
                result = decoder.result(args.nbest)
                row = dataset.rows[processed]
                reference = reference_words(row["transcript"])
                candidates = result["candidates"]
                errors = [edit_distance(reference, candidate["words"])
                          for candidate in candidates]
                top_error = errors[0] if errors else len(reference)
                oracle_error = min(errors, default=len(reference))
                record = {
                    "format": "lrs3-decoder-evaluation-example-0.1",
                    "clip_id": clip_id,
                    "split": "validation",
                    "reference_words": reference,
                    "reference_phones": row["phonemes"],
                    "greedy_visual_tokens": [inventory[token - 1] for token in greedy[index]],
                    "decoder_status": result["status"],
                    "candidates": candidates,
                    "top_1_word_errors": top_error,
                    "oracle_word_errors_at_n": oracle_error,
                    "reference_in_nbest": oracle_error == 0,
                }
                output.write(json.dumps(record, allow_nan=False) + "\n")
                total_words += len(reference)
                top_errors += top_error
                oracle_errors += oracle_error
                exact_hits += oracle_error == 0
                statuses[result["status"]] += 1
                processed += 1
            if batch_number == 1 or batch_number % 5 == 0:
                LOGGER.info("processed=%d top1_WER=%.2f%% oracle_WER@%d=%.2f%%",
                            processed, 100 * top_errors / max(total_words, 1), args.nbest,
                            100 * oracle_errors / max(total_words, 1))
    elapsed = time.perf_counter() - started
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "format": "lrs3-decoder-evaluation-summary-0.1",
        "scope": "development validation",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint["epoch"],
        "architecture": architecture,
        "clips": processed,
        "reference_words": total_words,
        "decoder_statuses": dict(statuses),
        "top_1_word_errors": top_errors,
        "top_1_wer": top_errors / total_words,
        "oracle_word_errors_at_n": oracle_errors,
        "oracle_wer_at_n": oracle_errors / total_words,
        "nbest": args.nbest,
        "reference_in_nbest": exact_hits,
        "reference_in_nbest_rate": exact_hits / processed,
        "elapsed_seconds": elapsed,
        "clips_per_second": processed / elapsed,
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    LOGGER.info("complete clips=%d top1_WER=%.2f%% oracle_WER@%d=%.2f%% seconds=%.1f",
                processed, 100 * summary["top_1_wer"], args.nbest,
                100 * summary["oracle_wer_at_n"], elapsed)
    for handler in LOGGER.handlers:
        handler.flush()


if __name__ == "__main__":
    main()
