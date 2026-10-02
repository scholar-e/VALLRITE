"""Shared data and prompt contract for noisy visual-phone teacher experiments."""
from __future__ import annotations

import json
from pathlib import Path

PROMPT_VERSION = "visual-phone-nbest-to-words-1"


def nbest_prompt(record: dict) -> str:
    hypotheses = record["phone_hypotheses"]
    best_score = float(hypotheses[0]["ctc_log_score"])
    lines = []
    for hypothesis in hypotheses:
        relative = float(hypothesis["ctc_log_score"]) - best_score
        lines.append(
            f"{int(hypothesis['rank'])}. {' '.join(hypothesis['phones'])} "
            f"[relative_score={relative:.3f}]"
        )
    return (
        "Recover the most likely English transcript from noisy visual-speech "
        "phoneme hypotheses. Similar-looking phonemes may be confused or missing.\n"
        "Return only the transcript.\nHypotheses:\n" + "\n".join(lines) + "\nTranscript:"
    )


def load_nbest(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    seen = set()
    for row in rows:
        clip_id = row["clip_id"]
        if clip_id in seen:
            raise ValueError(f"duplicate clip id: {clip_id}")
        seen.add(clip_id)
        if not row["phone_hypotheses"] or not row["reference_words"]:
            raise ValueError(f"empty hypotheses or reference: {clip_id}")
    if not rows:
        raise ValueError("N-best input is empty")
    return rows

