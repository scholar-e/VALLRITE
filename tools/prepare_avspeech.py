"""Materialize AVSpeech segments referenced by the official CSV manifests."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Segment:
    row: int
    youtube_id: str
    start: float
    end: float
    face_x: float
    face_y: float

    @property
    def stem(self) -> str:
        return f"{self.row:07d}_{self.youtube_id}_{self.start:.3f}_{self.end:.3f}"


def rows(path: Path, start_row: int):
    with path.open(newline="", encoding="utf-8") as source:
        for row_number, values in enumerate(csv.reader(source)):
            if row_number < start_row or len(values) != 5:
                continue
            try:
                yield Segment(
                    row=row_number,
                    youtube_id=values[0],
                    start=float(values[1]),
                    end=float(values[2]),
                    face_x=float(values[3]),
                    face_y=float(values[4]),
                )
            except ValueError:
                continue


def download(segment: Segment, output_dir: Path) -> tuple[Path | None, str]:
    destination = output_dir / f"{segment.stem}.mp4"
    if destination.exists():
        return destination, "already present"

    with tempfile.TemporaryDirectory(prefix="avspeech-") as temporary:
        source_video = Path(temporary) / "video"
        source_audio = Path(temporary) / "audio"
        base_command = [
            str(Path(sys.executable).with_name("yt-dlp")),
            "--no-playlist",
            "--quiet",
            "--no-warnings",
            "--max-filesize",
            "500M",
        ]
        url = f"https://www.youtube.com/watch?v={segment.youtube_id}"
        video_command = [
            *base_command,
            "--format",
            "bestvideo[ext=mp4]/bestvideo",
            "--output",
            str(source_video),
            url,
        ]
        audio_command = [
            *base_command,
            "--format",
            "bestaudio[ext=m4a]/bestaudio",
            "--output",
            str(source_audio),
            url,
        ]
        result = subprocess.run(
            video_command, capture_output=True, text=True, check=False
        )
        if result.returncode != 0 or not source_video.exists():
            message = (result.stderr or result.stdout).strip().splitlines()
            return None, message[
                -1
            ] if message else f"yt-dlp exited {result.returncode}"
        result = subprocess.run(
            audio_command, capture_output=True, text=True, check=False
        )
        if result.returncode != 0 or not source_audio.exists():
            message = (result.stderr or result.stdout).strip().splitlines()
            return None, message[
                -1
            ] if message else f"yt-dlp exited {result.returncode}"

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return None, "ffmpeg is required to trim downloaded sources"
        trim = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                str(segment.start),
                "-i",
                str(source_video),
                "-ss",
                str(segment.start),
                "-i",
                str(source_audio),
                "-t",
                str(segment.end - segment.start),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c",
                "copy",
                "-y",
                str(destination),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if trim.returncode == 0 and destination.exists():
            return destination, "downloaded"
        destination.unlink(missing_ok=True)
        message = (trim.stderr or trim.stdout).strip().splitlines()
        return None, message[-1] if message else f"ffmpeg exited {trim.returncode}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("datasets/avspeech/clips")
    )
    parser.add_argument(
        "--limit", type=int, default=1, help="number of successful clips"
    )
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=50)
    args = parser.parse_args()
    if args.limit < 1 or args.max_attempts < 1:
        parser.error("--limit and --max-attempts must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / "index.jsonl"
    successes = 0
    attempts = 0
    with index_path.open("a", encoding="utf-8") as index:
        for segment in rows(args.manifest, args.start_row):
            if successes >= args.limit or attempts >= args.max_attempts:
                break
            attempts += 1
            path, status = download(segment, args.output_dir)
            record = asdict(segment) | {
                "path": str(path) if path else None,
                "status": status,
            }
            index.write(json.dumps(record) + "\n")
            index.flush()
            if path:
                successes += 1
                print(path)
            else:
                print(f"Skipped row {segment.row}: {status}")

    if successes < args.limit:
        raise SystemExit(
            f"Found {successes}/{args.limit} clips after {attempts} attempts"
        )


if __name__ == "__main__":
    main()
