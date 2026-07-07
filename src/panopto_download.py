"""Download SCS Panopto lecture assets from a harvested manifest.

Phase B of the panopto-lecture-ingestion skill. The manifest is produced in the browser
during Phase A (see SKILL.md). No authentication is needed here: the CloudFront media URLs
in the manifest are open once obtained from DeliveryInfo.

Usage:
    python3 panopto_download.py <manifest.json> <output-dir>

Each lecture is written to <output-dir>/<key>/ with camera.mp4, screen.mp4, metadata.json,
and chapters.json. Per-stream format is auto-detected: direct .mp4 via curl, HLS
(.hls/master.m3u8) via ffmpeg. ffmpeg/ffprobe must be on PATH for HLS streams.
"""

import json
import os
import subprocess
import sys
import time
from typing import Any


METADATA_FIELDS = ["key", "id", "name", "durationSec", "course", "owner", "start"]


def run_command(command: list[str]) -> int:
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode


def probe_resolution(path: str) -> str:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0", path,
        ],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def download_stream(url: str, is_hls: bool, destination: str) -> int:
    if is_hls:
        return run_command([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", url, "-c", "copy", "-bsf:a", "aac_adtstoasc", destination,
        ])
    return run_command(["curl", "-s", "--fail", "--max-time", "1200", "-o", destination, url])


def file_size_mb(path: str) -> int:
    if not os.path.exists(path):
        return 0
    return os.path.getsize(path) // 1024 // 1024


def write_lecture_sidecars(lecture: dict[str, Any], lecture_dir: str) -> None:
    metadata = {field: lecture.get(field) for field in METADATA_FIELDS}
    json.dump(metadata, open(os.path.join(lecture_dir, "metadata.json"), "w"), indent=2)
    json.dump(lecture.get("chapters", []), open(os.path.join(lecture_dir, "chapters.json"), "w"), indent=2)


def ingest_lecture(lecture: dict[str, Any], output_dir: str) -> None:
    lecture_key = lecture["key"]
    lecture_dir = os.path.join(output_dir, lecture_key)
    os.makedirs(lecture_dir, exist_ok=True)
    write_lecture_sidecars(lecture, lecture_dir)

    for stream in lecture.get("streams", []):
        stream_type = stream["type"]
        is_hls = stream["isHls"]
        destination = os.path.join(lecture_dir, stream_type + ".mp4")
        started_at = time.time()
        return_code = download_stream(stream["url"], is_hls, destination)
        elapsed = round(time.time() - started_at)
        size_mb = file_size_mb(destination)
        resolution = probe_resolution(destination) if size_mb else "n/a"
        status = "ok" if return_code == 0 and size_mb else "FAILED"
        print(
            f"  {lecture_key} {stream_type:6} hls={str(is_hls):5} {status} "
            f"{size_mb}MB {resolution} {elapsed}s",
            flush=True,
        )


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    manifest_path = sys.argv[1]
    output_dir = sys.argv[2]
    manifest = json.load(open(manifest_path))
    lectures = manifest["lectures"]
    os.makedirs(output_dir, exist_ok=True)

    print(f"Ingesting {len(lectures)} lecture(s) into {output_dir}")
    for lecture in lectures:
        print(f"{lecture['key']}: {lecture.get('name')}")
        ingest_lecture(lecture, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
