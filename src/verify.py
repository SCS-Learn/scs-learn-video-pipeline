"""Check that a stage's output is plausible, not merely present.

This exists because of a real failure that went unnoticed: a
screen_with_cards.mp4 of 1,048,624 bytes sat next to a 169 MB source for a week.
The stage had exited, the file existed, and every downstream stage would have
happily consumed it and produced a broken final video. Exit status told nobody
anything.

Silent success with garbage output is the failure mode that most defeats
non-technical operation, so each stage's output is checked against the thing it
was derived from: duration within tolerance, frames present, size not absurd,
audio present where audio is expected.

    from src.verify import verify_stage
    verify_stage("cards", paths)          # raises VerificationError on failure

Standalone:
    python -m src.verify --lecture-dir data/15210-lecture12
    python -m src.verify --lecture-dir data/... --stage cards
"""

import argparse
import json
import os
import subprocess

from src.paths import LecturePaths


class VerificationError(RuntimeError):
    pass


def probe(path):
    """ffprobe a media file. Returns None if it cannot be read at all."""
    if not os.path.exists(path):
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=codec_type,codec_name,width,height,r_frame_rate,nb_frames",
             "-show_entries", "format=duration,size", "-of", "json", path],
            capture_output=True, text=True, check=True, timeout=120).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    try:
        info = json.loads(out)
    except json.JSONDecodeError:
        return None

    streams = info.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = info.get("format", {})
    d = {
        "path": path,
        "size": int(fmt.get("size") or os.path.getsize(path)),
        "duration": float(fmt.get("duration") or 0.0),
        "has_video": video is not None,
        "has_audio": audio is not None,
    }
    if video:
        num, _, den = (video.get("r_frame_rate") or "0/1").partition("/")
        try:
            d["fps"] = float(num) / float(den or 1)
        except (ValueError, ZeroDivisionError):
            d["fps"] = 0.0
        d["width"] = int(video.get("width") or 0)
        d["height"] = int(video.get("height") or 0)
        d["n_frames"] = int(video.get("nb_frames") or 0)
    return d


def check_media(path, ref_path=None, tol_s=2.0, min_bytes=100_000,
                expect_audio=None, label=""):
    """Return a list of human-readable problems. Empty list means healthy."""
    tag = label or os.path.basename(path)
    if not os.path.exists(path):
        return [f"{tag}: missing ({path})"]

    info = probe(path)
    if info is None:
        return [f"{tag}: exists but ffprobe cannot read it - truncated or not "
                f"a media file ({os.path.getsize(path)} bytes)"]

    problems = []
    if info["size"] < min_bytes:
        problems.append(f"{tag}: only {info['size']} bytes - almost certainly a "
                        f"failed or aborted write")
    if not info["has_video"]:
        problems.append(f"{tag}: no video stream")
    if info.get("n_frames", 0) == 0 and info["duration"] <= 0:
        problems.append(f"{tag}: zero frames and zero duration")
    if expect_audio is True and not info["has_audio"]:
        problems.append(f"{tag}: no audio stream, but this stage should carry "
                        f"audio through")

    if ref_path:
        ref = probe(ref_path)
        if ref is None:
            problems.append(f"{tag}: cannot probe reference {ref_path}")
        elif ref["duration"] > 0:
            drift = info["duration"] - ref["duration"]
            if abs(drift) > tol_s:
                pct = 100.0 * info["duration"] / ref["duration"]
                problems.append(
                    f"{tag}: duration {info['duration']:.1f}s vs "
                    f"{os.path.basename(ref_path)} {ref['duration']:.1f}s "
                    f"({pct:.1f}% of source, off by {drift:+.1f}s). This is what "
                    f"a truncated encode looks like.")
    return problems


def _stage_checks(stage, p):
    """(label, path, ref_path, kwargs) tuples for one stage's outputs."""
    if stage == "sync":
        # screen_sync is a trimmed screen, so it is legitimately shorter than
        # screen.mp4; compare against the camera it was aligned to instead.
        return [("screen_sync.mp4", p.screen_sync, p.camera,
                 {"tol_s": 5.0})]
    if stage == "transcription":
        return [("transcript_classified.json", p.transcript_classified, None, {})]
    if stage == "audio":
        return [("camera_muted.mp4", p.camera_muted, p.camera,
                 {"expect_audio": True})]
    if stage == "face_anon":
        src = p.camera_muted if os.path.exists(p.camera_muted) else p.camera
        return [("camera_muted_anon.mp4", p.camera_anon, src,
                 {"expect_audio": True})]
    if stage == "cards":
        return [("screen_with_cards.mp4", p.screen_with_cards, p.screen_sync, {})]
    if stage == "captions":
        return [("captions.srt", p.captions, None, {})]
    if stage == "assembly":
        ref = p.camera_anon if os.path.exists(p.camera_anon) else p.camera_muted
        return [("final video", p.final, ref, {"expect_audio": True})]
    return []


def _check_json(path, label):
    if not os.path.exists(path):
        return [f"{label}: missing ({path})"]
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return [f"{label}: unreadable JSON ({e})"]
    if not isinstance(data, list) or not data:
        return [f"{label}: expected a non-empty list of segments"]
    missing = [k for k in ("start", "end", "text") if k not in data[0]]
    if missing:
        return [f"{label}: segments lack {missing}"]
    if not any("is_student_question" in s for s in data):
        return [f"{label}: no segment carries is_student_question - the "
                f"classification step did not run"]
    return []


def _check_srt(path, label):
    if not os.path.exists(path):
        return [f"{label}: missing ({path})"]
    if os.path.getsize(path) < 32:
        return [f"{label}: {os.path.getsize(path)} bytes - effectively empty"]
    return []


def verify_stage(stage, paths, strict=True):
    """Check one stage's outputs. Raises VerificationError unless strict=False.

    Returns the list of problems (empty when healthy)."""
    p = paths if isinstance(paths, LecturePaths) else LecturePaths(paths)
    problems = []
    for label, path, ref, kwargs in _stage_checks(stage, p):
        if path.endswith(".json"):
            problems += _check_json(path, label)
        elif path.endswith(".srt"):
            problems += _check_srt(path, label)
        else:
            problems += check_media(path, ref_path=ref, label=label, **kwargs)

    if problems:
        msg = (f"stage '{stage}' produced output that does not look right:\n"
               + "\n".join(f"  - {x}" for x in problems))
        if strict:
            raise VerificationError(msg)
        print(f"[verify] {msg}")
    else:
        names = ", ".join(c[0] for c in _stage_checks(stage, p)) or "(nothing)"
        print(f"[verify] {stage}: OK ({names})")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lecture-dir", required=True)
    parser.add_argument("--stage", default=None,
                        help="Check one stage; default checks every stage whose "
                             "output exists")
    args = parser.parse_args()
    p = LecturePaths(args.lecture_dir)

    stages = ([args.stage] if args.stage else
              ["sync", "transcription", "audio", "face_anon", "cards",
               "captions", "assembly"])
    total = 0
    for stage in stages:
        checks = _stage_checks(stage, p)
        # When sweeping, skip stages that have not run yet rather than
        # reporting every not-yet-produced file as a failure.
        if not args.stage and checks and not any(
                os.path.exists(c[1]) for c in checks):
            print(f"[verify] {stage}: not run yet, skipping")
            continue
        total += len(verify_stage(stage, p, strict=False))
    if total:
        raise SystemExit(f"\n{total} problem(s) found")
    print("\nall checked stages look healthy")


if __name__ == "__main__":
    main()
