"""Put the screen and camera on a common clock, and drop the dead air in front.

Two separate problems, fixed in one pass because the second depends on the
first:

1. **Alignment.** Panopto starts the screen capture and the camera at different
   moments but stops them together, so the longer file simply began earlier.
   The duration difference *is* the offset; trimming that much off the front of
   whichever stream is longer lines them up.

2. **The black lead.** A lecture capture typically starts before the projector
   is showing anything, so the head of the screen recording is solid black. On
   lecture 12 that is 629.0 s -- over ten minutes. Nothing detected it: the
   black disappeared only because the alignment trim (718.7 s) happened to be
   longer, and the pipeline had no idea the frames it discarded were blank. The
   margin was 89.7 s. Reverse those two numbers -- a screen that starts 3
   minutes early but is dark for 10 -- and seven minutes of black lands at the
   head of the published video. verify.py would not catch it either: a
   black-but-well-encoded segment passes every duration, size and
   bytes-per-second check comfortably.

The trim has to come off BOTH streams. After step 1 they share a clock, so
cutting the screen alone would slide it out of sync with the camera by exactly
the amount removed. sync runs first in the pipeline, which is what makes this
safe: transcription, cards and captions all run afterwards and are therefore
timed against the trimmed camera, with no timestamps to rewrite.

The camera is only rewritten when a black trim is actually needed. When the
alignment trim already covers the black -- the common case, and what happened
on lecture 12 -- camera_sync.mp4 is not written and every stage keeps reading
camera.mp4 exactly as before.

    python -m src.sync --lecture-dir data/15210-lecture12
    python -m src.sync --lecture-dir data/... --keep-black
"""

import os
import re
import subprocess

from src.paths import LecturePaths, lecture_parser


def get_duration(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


_BLACK_RE = re.compile(r"black_start:([\d.]+)\s+black_end:([\d.]+)")


def leading_black(path, scan_seconds=900.0, pic_th=0.98, min_run=0.5,
                  gap_tolerance=3.0):
    """Seconds of black at the head of `path`; 0.0 if it does not start black.

    Only the head is scanned -- blackdetect over a full 80-minute lecture costs
    minutes and we only care about the front.

    Runs separated by a short gap are merged. This is not hypothetical tidying:
    lecture 12's screen reports black 0-131.2 s, a 1.12 s flash, then black
    132.32-628.96 s. Treating only the first run as the lead would under-trim by
    eight minutes.

    scan_seconds must comfortably exceed the expected lead -- scanning only the
    first 400 s of lecture 12 reports the black ending at 400 s, because that is
    where the scan stopped, not where the black did.

    blackdetect logs at info level, so ffmpeg must NOT be run at -v error here
    -- that silently returns "no black" for a completely black file.
    """
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats",
         "-t", str(scan_seconds), "-i", path,
         "-vf", f"blackdetect=d={min_run}:pic_th={pic_th}",
         "-an", "-f", "null", "-"],
        capture_output=True, text=True)
    runs = sorted((float(s), float(e))
                  for s, e in _BLACK_RE.findall(proc.stderr))
    if not runs or runs[0][0] > gap_tolerance:
        return 0.0

    end = 0.0
    for start, stop in runs:
        if start > end + gap_tolerance:
            break
        end = max(end, stop)

    if end >= scan_seconds - 1.0:
        print(f"[sync] WARNING black still running at {end:.0f}s, where the "
              f"scan window ends -- the real lead is probably longer. Re-run "
              f"with a larger --black-scan-seconds before trusting this.")
    return end


def _trim(src_path, out_path, seconds):
    """Copy `src_path` to `out_path` minus its first `seconds`.

    -ss before -i so ffmpeg seeks rather than decoding to the cut point, and
    -c copy because re-encoding an 80-minute lecture to remove a few minutes of
    black would cost more than every other CPU stage combined. The cut lands on
    the nearest keyframe, so the caller checks the achieved durations.
    """
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-ss", f"{seconds:.3f}", "-i", src_path,
         "-c", "copy", "-avoid_negative_ts", "make_zero", out_path],
        check=True)
    return out_path


def align_screen_to_camera(camera_path, screen_path, out_screen_path,
                           out_camera_path=None, min_offset=0.5,
                           trim_black=True, scan_seconds=900.0,
                           max_black_trim=600.0, drift_tolerance=1.5):
    """Trim both streams to a shared start. Returns (screen_out, camera_out).

    camera_out is the original camera path when no black trim was needed.
    """
    camera_duration = get_duration(camera_path)
    screen_duration = get_duration(screen_path)
    offset = screen_duration - camera_duration

    # Whichever stream is longer began earlier. The negative case used to fall
    # through the `offset <= min_offset` return below, leaving screen_sync.mp4
    # unwritten while the stage still reported success.
    screen_trim = max(0.0, offset)
    camera_trim = max(0.0, -offset)
    print(f"[sync] camera {camera_duration:.1f}s, screen {screen_duration:.1f}s "
          f"-> align by {abs(offset):.1f}s off the "
          f"{'screen' if offset > 0 else 'camera'}")

    residual = 0.0
    if trim_black:
        black = leading_black(screen_path, scan_seconds=scan_seconds)
        residual = max(0.0, black - screen_trim)
        if black:
            print(f"[sync] screen is black for its first {black:.1f}s; "
                  f"alignment removes {min(black, screen_trim):.1f}s of that")
        if residual > 0:
            if residual > max_black_trim:
                raise SystemExit(
                    f"[sync] refusing to cut {residual:.0f}s of leading black "
                    f"(limit {max_black_trim:.0f}s).\n"
                    f"That is {100 * residual / camera_duration:.0f}% of the "
                    f"lecture, which usually means the projector was off far "
                    f"longer than expected or blackdetect misfired -- not "
                    f"something to discard silently.\n"
                    f"Raise --max-black-trim if it is genuinely correct, or "
                    f"pass --keep-black to leave it in.")
            print(f"[sync] trimming a further {residual:.1f}s of black from "
                  f"BOTH streams (pre-lecture dead air)")
        elif black:
            print(f"[sync] no black survives alignment; camera untouched")

    screen_trim += residual
    camera_trim += residual

    # screen_sync.mp4 is written even when the trim is zero. cards.py and
    # verify.py both address it by name, so returning the untrimmed screen_path
    # meant this stage could exit 0 having produced no such file, and cards then
    # failed several stages later on a missing input rather than here.
    if screen_trim > min_offset:
        screen_out = _trim(screen_path, out_screen_path, screen_trim)
    else:
        print(f"[sync] screen needs no trim; remuxing to "
              f"{os.path.basename(out_screen_path)} so downstream stages have "
              f"the file they expect")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", screen_path,
                        "-c", "copy", out_screen_path], check=True)
        screen_out = out_screen_path
    # Only rewrite the camera when the black trim demands it, so lectures that
    # need no black cut keep reading camera.mp4 and nothing downstream changes.
    camera_out = camera_path
    if camera_trim > min_offset:
        if not out_camera_path:
            raise ValueError("camera needs trimming but out_camera_path is unset")
        camera_out = _trim(camera_path, out_camera_path, camera_trim)

    # -c copy cuts at keyframes, so the achieved trims can differ from the
    # requested ones by up to a GOP. Independent rounding on two files is how a
    # fix for black turns into a lip-sync bug, so check rather than assume.
    got_screen, got_camera = get_duration(screen_out), get_duration(camera_out)
    drift = abs(got_screen - got_camera)
    print(f"[sync] screen {got_screen:.1f}s, camera {got_camera:.1f}s "
          f"(drift {drift:.2f}s)")
    if drift > drift_tolerance:
        raise SystemExit(
            f"[sync] screen and camera are {drift:.2f}s apart after trimming "
            f"(tolerance {drift_tolerance}s).\n"
            f"-c copy cut each at its nearest keyframe and they rounded "
            f"differently. Publishing this would put the slides out of step "
            f"with the instructor for the whole lecture.")
    return screen_out, camera_out


def main():
    parser = lecture_parser("Trim the screen recording to align with the camera, "
                            "and drop the black lead in front of the lecture.")
    parser.add_argument("--keep-black", action="store_true",
                        help="Align only; leave any leading black in place")
    parser.add_argument("--black-scan-seconds", type=float, default=900.0,
                        help="How far into the screen to look for black "
                             "(default: 900)")
    parser.add_argument("--max-black-trim", type=float, default=600.0,
                        help="Refuse to cut more leading black than this "
                             "(default: 600)")
    args = parser.parse_args()
    p = LecturePaths(args.lecture_dir)

    screen_out, camera_out = align_screen_to_camera(
        camera_path=p.camera,
        screen_path=p.screen,
        out_screen_path=p.screen_sync,
        out_camera_path=p.camera_sync,
        trim_black=not args.keep_black,
        scan_seconds=args.black_scan_seconds,
        max_black_trim=args.max_black_trim,
    )
    print(f"[sync] screen aligned -> {screen_out}")
    if camera_out != p.camera:
        print(f"[sync] camera trimmed -> {camera_out}  "
              f"(every later stage reads this via paths.resolve_camera())")
    elif os.path.exists(p.camera_sync):
        # A stale trim from an earlier run would silently outrank camera.mp4.
        print(f"[sync] WARNING {p.camera_sync} exists but this run needed no "
              f"camera trim. It is left over from a previous run and will be "
              f"picked up by resolve_camera(); delete it.")


if __name__ == "__main__":
    main()
