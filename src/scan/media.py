"""ffmpeg/ffprobe plumbing shared by the scanner's measurement passes.

Scanning a semester means touching every lecture, so the cost of a pass matters
in a way it does not for the pipeline stages, which touch one lecture at a time
and are allowed to take an hour. Two decisions follow from that:

**Frames come off the keyframe grid, not a full decode.** Panopto encodes with a
fixed 2.4s GOP, so `-skip_frame nokey` yields a uniform sample grid roughly
every 2.4 seconds for a twentieth of the CPU -- measured on 15-210 lecture 12,
7.7s wall against 26s (and 9s of CPU against 172s) for the same 79-minute
screen capture. Every visual metric here is a rate or a fraction over minutes,
so 2.4s resolution is far finer than anything being asked. `keyframe_grid_ok`
checks that assumption per file and the caller falls back to a timed `fps=`
decode when a source turns out to have irregular or sparse keyframes.

**The audio is decoded once.** `decode_audio` runs ebur128 as a filter while
piping 16 kHz mono PCM to stdout, so the standardised loudness figures and the
raw samples for the prosody maths come out of a single pass rather than two.

Nothing here decides whether a number is good -- that is src/scan/rubric.py.
"""

import json
import os
import re
import subprocess

import numpy as np

# Longest a single ffmpeg pass may run before the scanner gives up on a
# lecture. A whole-semester scan must not be able to hang on one bad file.
FFMPEG_TIMEOUT = 1800

# Above this mean gap, or below this many samples, the keyframe grid is too
# coarse to measure a slide change against and callers should decode instead.
MAX_KEYFRAME_GAP_S = 8.0
MIN_KEYFRAMES = 40


class MediaError(RuntimeError):
    pass


def _run(cmd, timeout=FFMPEG_TIMEOUT):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def probe(path):
    """Container and stream facts. None when the file cannot be read at all.

    Deliberately close to src.verify.probe -- that one is the pipeline's
    output check and answers "did the stage write something plausible"; this
    one is the scanner's input survey and additionally needs aspect ratio,
    codec and bitrate to judge a *source*. Kept separate rather than widened
    so that loosening something here cannot weaken the publish-time check.
    """
    if not os.path.exists(path):
        return None
    try:
        r = _run(["ffprobe", "-v", "error",
                  "-show_entries",
                  "stream=codec_type,codec_name,width,height,r_frame_rate,"
                  "nb_frames,sample_rate,channels,bit_rate",
                  "-show_entries", "format=duration,size,bit_rate",
                  "-of", "json", path], timeout=180)
        info = json.loads(r.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        return None
    if not info.get("streams"):
        return None

    streams = info["streams"]
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = info.get("format", {})
    try:
        size = int(fmt.get("size") or os.path.getsize(path))
    except OSError:
        size = 0
    d = {
        "path": path,
        "size": size,
        "duration": float(fmt.get("duration") or 0.0),
        "has_video": v is not None,
        "has_audio": a is not None,
        "bit_rate": int(fmt.get("bit_rate") or 0),
    }
    if v:
        num, _, den = (v.get("r_frame_rate") or "0/1").partition("/")
        try:
            d["fps"] = float(num) / float(den or 1)
        except (ValueError, ZeroDivisionError):
            d["fps"] = 0.0
        d["width"] = int(v.get("width") or 0)
        d["height"] = int(v.get("height") or 0)
        d["vcodec"] = v.get("codec_name")
        d["aspect"] = (d["width"] / d["height"]) if d["height"] else 0.0
    if a:
        d["sample_rate"] = int(a.get("sample_rate") or 0)
        d["channels"] = int(a.get("channels") or 0)
        d["acodec"] = a.get("codec_name")
    return d


# --------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------

_SHOWINFO_PTS = re.compile(r"pts_time:([0-9.]+)")


def keyframe_times(path):
    """Presentation times of the video keyframes, in seconds."""
    try:
        r = _run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                  "-skip_frame", "nokey", "-show_entries", "frame=pts_time",
                  "-of", "csv=p=0", path], timeout=600)
    except subprocess.SubprocessError:
        return []
    out = []
    for line in r.stdout.splitlines():
        line = line.strip().rstrip(",")
        try:
            out.append(float(line))
        except ValueError:
            continue
    return out


def keyframe_grid_ok(times, duration):
    """Is this keyframe grid dense and regular enough to sample against?"""
    if len(times) < MIN_KEYFRAMES:
        return False
    gaps = np.diff(np.asarray(times, dtype=float))
    if gaps.size == 0:
        return False
    # A long tail matters more than the mean: one 5-minute gap hides a slide.
    return float(np.mean(gaps)) <= MAX_KEYFRAME_GAP_S and \
        float(np.percentile(gaps, 98)) <= MAX_KEYFRAME_GAP_S * 3


def iter_frames(path, width, height, pix_fmt="gray", keyframes_only=True,
                fps=None, stride=1):
    """Yield (time_s, frame) pairs as numpy arrays, streaming.

    Streams rather than returning a list: a 90-minute lecture is ~2,300
    keyframes and holding them all at 480x270 RGB would be 900 MB for no
    reason, when every consumer here needs at most the previous frame.

    keyframes_only uses the GOP grid (cheap, ~2.4s spacing on Panopto output).
    Pass fps= instead to force a timed decode when the grid is unusable.
    `stride` yields every Nth sampled frame, for consumers like face detection
    that want far fewer frames than the grid provides.
    """
    depth = {"gray": 1, "rgb24": 3}[pix_fmt]
    frame_bytes = width * height * depth

    cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-nostdin"]
    if keyframes_only and fps is None:
        cmd += ["-skip_frame", "nokey"]
    cmd += ["-i", path, "-an"]
    vf = []
    if fps is not None:
        vf.append(f"fps={fps}")
    vf += [f"scale={width}:{height}:flags=fast_bilinear", f"format={pix_fmt}"]
    if fps is None:
        cmd += ["-fps_mode", "passthrough"]
    cmd += ["-vf", ",".join(vf), "-f", "rawvideo", "-"]

    # stderr goes to the void on purpose. The obvious way to get timestamps
    # out of this pass is a `showinfo` filter, but that writes a line per
    # frame -- ~460 kB over a 90-minute lecture -- and a consumer that reads
    # stdout to exhaustion before touching stderr deadlocks the moment that
    # pipe fills. Timestamps come from frame_times() instead, which is one
    # cheap ffprobe rather than a second decode.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, bufsize=frame_bytes * 4)
    try:
        idx = 0
        while True:
            buf = proc.stdout.read(frame_bytes)
            if not buf or len(buf) < frame_bytes:
                break
            if idx % stride == 0:
                arr = np.frombuffer(buf, dtype=np.uint8)
                arr = (arr.reshape(height, width) if depth == 1
                       else arr.reshape(height, width, depth))
                yield idx, arr
            idx += 1
    finally:
        # A consumer that stops early (islice, an exception) leaves ffmpeg
        # writing into a pipe nobody reads, so it has to be killed rather
        # than waited on.
        try:
            proc.stdout.close()
        except OSError:
            pass
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=60)


def frame_times(path, count, duration, keyframes_only=True, fps=None):
    """Timestamps for the frames iter_frames just produced.

    Uses the real keyframe PTS list when sampling the GOP grid, and falls back
    to a uniform grid when that is unavailable or the wrong length -- an even
    spacing over the known duration is right for an fps= decode and close
    enough for a fixed-GOP source.
    """
    if fps is None and keyframes_only:
        t = keyframe_times(path)
        if len(t) >= count:
            return np.asarray(t[:count], dtype=float)
    if count <= 0:
        return np.zeros(0)
    step = (duration / count) if duration > 0 else (1.0 / (fps or 1.0))
    return np.arange(count, dtype=float) * step


# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------

_EBUR128_FIELDS = {
    "loudness_lufs": re.compile(r"^\s*I:\s*(-?[0-9.]+)\s*LUFS", re.M),
    "loudness_range_lu": re.compile(r"^\s*LRA:\s*(-?[0-9.]+)\s*LU", re.M),
    "true_peak_dbtp": re.compile(r"Peak:\s*(-?[0-9.]+)\s*dBFS"),
    "lra_low": re.compile(r"^\s*LRA low:\s*(-?[0-9.]+)", re.M),
    "lra_high": re.compile(r"^\s*LRA high:\s*(-?[0-9.]+)", re.M),
}


def decode_audio(path, sample_rate=16000):
    """(samples, loudness) from ONE decode pass.

    samples   float32 mono in [-1, 1] at `sample_rate`
    loudness  the ebur128 summary: integrated LUFS, LRA, true peak

    Running ebur128 as a filter while the resampled PCM goes to stdout means
    the standardised loudness numbers and the raw signal for the prosody
    maths cost one decode between them instead of two. On a 79-minute lecture
    that pass is ~21s.
    """
    cmd = ["ffmpeg", "-hide_banner", "-v", "info", "-nostdin", "-i", path,
           "-vn", "-af", "ebur128=peak=true", "-ar", str(sample_rate),
           "-ac", "1", "-f", "s16le", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=FFMPEG_TIMEOUT)
    except subprocess.SubprocessError as e:
        raise MediaError(f"audio decode failed for {path}: {e}") from e
    if not proc.stdout:
        raise MediaError(f"audio decode produced nothing for {path}")

    pcm = np.frombuffer(proc.stdout, dtype="<i2").astype(np.float32) / 32768.0
    err = proc.stderr.decode("utf-8", "replace")
    loudness = {}
    for key, rx in _EBUR128_FIELDS.items():
        m = rx.findall(err)
        if m:
            # ebur128 prints running values then a Summary block; the summary
            # is last, and it is the only one measured over the whole file.
            try:
                loudness[key] = float(m[-1])
            except ValueError:
                pass
    return pcm, loudness


def ffmpeg_log(cmd_tail, path, timeout=FFMPEG_TIMEOUT):
    """Run a detect-style filter and return ffmpeg's stderr.

    At -v info deliberately: blackdetect, freezedetect and silencedetect all
    log their findings at info level, so at -v error a wholly black file
    reports no black at all. CLAUDE.md records that trap for blackdetect and
    src/video/scenes.py records it again for freezedetect; it is the same
    mistake with a different filter, and it is silent both times.
    """
    cmd = (["ffmpeg", "-hide_banner", "-v", "info", "-nostdin", "-i", path]
           + cmd_tail + ["-f", "null", "-"])
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.SubprocessError as e:
        raise MediaError(f"ffmpeg failed on {path}: {e}") from e
    return proc.stderr
