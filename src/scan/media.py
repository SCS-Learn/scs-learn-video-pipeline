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

**A decode that stops early raises.** Both passes check that ffmpeg exited
cleanly, and the frame pass additionally cross-checks how far into the file it
actually got against the duration the container advertises -- because a
byte-truncated mp4 whose moov atom survives still probes as full length, still
exits 0, and turns every per-second metric downstream into a proportionally
inflated fiction. `_verify_decode` has the numbers and the argument for
raising rather than flagging.

Nothing here decides whether a number is good -- that is src/scan/rubric.py.
"""

import json
import os
import re
import subprocess
import tempfile

import numpy as np

# Longest a single ffmpeg pass may run before the scanner gives up on a
# lecture. A whole-semester scan must not be able to hang on one bad file.
FFMPEG_TIMEOUT = 1800

# Above this mean gap, or below this many samples, the keyframe grid is too
# coarse to measure a slide change against and callers should decode instead.
MAX_KEYFRAME_GAP_S = 8.0
MIN_KEYFRAMES = 40

# How much of a failed decode's stderr to quote back. The fatal message lands
# at the tail, and a damaged file emits one "Invalid NAL unit size" line per
# broken packet -- a semester scan does not need thousands of them inside an
# exception message.
FFMPEG_ERR_TAIL_BYTES = 4000
FFMPEG_ERR_TAIL_LINES = 6

# A frame decode is trusted only if the frames it produced span at least this
# fraction of the duration the container advertises. Deliberately loose:
# measured on this corpus, an intact 5,492.4s screen capture on a 2.4s GOP
# decodes to a last keyframe at 5,492.2s (99.996% covered) and even a sparse
# one-keyframe-a-minute source would clear 98%, while the half-truncated file
# that motivated the check covers 63%.
MIN_DECODE_COVERAGE = 0.90

# ffmpeg's -progress stream. `out_time_us` is the output stream's furthest
# timestamp, which is exactly "how far into the file did this decode get".
_PROGRESS_TIME = re.compile(rb"out_time_us=(\d+)")
_PROGRESS_KV = re.compile(r"^[A-Za-z_0-9]+=")


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


def _decode_report(errf):
    """(furthest output timestamp in seconds, quotable stderr tail).

    Reads only the tail of the capture, which is where both signals live: the
    fatal message, and the `progress=end` block carrying the last
    `out_time_us`. Everything before that is per-packet decoder noise on a
    damaged file and running commentary on a healthy one.
    """
    try:
        errf.seek(0, os.SEEK_END)
        size = errf.tell()
        errf.seek(max(0, size - FFMPEG_ERR_TAIL_BYTES))
        blob = errf.read()
    except (OSError, ValueError):
        return None, ""
    hits = _PROGRESS_TIME.findall(blob)
    out_time = (int(hits[-1]) / 1e6) if hits else None
    lines = [ln.strip() for ln in blob.decode("utf-8", "replace").splitlines()
             if ln.strip() and not _PROGRESS_KV.match(ln.strip())]
    return out_time, " | ".join(lines[-FFMPEG_ERR_TAIL_LINES:])


def _verify_decode(path, returncode, frames, out_time_s, duration, err_tail):
    """Fail loudly when a decode did not actually survey the whole file.

    Two independent checks, because on real damage only the second one fires.
    The case this was written against -- a 300s screen capture byte-truncated
    to half its length, with the moov atom moved to the front so ffprobe still
    reports 300.0s -- makes ffmpeg log `Invalid NAL unit size` per broken
    packet and then **exit 0**. So the exit status alone would have caught
    nothing.

    That matters because every consumer divides the container's duration by
    the frames it received: video_metrics does exactly that for its sample
    step, so a decode that stopped at 63% inflated each second-valued metric
    by 1.6x -- black_lead_s read 45.0 against a true 28.6, and
    longest_static_slide_s 123.8 against 78.6. Both feed the sync-risk warning
    that exists to stop a lecture publishing with minutes of black in front of
    it, and nothing downstream could tell the difference between that and a
    lecture that really is like that.

    Raising rather than returning a flag is deliberate. Every consumer here
    hands back a flat dict of numbers that goes into scan.json and then into a
    grade; there is nowhere in that dict to say "these came from a third of
    the lecture", and a flag would have to be threaded through three metric
    modules and honoured by all of them. scanner.py already catches per-tier
    exceptions into the lecture's `errors` list and leaves the tier out of
    `tiers_run`, so an exception costs the tier, shows up in the report, and
    lowers the reported coverage -- which is the honest answer. It is also the
    same fail-closed stance the rest of this repo takes: no measurement beats
    a confident wrong one.
    """
    if returncode not in (0, None):
        raise MediaError(
            f"frame decode of {path} exited {returncode} after {frames} "
            f"frames -- the sample is partial and every per-second metric "
            f"derived from it would be wrong: "
            f"{err_tail or 'no stderr captured'}")
    if not duration or duration <= 0:
        return                  # nothing to cross-check the coverage against
    if frames <= 0:
        raise MediaError(
            f"frame decode of {path} produced no frames at all, against a "
            f"container claiming {duration:.1f}s: "
            f"{err_tail or 'no stderr captured'}")
    if out_time_s is None:
        # No -progress output to compare against, so the exit status above is
        # the only thing there was to check.
        return
    covered = out_time_s / duration
    if covered < MIN_DECODE_COVERAGE:
        raise MediaError(
            f"frame decode of {path} covered only {covered:.0%} of the "
            f"{duration:.1f}s the container advertises -- {frames} frames, "
            f"the last at {out_time_s:.1f}s. The file is truncated or its "
            f"keyframes stop early; per-second metrics from this sample "
            f"would be inflated {1.0 / max(covered, 1e-6):.2f}x"
            + (f". ffmpeg said: {err_tail}" if err_tail else ""))


def iter_frames(path, width, height, pix_fmt="gray", keyframes_only=True,
                fps=None, stride=1, duration=None):
    """Yield (time_s, frame) pairs as numpy arrays, streaming.

    Streams rather than returning a list: a 90-minute lecture is ~2,300
    keyframes and holding them all at 480x270 RGB would be 900 MB for no
    reason, when every consumer here needs at most the previous frame.

    keyframes_only uses the GOP grid (cheap, ~2.4s spacing on Panopto output).
    Pass fps= instead to force a timed decode when the grid is unusable.
    `stride` yields every Nth sampled frame, for consumers like face detection
    that want far fewer frames than the grid provides. `duration` is the
    container duration to cross-check the decode against; it is probed when
    not supplied, which costs one ffprobe (~0.1s against a decode that costs
    seconds) and is worth it every time.

    Raises MediaError, once the decode has ended of its own accord, if it did
    not survey the whole file -- see _verify_decode for why that is an
    exception and not a flag. A consumer that walks away early (islice, an
    exception of its own) is never second-guessed: the decode was cut short on
    purpose, so there is nothing to verify.
    """
    depth = {"gray": 1, "rgb24": 3}[pix_fmt]
    frame_bytes = width * height * depth
    if duration is None:
        duration = (probe(path) or {}).get("duration") or 0.0

    cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-nostdin",
           # -progress is what makes the cross-check free. It reports the
           # output stream's furthest timestamp every half second, so "how far
           # into the file did this get" comes out of the decode we were
           # running anyway rather than a second pass over the container.
           "-progress", "pipe:2"]
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

    # stderr goes to a temp FILE, not a pipe, and that is the whole trick.
    # This used to be DEVNULL, which threw away the exit status and the error
    # text with it; the reason it was DEVNULL is that a consumer draining
    # stdout to exhaustion deadlocks the moment ffmpeg fills a stderr pipe,
    # which is also why the obvious `showinfo` filter (a line per frame, ~460
    # kB over a 90-minute lecture) was rejected for timestamps. A file cannot
    # fill, so capturing here reintroduces nothing: the no-deadlock property
    # is preserved by construction rather than by not looking. Timestamps
    # still come from frame_times(), one cheap ffprobe rather than a second
    # decode.
    errf = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=errf, bufsize=frame_bytes * 4)
    drained = False
    idx = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if not buf or len(buf) < frame_bytes:
                # A short read means ffmpeg closed stdout: the decode ended by
                # itself. That, and only that, is when the exit status and the
                # coverage are worth looking at.
                drained = True
                break
            if idx % stride == 0:
                arr = np.frombuffer(buf, dtype=np.uint8)
                arr = (arr.reshape(height, width) if depth == 1
                       else arr.reshape(height, width, depth))
                yield idx, arr
            idx += 1
    finally:
        try:
            proc.stdout.close()
        except OSError:
            pass
        returncode = None
        if drained:
            # ffmpeg has closed its output and is on the way out, so wait for
            # the real status. Killing here -- which this did unconditionally,
            # harmlessly, while nobody read the status -- would overwrite it
            # with -9 and make every healthy decode look aborted.
            try:
                returncode = proc.wait(timeout=60)
            except subprocess.TimeoutExpired:               # pragma: no cover
                proc.kill()
                returncode = proc.wait(timeout=60)
        else:
            # A consumer that stops early leaves ffmpeg writing into a pipe
            # nobody reads, so it has to be killed rather than waited on.
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=60)
        out_time, err_tail = _decode_report(errf)
        errf.close()
        if drained:
            _verify_decode(path, returncode, idx, out_time, duration,
                           err_tail)


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
    err = proc.stderr.decode("utf-8", "replace")
    if proc.returncode != 0:
        # Same fail-closed rule as the frame pass, for the same reason: a
        # short PCM buffer is silently a shorter lecture, and every level and
        # prosody figure below is a statistic over it. No deadlock risk here
        # -- subprocess.run drains both pipes concurrently, which is exactly
        # what iter_frames cannot do and works around with a temp file.
        tail = " | ".join(ln.strip() for ln in err.splitlines()
                          if ln.strip())[-FFMPEG_ERR_TAIL_BYTES:]
        raise MediaError(f"audio decode of {path} exited "
                         f"{proc.returncode}: {tail or 'no stderr captured'}")

    pcm = np.frombuffer(proc.stdout, dtype="<i2").astype(np.float32) / 32768.0
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
