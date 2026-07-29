"""Burn student-question cards into the screen recording.

Why this no longer uses ffmpeg's overlay filter
-----------------------------------------------
assets/student-question-card-template.png is a *fully opaque* 1920x1080 image
and screen.mp4 is exactly 1920x1080, so during a question the card replaces the
frame outright -- there is nothing to composite. The previous approach opened
one full-length looped image input per question and chained one overlay per
question across the whole timeline, which costs O(n_questions x duration): for
lecture 12 (91.5 min, 137,309 frames) with 20 questions that is ~2.7M
synthesised 1080p frames on top of the real ones. `enable='between(...)'` only
gates whether the overlay composites; it does not stop the input from being
decoded or the filter from running. That, not the choice of encoder, is what
made this stage unrunnable on a cluster node.

Instead: cut the screen into the gaps between questions, render each card as a
clip of exactly its own length, and concat. Cost is O(duration), and because
every piece is an independent ffmpeg job the pieces encode in parallel
(--jobs), which is how you actually use a 40-core Bridges-2 node.

Why there is no NVENC path on PSC
---------------------------------
Verified on-cluster: PSC's `ffmpeg/4.3.1` module is built **without** nvenc at
all (hwaccels are only vdpau/vaapi/vulkan), so there is no GPU encode path
whichever node you land on. Separately, most Bridges-2 GPUs could not do it
anyway: the partition is heterogeneous (v100-16, v100-32, l40s-48, h100-80) and
the compute-class parts (V100/A100/H100) ship NVDEC but *no* NVENC encoder
block. Only L40S has one. CUDA compute availability says nothing about encode
silicon. This is why commit 0089000 "fix: run on gpu" moved to h264_nvenc and
57cf9f4 had to move back to libx264.

pick_encoder() therefore *smoke-tests* each candidate instead of trusting
`ffmpeg -encoders`, since a GPU with no encoder block still lists h264_nvenc and
fails only when the session is opened. nvenc is used on hardware that really has
it; `libx264 -threads <ncores>` everywhere else.

Output is video-only, matching the previous behaviour: screen.mp4 has no audio
track and src/assembly/assembly.py takes audio from camera_muted.mp4.

Usage:
    python -m src.audio.cards
    python -m src.audio.cards --jobs 20 --encoder libx264
    python -m src.audio.cards --dry-run
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageDraw, ImageFont

from src.audio.audio import merge_speaker_spans
from src.audio.transcription import get_instructor_label
from src.paths import LecturePaths

TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "assets",
    "student-question-card-template.png",
)
FONT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "assets", "fonts", "OpenSans-Regular.ttf"
)

BLACK = (0, 0, 0)

# The card template's native size. Screen frames are scaled into this so every
# concat segment shares one geometry.
CARD_W, CARD_H = 1920, 1080

# Safe text area, measured off the template rather than guessed. Its content
# bands are: y 0-44 top border, 95-207 CMU branding, 267-326 the "Student
# Question" heading, 1035-1079 bottom border. Text used to be laid out from
# y=300 against a height budget of HEIGHT-400=680, which is generous enough that
# an 8-line question passed the fit check at full size and was then centred to
# y=312 -- on top of a heading that runs to 326. A long question visibly
# collided with the heading in the published video.
TEXT_TOP, TEXT_BOTTOM = 360, 1010

# Above this, a card renders as a wall of small text. Measured: 3 of 4
# questions on lecture 12 came back at 18-79 chars, one at 311.
CARD_TEXT_BUDGET = 160


def cpu_count():
    """Cores actually available to this process (respects a Slurm cpuset)."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 4


# ---------------------------------------------------------------------------
# Encoder / hwaccel capability probing
# ---------------------------------------------------------------------------
def _encoder_listed(name):
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True
    ).stdout
    return re.search(rf"^\s*\S+\s+{re.escape(name)}\s", out, re.M) is not None


def _encoder_works(name, extra_args):
    """Actually open an encode session -- listing an encoder is not enough.

    A V100 node whose ffmpeg was built with --enable-nvenc still *lists*
    h264_nvenc; it fails only when the session is opened, because the GPU has no
    encoder block. This throwaway null encode is what tells the two cases apart.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=0.2",
        "-c:v", name, *extra_args, "-f", "null", "-",
    ]
    return subprocess.run(cmd, capture_output=True, text=True).returncode == 0


def encoder_candidates(threads):
    """(name, args) in preference order. Quality targets are ~equivalent."""
    return [
        ("h264_nvenc",
         ["-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "19", "-b:v", "0"]),
        ("h264_videotoolbox", ["-q:v", "55"]),
        ("libx264",
         ["-crf", "18", "-preset", "veryfast", "-threads", str(threads)]),
    ]


def pick_encoder(prefer="auto", threads=None):
    """Return (encoder_name, extra_args), smoke-testing each candidate.

    prefer="auto" walks the candidate list; an explicit name is still verified
    so a bad choice fails here with a clear message rather than mid-encode.
    """
    threads = threads or cpu_count()
    candidates = encoder_candidates(threads)

    if prefer and prefer != "auto":
        match = [c for c in candidates if c[0] == prefer] or [(prefer, [])]
        name, extra = match[0]
        if not _encoder_listed(name):
            raise RuntimeError(f"ffmpeg has no encoder {name!r} (check the build)")
        if not _encoder_works(name, extra):
            raise RuntimeError(
                f"{name!r} is listed by ffmpeg but fails to open an encode "
                f"session -- the GPU has no encoder block (true of V100/A100/"
                f"H100). Use --encoder libx264."
            )
        print(f"[cards] encoder: {name} (explicit)")
        return name, extra

    for name, extra in candidates:
        if not _encoder_listed(name):
            continue
        if not _encoder_works(name, extra):
            print(f"[cards] {name} listed but unusable (no encode hardware); skipping")
            continue
        print(f"[cards] encoder: {name} (auto)")
        return name, extra

    raise RuntimeError("no usable H.264 encoder found")


def cuda_decode_available():
    """True if ffmpeg can NVDEC-decode. V100 does have NVDEC (unlike NVENC)."""
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-hwaccels"], capture_output=True, text=True
    ).stdout
    if "cuda" not in out:
        return False
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-hwaccel", "cuda", "-f", "lavfi",
        "-i", "testsrc=size=320x240:rate=25:duration=0.2",
        "-f", "null", "-",
    ]
    return subprocess.run(cmd, capture_output=True, text=True).returncode == 0


# ---------------------------------------------------------------------------
# Probing the source
# ---------------------------------------------------------------------------
def _has_audio(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
        capture_output=True, text=True).stdout
    return "audio" in out


def probe_video(path):
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-show_entries", "format=duration",
            "-of", "json", path,
        ],
        capture_output=True, text=True, check=True,
    ).stdout
    info = json.loads(out)
    stream = info["streams"][0]
    num, _, den = stream["r_frame_rate"].partition("/")
    fps = float(num) / float(den or 1)
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "duration": float(info["format"]["duration"]),
    }


def get_duration(path):
    """Kept for callers that imported it from here."""
    return probe_video(path)["duration"]


# ---------------------------------------------------------------------------
# Card rendering (visual output unchanged)
# ---------------------------------------------------------------------------
def render_card(
    question_text,
    template_path=TEMPLATE_PATH,
    font_path=FONT_PATH,
    out_path="card.png",
):
    img = Image.open(template_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    WIDTH = CARD_W
    HEIGHT = CARD_H
    MAX_FONT_SIZE = 64
    MIN_FONT_SIZE = 32
    WRAP_WIDTH = 42

    font_size = MAX_FONT_SIZE
    avail_h = TEXT_BOTTOM - TEXT_TOP
    while font_size >= MIN_FONT_SIZE:
        font = ImageFont.truetype(font_path, font_size)
        wrapped = textwrap.fill(question_text, width=WRAP_WIDTH)
        bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=18)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        if text_w <= WIDTH - 200 and text_h <= avail_h:
            break
        font_size -= 4

    # bbox[1] is the ascender offset, not zero; ignoring it shifted every card
    # down by ~18px and pushed long text past the safe area.
    x = (WIDTH - text_w) / 2 - bbox[0]
    y = TEXT_TOP + (avail_h - text_h) / 2 - bbox[1]

    draw.multiline_text(
        (x, y), wrapped, font=font, fill=BLACK, spacing=18, align="center"
    )

    if len(question_text) > CARD_TEXT_BUDGET:
        print(f"[cards] WARNING card text is {len(question_text)} chars "
              f"(> {CARD_TEXT_BUDGET}); it will render small and dense. The "
              f"transcription step is meant to condense questions -- check "
              f"identify_student_questions() output.", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    img.save(out_path)
    return out_path


def get_span_text(segments, start, end):
    texts = []
    for seg in segments:
        if seg["start"] < end and seg["end"] > start:
            texts.append(seg["text"].strip())
    return " ".join(texts)


# ---------------------------------------------------------------------------
# Timeline planning -- everything on an integer frame grid so the pieces tile
# the output exactly and no rounding drift accumulates over 90 minutes.
# ---------------------------------------------------------------------------
def plan_timeline(question_intervals, duration, fps):
    """Return a list of pieces tiling [0, total_frames) with no gaps.

    Each piece is {"kind": "screen"|"card", "start_frame", "n_frames"} plus,
    for cards, "index" into question_intervals.
    """
    total = int(round(duration * fps))

    spans = []
    for i, iv in enumerate(question_intervals):
        a = max(0, int(round(iv["start"] * fps)))
        b = min(total, int(round(iv["end"] * fps)))
        if b > a:
            spans.append((a, b, i))
    spans.sort()

    # Clip overlaps rather than letting concat emit a shorter video than the
    # source: a later card starting inside an earlier one is pushed forward.
    merged = []
    for a, b, i in spans:
        if merged and a < merged[-1][1]:
            a = merged[-1][1]
            if b <= a:
                continue
        merged.append((a, b, i))

    pieces = []
    cursor = 0
    for a, b, i in merged:
        if a > cursor:
            pieces.append(
                {"kind": "screen", "start_frame": cursor, "n_frames": a - cursor}
            )
        pieces.append(
            {"kind": "card", "start_frame": a, "n_frames": b - a, "index": i}
        )
        cursor = b
    if cursor < total:
        pieces.append(
            {"kind": "screen", "start_frame": cursor, "n_frames": total - cursor}
        )
    return pieces


# ---------------------------------------------------------------------------
# Segment encoding
# ---------------------------------------------------------------------------
def _common_output_args(encoder, enc_args, fps):
    # Identical geometry, framerate, pixel format, codec and timescale across
    # every segment -- the concat demuxer stream-copies, so any mismatch shows
    # up as a glitch or a hard failure at join time.
    return [
        "-an",
        "-r", f"{fps:.10g}",
        "-pix_fmt", "yuv420p",
        "-c:v", encoder, *enc_args,
        "-video_track_timescale", "90000",
    ]


def _encode_screen_segment(screen_path, piece, out_path, encoder, enc_args, fps, hwaccel):
    start_sec = piece["start_frame"] / fps
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    if hwaccel == "cuda":
        cmd += ["-hwaccel", "cuda"]
    cmd += ["-ss", f"{start_sec:.6f}", "-i", screen_path]
    vf = (
        f"scale={CARD_W}:{CARD_H}:force_original_aspect_ratio=decrease,"
        f"pad={CARD_W}:{CARD_H}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    )
    cmd += ["-frames:v", str(piece["n_frames"]), "-vf", vf]
    cmd += _common_output_args(encoder, enc_args, fps) + [out_path]
    subprocess.run(cmd, check=True)
    return out_path


def _encode_card_segment(card_png, piece, out_path, encoder, enc_args, fps):
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-loop", "1", "-framerate", f"{fps:.10g}", "-i", card_png,
        "-frames:v", str(piece["n_frames"]),
        "-vf", f"scale={CARD_W}:{CARD_H},setsar=1",
    ]
    cmd += _common_output_args(encoder, enc_args, fps) + [out_path]
    subprocess.run(cmd, check=True)
    return out_path


def burn_question_cards(
    screen_path,
    segments,
    question_intervals,
    out_path,
    work_dir=None,
    encoder="auto",
    hwaccel="none",
    jobs=None,
    keep_work=False,
):
    """Replace each question span in screen_path with a rendered card.

    Returns out_path. Video-only, matching the previous behaviour.
    """
    info = probe_video(screen_path)
    fps, duration = info["fps"], info["duration"]

    if not question_intervals:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", screen_path,
             "-an", "-c:v", "copy", out_path],
            check=True,
        )
        print(f"[cards] no question intervals; stream-copied to {out_path}")
        return out_path

    pieces = plan_timeline(question_intervals, duration, fps)
    n_cards = sum(1 for p in pieces if p["kind"] == "card")
    total_frames = sum(p["n_frames"] for p in pieces)
    print(
        f"[cards] {info['width']}x{info['height']} @ {fps:g}fps, "
        f"{duration:.1f}s ({total_frames} frames) -> "
        f"{len(pieces)} segments, {n_cards} cards"
    )

    jobs = jobs or max(1, min(cpu_count(), len(pieces)))
    threads_per_job = max(1, cpu_count() // jobs)
    encoder_name, enc_args = pick_encoder(encoder, threads=threads_per_job)
    if hwaccel == "auto":
        hwaccel = "cuda" if cuda_decode_available() else "none"
    if hwaccel == "cuda":
        print("[cards] NVDEC decode enabled (decode only; V100 has no NVENC)")
    print(f"[cards] {jobs} parallel jobs x {threads_per_job} threads")

    owns_work_dir = work_dir is None
    if owns_work_dir:
        parent = os.path.dirname(os.path.abspath(out_path)) or "."
        os.makedirs(parent, exist_ok=True)
        work_dir = tempfile.mkdtemp(prefix="cards-", dir=parent)
    cards_dir = os.path.join(work_dir, "cards")
    segs_dir = os.path.join(work_dir, "segments")
    os.makedirs(cards_dir, exist_ok=True)
    os.makedirs(segs_dir, exist_ok=True)

    try:
        # Render the card PNGs first (cheap, and needed before any encode).
        for p in pieces:
            if p["kind"] != "card":
                continue
            iv = question_intervals[p["index"]]
            text = get_span_text(segments, iv["start"], iv["end"])
            p["png"] = render_card(
                text, out_path=os.path.join(cards_dir, f"card_{p['index']:03d}.png")
            )

        def encode(args):
            i, p = args
            out = os.path.join(segs_dir, f"seg_{i:04d}.mp4")
            if p["kind"] == "card":
                _encode_card_segment(p["png"], p, out, encoder_name, enc_args, fps)
            else:
                _encode_screen_segment(
                    screen_path, p, out, encoder_name, enc_args, fps, hwaccel
                )
            print(f"[cards] segment {i + 1}/{len(pieces)} "
                  f"({p['kind']}, {p['n_frames']} frames)", flush=True)
            return out

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            seg_paths = list(pool.map(encode, enumerate(pieces)))

        list_path = os.path.join(work_dir, "concat.txt")
        with open(list_path, "w") as f:
            for s in seg_paths:
                f.write(f"file '{os.path.abspath(s)}'\n")

        # Segments are encoded video-only so the concat demuxer sees uniform
        # streams. If the SOURCE carries audio, mux it back afterwards: that is
        # the case when cards are burned into the camera rather than the screen
        # (screen.mp4 has no audio track, the camera has the muted one), and
        # silently dropping it would produce a mute deliverable.
        src_has_audio = _has_audio(screen_path)
        concat_target = (os.path.join(work_dir, "concat_out.mp4")
                         if src_has_audio else out_path)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
             "-i", list_path, "-c", "copy", "-movflags", "+faststart",
             concat_target],
            check=True,
        )
        if src_has_audio:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-i", concat_target, "-i", screen_path,
                 "-map", "0:v", "-map", "1:a",
                 "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest",
                 "-movflags", "+faststart", out_path],
                check=True,
            )
            print("[cards] carried the source audio track through")
    finally:
        if owns_work_dir and not keep_work:
            shutil.rmtree(work_dir, ignore_errors=True)

    final = probe_video(out_path)
    print(f"[cards] wrote {out_path} ({final['duration']:.1f}s, "
          f"source {duration:.1f}s, {n_cards} cards burned in)")
    return out_path


def overlay_question_cards(
    screen_path,
    segments,
    question_intervals,
    instructor_label=None,
    out_path="screen_with_cards.mp4",
    **kwargs,
):
    """Back-compat wrapper. instructor_label was never used by this stage."""
    kwargs.pop("width", None)
    kwargs.pop("height", None)
    return burn_question_cards(
        screen_path, segments, question_intervals, out_path, **kwargs
    )


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--lecture-dir", default="data/15210-lecture12")
    parser.add_argument("--transcript", default=None,
                        help="Default: <lecture-dir>/transcript_classified.json")
    parser.add_argument("--screen", default=None,
                        help="Default: <lecture-dir>/screen_sync.mp4")
    parser.add_argument("--out", default=None,
                        help="Default: <lecture-dir>/screen_with_cards.mp4")
    parser.add_argument("--encoder", default="auto",
                        help="auto | libx264 | h264_nvenc | h264_videotoolbox")
    parser.add_argument("--hwaccel", default="none", choices=["none", "cuda", "auto"],
                        help="NVDEC decode acceleration (encode is CPU on V100)")
    parser.add_argument("--jobs", type=int, default=None,
                        help=f"Parallel segment encodes "
                             f"(default: min(cores={cpu_count()}, segments))")
    parser.add_argument("--keep-work", action="store_true",
                        help="Keep the intermediate segments/ and cards/ dirs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the segment plan and exit without encoding")
    args = parser.parse_args()

    lp = LecturePaths(args.lecture_dir)
    screen = args.screen or lp.screen_sync
    out = args.out or lp.screen_with_cards
    transcript = args.transcript or lp.resolve_transcript_classified()

    with open(transcript) as f:
        segments = json.load(f)

    instructor_label = get_instructor_label(segments)
    intervals = merge_speaker_spans(segments, instructor_label)
    question_intervals = [iv for iv in intervals if iv["is_student_question"]]

    for i, iv in enumerate(question_intervals):
        print(f"{i}: {iv['start']:.2f} - {iv['end']:.2f}")

    if args.dry_run:
        info = probe_video(screen)
        pieces = plan_timeline(question_intervals, info["duration"], info["fps"])
        card_frames = sum(p["n_frames"] for p in pieces if p["kind"] == "card")
        all_frames = sum(p["n_frames"] for p in pieces)
        print(f"\n[cards] plan for {screen} "
              f"({info['duration']:.1f}s @ {info['fps']:g}fps):")
        for i, p in enumerate(pieces):
            t0 = p["start_frame"] / info["fps"]
            print(f"  {i:4d} {p['kind']:6s} {t0:9.2f}s +{p['n_frames']:7d} frames")
        print(f"[cards] {len(pieces)} segments, "
              f"{card_frames} card frames of {all_frames}")
        return

    burn_question_cards(
        screen_path=screen,
        segments=segments,
        question_intervals=question_intervals,
        out_path=out,
        encoder=args.encoder,
        hwaccel=args.hwaccel,
        jobs=args.jobs,
        keep_work=args.keep_work,
    )


if __name__ == "__main__":
    main()
