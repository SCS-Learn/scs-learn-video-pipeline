"""Render a full lecture end to end in the theme layout, for both themes.

    python scripts/render_full_lecture.py
    python scripts/render_full_lecture.py --theme fun

Produces data/<lecture>/<key>-<theme>.mp4: the whole lecture, slide in the
left region of the supplied pip-frame art, the anonymized instructor-tracked
camera in the rail, every student question replaced by a full-frame card with
the sting mixed over it, and the theme's intro on the front.

The body is rendered ONCE and shared. The two themes differ only in their
intro, so re-rendering 80 minutes per theme would double the cost for a few
seconds of difference; the intro is concatenated onto a cached body instead.

Why the card is a separate segment rather than burned into the screen:
cards.py substitutes whole SCREEN frames, which in this layout would put the
card inside the 1380px slide slot -- shrinking its type and showing two sets of
CMU branding at once. The card is authored as a complete 1920x1080, so it
replaces the whole frame. That makes the timeline layout-vs-card rather than a
single filter graph, which is the same cut-render-concat shape cards.py uses.
"""

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from render_demo import (  # noqa: E402
    FPS, ASSETS, build_intro_fun, build_intro_pro, rail_overlay, run,
    SLIDE_W, SLIDE_H, SLIDE_Y, CAM_X, CAM_Y, CAM_W, CAM_H,
)
from render_theme_samples import THEMES, ROOT, W, H  # noqa: E402


# Resolved once, at first use. cards.py already owns the smoke-tested picker --
# it verifies an encoder by opening a real session rather than trusting
# `ffmpeg -encoders`, which lists h264_nvenc on GPUs that have no encoder block.
# On Apple Silicon this selects h264_videotoolbox: measured 17.4x realtime
# against 9.5x for libx264, and it runs on the media engine instead of pinning
# all ten cores. On PSC there is no VideoToolbox and no usable NVENC, so it
# falls through to libx264 -- which is correct there, not a fallback failure.
_ENC = None


def encoder(prefer="auto"):
    global _ENC
    if _ENC is None:
        from src.audio.cards import pick_encoder
        _ENC = pick_encoder(prefer)
    return _ENC


def vcodec(prefer="auto"):
    name, extra = encoder(prefer)
    return ["-c:v", name, *extra]


def probe_dur(path):
    return float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip())


def plan(questions, duration, pad=0.0):
    """Interleave layout and card segments across the whole lecture."""
    spans = sorted((max(0.0, q["start"] - pad), min(duration, q["end"] + pad), q)
                   for q in questions)
    merged, cur = [], None
    for a, b, q in spans:
        if cur and a <= cur[1]:
            cur = (cur[0], max(cur[1], b), cur[2])
        else:
            if cur:
                merged.append(cur)
            cur = (a, b, q)
    if cur:
        merged.append(cur)

    pieces, t = [], 0.0
    for a, b, q in merged:
        if a > t:
            pieces.append({"kind": "layout", "start": t, "dur": a - t})
        pieces.append({"kind": "card", "start": a, "dur": b - a, "q": q})
        t = b
    if t < duration:
        pieces.append({"kind": "layout", "start": t, "dur": duration - t})
    return pieces


def layout_segment(name, out, start, dur, lecture_dir, backdrop_png, _unused=None):
    """One static backdrop, two video overlays.

    pip-frame.png and the rail type are BOTH static, so compositing them
    per-frame burned an extra 1080p overlay stage on every one of 119,000
    frames for a picture that never changes. They are flattened into a single
    backdrop once, in build_body.
    """
    screen = os.path.join(lecture_dir, "screen_sync.mp4")
    cam = os.path.join(lecture_dir, "camera_muted_anon_tracked.mp4")
    fc = (
        f"[0:v]scale={SLIDE_W}:{SLIDE_H},setsar=1,fps={FPS}[sl];"
        f"[1:v]scale={CAM_W}:{CAM_H},setsar=1,fps={FPS}[cm];"
        f"[2:v][sl]overlay=0:{SLIDE_Y}:shortest=1[a];"
        f"[a][cm]overlay={CAM_X}:{CAM_Y},format=yuv420p[v]"
    )
    run(["ffmpeg", "-y", "-v", "error",
         "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", screen,
         "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", cam,
         "-loop", "1", "-t", f"{dur:.3f}", "-i", backdrop_png,
         "-filter_complex", fc, "-map", "[v]", "-map", "1:a",
         *vcodec(), "-c:a", "aac", "-ar", "48000", "-ac", "2",
         "-t", f"{dur:.3f}", out])
    return out


def card_segment(name, out, start, dur, text, lecture_dir, work, gain=0.22):
    from src.audio.cards import render_card
    cam = os.path.join(lecture_dir, "camera_muted_anon_tracked.mp4")
    png = os.path.join(work, f"card-{int(start)}.png")
    render_card(text, out_path=png)
    sound = os.path.join(ASSETS, name, "question-card-sound.mp3")

    fade = min(0.40, dur / 4)
    delay = int(min(800, dur * 1000 / 4))
    fc = (f"[1:a]volume={gain},adelay={delay}:all=1[s];"
          f"[2:a][s]amix=inputs=2:duration=first:normalize=0[aout];"
          f"[0:v]fps={FPS},fade=t=in:st=0:d={fade:.2f},"
          f"fade=t=out:st={dur - fade:.2f}:d={fade:.2f},format=yuv420p[v]")
    run(["ffmpeg", "-y", "-v", "error",
         "-loop", "1", "-t", f"{dur:.3f}", "-i", png,
         "-i", sound,
         "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", cam,
         "-filter_complex", fc, "-map", "[v]", "-map", "[aout]",
         *vcodec(), "-c:a", "aac", "-ar", "48000", "-ac", "2",
         "-t", f"{dur:.3f}", out])
    return out


def build_body(lecture_dir, work, theme_for_assets="professional"):
    """The lecture itself, identical for both themes."""
    cam = os.path.join(lecture_dir, "camera_muted_anon_tracked.mp4")
    duration = probe_dur(cam)
    qs = [{"start": s["start"], "end": s["end"], "text": s["text"].strip()}
          for s in json.load(open(os.path.join(
              lecture_dir, "transcript_classified.json")))
          if s.get("is_student_question")]
    pieces = plan(qs, duration)

    from PIL import Image
    backdrop = Image.open(os.path.join(ASSETS, theme_for_assets,
                                       "pip-frame.png")).convert("RGBA")
    backdrop = Image.alpha_composite(
        backdrop, rail_overlay(theme_for_assets, THEMES[theme_for_assets]))
    rail_png = os.path.join(work, "backdrop.png")
    backdrop.convert("RGB").save(rail_png)
    frame_png = None

    print(f"[full] {duration/60:.1f} min, {len(qs)} question cards, "
          f"{len(pieces)} segments", flush=True)

    outs, t0 = [], time.time()
    for i, p in enumerate(pieces):
        o = os.path.join(work, f"body_{i:03d}.mp4")
        if p["kind"] == "layout":
            layout_segment(theme_for_assets, o, p["start"], p["dur"],
                           lecture_dir, rail_png, frame_png)
        else:
            card_segment(theme_for_assets, o, p["start"], p["dur"],
                         p["q"]["text"], lecture_dir, work)
        outs.append(o)
        done = sum(x["dur"] for x in pieces[:i + 1])
        el = time.time() - t0
        rate = done / max(el, 1e-9)
        print(f"[full] segment {i+1}/{len(pieces)} ({p['kind']}, "
              f"{p['dur']:.0f}s) | {done/60:.1f}/{duration/60:.1f} min done, "
              f"{rate:.1f}x realtime, ETA {(duration-done)/max(rate,1e-9)/60:.0f} min",
              flush=True)

    lst = os.path.join(work, "body.txt")
    with open(lst, "w") as f:
        for o in outs:
            f.write(f"file '{os.path.abspath(o)}'\n")
    body = os.path.join(work, "body.mp4")
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
         "-i", lst, "-c", "copy", body])
    print(f"[full] body: {probe_dur(body)/60:.2f} min", flush=True)
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lecture-dir", default="data/15210-lecture12")
    ap.add_argument("--theme", choices=list(THEMES), default=None)
    ap.add_argument("--encoder", default="auto",
                    help="auto | h264_videotoolbox | h264_nvenc | libx264. "
                         "auto smoke-tests in that order.")
    ap.add_argument("--reuse-body", action="store_true",
                    help="Skip the body render if .full-work/body.mp4 exists")
    args = ap.parse_args()

    encoder(args.encoder)          # resolve and report once, before any work
    work = os.path.join(ROOT, ".full-work")
    os.makedirs(work, exist_ok=True)
    body = os.path.join(work, "body.mp4")
    if not (args.reuse_body and os.path.exists(body)):
        body = build_body(args.lecture_dir, work)

    key = os.path.basename(os.path.normpath(args.lecture_dir))
    for name in ([args.theme] if args.theme else list(THEMES)):
        if name == "fun":
            intro, _ = build_intro_fun(os.path.join(work, f"intro-{name}.mp4"), work)
        else:
            intro, _ = build_intro_pro(os.path.join(work, f"intro-{name}.mp4"),
                                       6.0, work)
        lst = os.path.join(work, f"final-{name}.txt")
        with open(lst, "w") as f:
            f.write(f"file '{os.path.abspath(intro)}'\n")
            f.write(f"file '{os.path.abspath(body)}'\n")
        out = os.path.join(args.lecture_dir, f"{key}-{name}.mp4")
        run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
             "-i", lst, "-c", "copy", "-movflags", "+faststart", out])
        print(f"[full] {out}  {probe_dur(out)/60:.2f} min, "
              f"{os.path.getsize(out)/1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
