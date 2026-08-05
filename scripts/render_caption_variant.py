"""Professional variant: YouTube-style caption instead of a full-frame card.

    python scripts/render_caption_variant.py

Writes data/<lecture>/<key>-professional-captions.mp4.

Differences from the card version, both requested:
  * a student question appears as a caption overlaid on the running lecture,
    so the slide and the instructor stay visible through it
  * no card sting -- the lecture's own (student-muted) audio is untouched

Reuses the cached layout segments from .full-work and re-renders only the
question spans, since only those change. Type is Open Sans, matching the cards.
"""

import argparse
import json
import os
import subprocess
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from render_full_lecture import (  # noqa: E402
    plan, layout_segment, probe_dur, run, encoder, vcodec,
)
from render_demo import (  # noqa: E402
    FPS, ASSETS, build_intro_pro, rail_overlay, SLIDE_W, SLIDE_Y, SLIDE_H,
    CAM_X, CAM_Y, CAM_W, CAM_H,
)
from render_theme_samples import THEMES, ROOT, W, H, font, fit_block, wrap  # noqa: E402

# The caption sits over the SLIDE region only. Centring it on the full frame
# would push it under the branded rail, which is opaque art.
CAP_CX = SLIDE_W // 2
CAP_MAX_W = 1120
CAP_PAD_X, CAP_PAD_Y = 34, 22
CAP_SIZE = 46
CAP_LEADING = 1.34
CAP_CY = H // 2            # "in the middle", as asked


def caption_png(text, out_path):
    """A YouTube-style caption: translucent slab, white Open Sans, centred.

    Drawn as RGBA and composited over the running video, so unlike the card it
    never replaces the picture.
    """
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    fnt = font("SemiBold", CAP_SIZE)
    lines = wrap(d, text, fnt, CAP_MAX_W)
    lead = int(CAP_SIZE * CAP_LEADING)

    tw = max(d.textlength(ln, font=fnt) for ln in lines)
    th = len(lines) * lead
    x0 = CAP_CX - tw / 2 - CAP_PAD_X
    x1 = CAP_CX + tw / 2 + CAP_PAD_X
    y0 = CAP_CY - th / 2 - CAP_PAD_Y
    y1 = CAP_CY + th / 2 + CAP_PAD_Y

    # 0.78 alpha: dark enough to carry white type over a bright slide, light
    # enough that the slide underneath still reads. Fully opaque would just be
    # a card again.
    d.rounded_rectangle([x0, y0, x1, y1], radius=14, fill=(0, 0, 0, 199))

    y = CAP_CY - th / 2
    for ln in lines:
        lw = d.textlength(ln, font=fnt)
        d.text((CAP_CX - lw / 2, y), ln, font=fnt, fill=(255, 255, 255, 255))
        y += lead
    lay.save(out_path)
    return out_path


def caption_segment(out, start, dur, text, lecture_dir, work, backdrop_png):
    """A layout segment with the caption faded in over it. No sting."""
    screen = os.path.join(lecture_dir, "screen_sync.mp4")
    cam = os.path.join(lecture_dir, "camera_muted_anon_tracked.mp4")
    cap = caption_png(text, os.path.join(work, f"cap-{int(start)}.png"))
    fade = min(0.35, dur / 5)

    fc = (
        f"[0:v]scale={SLIDE_W}:{SLIDE_H},setsar=1,fps={FPS}[sl];"
        f"[1:v]scale={CAM_W}:{CAM_H},setsar=1,fps={FPS}[cm];"
        f"[2:v][sl]overlay=0:{SLIDE_Y}:shortest=1[a];"
        f"[a][cm]overlay={CAM_X}:{CAM_Y}[b];"
        # -loop 1 on the caption input: a still image is one frame at t=0, so a
        # fade would otherwise never evaluate and the overlay stay invisible.
        f"[3:v]format=rgba,fade=t=in:st=0:d={fade:.2f}:alpha=1,"
        f"fade=t=out:st={dur - fade:.2f}:d={fade:.2f}:alpha=1[cap];"
        f"[b][cap]overlay=0:0:format=auto,format=yuv420p[v]"
    )
    run(["ffmpeg", "-y", "-v", "error",
         "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", screen,
         "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", cam,
         "-loop", "1", "-t", f"{dur:.3f}", "-i", backdrop_png,
         "-loop", "1", "-framerate", str(FPS), "-t", f"{dur:.3f}", "-i", cap,
         "-filter_complex", fc, "-map", "[v]", "-map", "1:a",
         *vcodec(), "-c:a", "aac", "-ar", "48000", "-ac", "2",
         "-t", f"{dur:.3f}", out])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lecture-dir", default="data/15210-lecture12")
    ap.add_argument("--encoder", default="libx264")
    args = ap.parse_args()
    encoder(args.encoder)

    work = os.path.join(ROOT, ".full-work")
    os.makedirs(work, exist_ok=True)
    LD = args.lecture_dir
    dur = probe_dur(os.path.join(LD, "camera_muted_anon_tracked.mp4"))
    qs = [{"start": s["start"], "end": s["end"], "text": s["text"].strip()}
          for s in json.load(open(os.path.join(LD, "transcript_classified.json")))
          if s.get("is_student_question")]
    pieces = plan(qs, dur)

    bd = Image.open(os.path.join(ASSETS, "professional", "pip-frame.png")).convert("RGBA")
    bd = Image.alpha_composite(bd, rail_overlay("professional", THEMES["professional"]))
    backdrop = os.path.join(work, "backdrop.png")
    bd.convert("RGB").save(backdrop)

    outs = []
    for i, p in enumerate(pieces):
        cached = os.path.join(work, f"v2_{i:03d}.mp4")
        o = os.path.join(work, f"cap_{i:03d}.mp4")
        if p["kind"] == "layout" and os.path.exists(cached):
            outs.append(cached)                      # unchanged, reuse
            print(f"  seg {i+1}/{len(pieces)} layout  {p['dur']:7.1f}s  (cached)",
                  flush=True)
        elif p["kind"] == "layout":
            layout_segment("professional", o, p["start"], p["dur"], LD, backdrop)
            outs.append(o)
            print(f"  seg {i+1}/{len(pieces)} layout  {p['dur']:7.1f}s", flush=True)
        else:
            caption_segment(o, p["start"], p["dur"], p["q"]["text"], LD, work,
                            backdrop)
            outs.append(o)
            print(f"  seg {i+1}/{len(pieces)} caption {p['dur']:7.1f}s  "
                  f"\"{p['q']['text'][:48]}\"", flush=True)

    lst = os.path.join(work, "body_cap.txt")
    with open(lst, "w") as f:
        for o in outs:
            f.write(f"file '{os.path.abspath(o)}'\n")
    body = os.path.join(work, "body_cap.mp4")
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
         "-i", lst, "-c", "copy", body])

    intro, _ = build_intro_pro(os.path.join(work, "intro-professional.mp4"),
                               6.0, work)
    fl = os.path.join(work, "final-cap.txt")
    with open(fl, "w") as f:
        f.write(f"file '{os.path.abspath(intro)}'\nfile '{os.path.abspath(body)}'\n")
    key = os.path.basename(os.path.normpath(LD))
    out = os.path.join(LD, f"{key}-professional-captions.mp4")
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
         "-i", fl, "-c", "copy", "-movflags", "+faststart", out])
    print(f"\n{out}  {probe_dur(out)/60:.2f} min, "
          f"{os.path.getsize(out)/1e6:.0f} MB")


if __name__ == "__main__":
    main()
