"""Build a 20s demo per theme: the intro running into the live layout.

    python scripts/render_demo.py
    python scripts/render_demo.py --theme fun --seconds 20

The two themes differ ONLY in the intro. Everything after it -- the layout, the
question card, the card sound -- is shared, and the supplied assets for those
are byte-identical between the themes.

    professional  intro-card-professional.png   still, details overlaid
    fun           intro-card-fun.mp4            motion, details overlaid while
                                                the notecard is settled

Supplied art is a backdrop only: pip-frame.png carries no type and no camera
frame, and neither intro card carries the lecture details. This module draws
those on top and composites the video into the regions the art leaves for it.

Normalised to 1920x1080 @ 25 fps, 48 kHz stereo throughout. The segments are
joined with the concat demuxer (a stream copy), and the fun intro is natively
30 fps -- feeding that in unconverted yields a file that plays one segment and
then stalls rather than failing loudly.
"""

import argparse
import os
import subprocess
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# The repo root too, so `src.audio.cards` resolves when this is run as a script
# from anywhere rather than only from the project directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from render_theme_samples import (  # noqa: E402
    W, H, THEMES, ROOT, COURSE_NUM, COURSE_TITLE, LECTURE, TERM,
    font, serif, fit_block, wrap,
)

FPS = 25
ASSETS = os.path.join(ROOT, "assets", "themes")

# --- layout geometry, measured off the supplied pip-frame.png ----------------
# The art is black over x 0..1379 and patterned red over x 1380..1919, which
# matches the Figma exactly. Nothing here is a guess.
RAIL_X, RAIL_W = 1380, W - 1380
SLIDE_W, SLIDE_H = RAIL_X, int(RAIL_X * 9 / 16)
SLIDE_Y = (H - SLIDE_H) // 2
CAM_M = 30
CAM_X, CAM_W = RAIL_X + CAM_M, RAIL_W - 2 * CAM_M
CAM_H = int(CAM_W * 9 / 16)
CAM_Y = 60

# --- fun intro, measured off intro-card-fun.mp4 -----------------------------
# The notecard animates in, holds perfectly still from 3.00s to 5.40s (frame
# difference is exactly 0.00 across that span), then leaves. Details are only
# legible while it is still, so they live inside that window.
FUN_HOLD_IN, FUN_HOLD_OUT = 3.00, 5.40
FUN_MARGIN_X = 400          # just right of the card's red margin rule at x=360
# The printed writing lines run 409, 536, 663, 794 (127px apart). Text is
# centred in the GAPS between them rather than sitting on them: on the line, the
# first row crowded the printed "School of Computer Science" directly above it,
# and every row had a rule cutting through its descenders.
FUN_RULES = (409, 536, 663, 794)
FUN_SLOTS = tuple((FUN_RULES[i] + FUN_RULES[i + 1]) // 2
                  for i in range(len(FUN_RULES) - 1))   # 472, 599, 728

# --- professional intro -----------------------------------------------------
# Its lockup is printed at x 107..715, y 106..209, so the overlay aligns to
# x=107 rather than inventing a margin.
PRO_X = 107


def draw_fun_details(lay):
    """Write the lecture details into the gaps between the notecard's rules.

    anchor="lm" puts the given y at the text's vertical middle, so each row
    lands centred in its slot regardless of font size.
    """
    d = ImageDraw.Draw(lay)
    ink = (24, 26, 34, 255)
    avail = 1689 - FUN_MARGIN_X - 40

    fnt, lines, _ = fit_block(d, LECTURE, "Bold", avail, 120, start=64)
    d.text((FUN_MARGIN_X, FUN_SLOTS[0]), lines[0], font=fnt, fill=ink,
           anchor="lm")

    sub = f"{COURSE_NUM}: {COURSE_TITLE}"
    sf, slines, _ = fit_block(d, sub, "Regular", avail, 120, start=42)
    d.text((FUN_MARGIN_X, FUN_SLOTS[1]), slines[0], font=sf, fill=ink,
           anchor="lm")

    d.text((FUN_MARGIN_X, FUN_SLOTS[2]), TERM, font=font("Regular", 40),
           fill=(90, 92, 104, 255), anchor="lm")
    return lay


def draw_pro_details(img):
    """Details onto the still professional card, under its printed lockup."""
    d = ImageDraw.Draw(img)
    red, ink, muted = (196, 18, 48), (26, 26, 26), (90, 90, 95)
    fnt, lines, lead = fit_block(d, LECTURE, "Bold", W - 2 * PRO_X - 60, 240,
                                 start=96)
    y = 520
    for ln in lines:
        d.text((PRO_X, y), ln, font=fnt, fill=red)
        y += lead
    sub = f"{COURSE_NUM}: {COURSE_TITLE}"
    sf, slines, slead = fit_block(d, sub, "SemiBold", W - 2 * PRO_X, 140, start=46)
    for ln in slines:
        d.text((PRO_X, y), ln, font=sf, fill=ink)
        y += slead
    d.text((PRO_X, 930), TERM, font=font("Regular", 40), fill=muted)
    return img


def rail_overlay(name, t):
    """Type and camera frame for the rail. Transparent everywhere else, so the
    supplied pip-frame art shows through untouched."""
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    d.rectangle([CAM_X - 3, CAM_Y - 3, CAM_X + CAM_W + 3, CAM_Y + CAM_H + 3],
                outline=(255, 255, 255, 255), width=3)

    y = CAM_Y + CAM_H + 70
    d.text((CAM_X, y), COURSE_NUM, font=font("ExtraBold", 92),
           fill=(255, 255, 255, 255))
    y += 118
    lf = font("SemiBold", 34)
    for ln in wrap(d, LECTURE, lf, CAM_W):
        d.text((CAM_X, y), ln, font=lf, fill=(255, 255, 255, 255))
        y += 44
    d.text((CAM_X, y + 6), TERM, font=font("Regular", 32),
           fill=(255, 235, 238, 255))
    # The CMU seal and wordmark are PRINTED INTO pip-frame.png now, occupying
    # y 926..1057 of the rail. Drawing our own would double them up, so this
    # layer stops at the term and leaves everything below y=926 to the art.
    return lay


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"ffmpeg failed:\n{' '.join(cmd)}\n{r.stderr[-1500:]}")


def build_intro_pro(out, seconds, work):
    card = os.path.join(ASSETS, "professional", "intro-card-professional.png")
    img = Image.open(card).convert("RGB")
    draw_pro_details(img)
    png = os.path.join(work, "pro-intro.png")
    img.save(png)
    run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-t", str(seconds), "-i", png,
         "-f", "lavfi", "-t", str(seconds),
         "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
         "-vf", f"fps={FPS},format=yuv420p,fade=t=out:st={seconds - 0.5}:d=0.5",
         "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-c:a", "aac", "-ar", "48000", "-ac", "2", "-shortest", out])
    return out, seconds


def build_intro_fun(out, work):
    """Overlay the details for exactly as long as the notecard is settled.

    Runs the clip's full length so its own animation and audio sting survive;
    only the type is time-gated.
    """
    clip = os.path.join(ASSETS, "fun", "intro-card-fun.mp4")
    lay = draw_fun_details(Image.new("RGBA", (W, H), (0, 0, 0, 0)))
    png = os.path.join(work, "fun-details.png")
    lay.save(png)

    clip_dur = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", clip], capture_output=True, text=True).stdout.strip())

    fade = 0.25
    fc = (
        f"[0:v]fps={FPS},scale={W}:{H},setsar=1[bg];"
        f"[1:v]format=rgba,"
        f"fade=t=in:st={FUN_HOLD_IN:.2f}:d={fade}:alpha=1,"
        f"fade=t=out:st={FUN_HOLD_OUT - fade:.2f}:d={fade}:alpha=1[ov];"
        f"[bg][ov]overlay=0:0:format=auto,format=yuv420p[v]"
    )
    # -loop 1 is load-bearing. A still image input is a SINGLE frame at t=0, so
    # a fade starting at 3s never evaluates -- alpha stays 0 and the overlay is
    # invisible for the whole clip, with ffmpeg reporting success.
    run(["ffmpeg", "-y", "-v", "error", "-i", clip,
         "-loop", "1", "-framerate", str(FPS), "-t", f"{clip_dur:.3f}", "-i", png,
         "-filter_complex", fc, "-map", "[v]", "-map", "0:a",
         "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-c:a", "aac", "-ar", "48000", "-ac", "2", out])
    dur = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", out], capture_output=True, text=True).stdout.strip())
    return out, dur


def build_layout(name, t, out, seconds, start, lecture_dir, work):
    screen = os.path.join(lecture_dir, "screen_sync.mp4")
    cam = os.path.join(lecture_dir, "camera_muted_anon_tracked.mp4")
    frame = os.path.join(ASSETS, name, "pip-frame.png")
    for p in (screen, cam, frame):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")

    rail = os.path.join(work, f"rail-{name}.png")
    rail_overlay(name, t).save(rail)

    fc = (
        f"[2:v]scale={W}:{H},setsar=1[bgart];"
        f"[0:v]scale={SLIDE_W}:{SLIDE_H},setsar=1,fps={FPS}[sl];"
        f"[1:v]scale={CAM_W}:{CAM_H},setsar=1,fps={FPS}[cm];"
        f"[bgart][sl]overlay=0:{SLIDE_Y}:shortest=1[a];"
        f"[a][3:v]overlay=0:0[b];"
        f"[b][cm]overlay={CAM_X}:{CAM_Y}[v0];"
        f"[v0]fade=t=in:st=0:d=0.4,"
        f"fade=t=out:st={max(0.0, seconds - 0.3):.2f}:d=0.3,format=yuv420p[v]"
    )
    run(["ffmpeg", "-y", "-v", "error",
         "-ss", str(start), "-t", str(seconds), "-i", screen,
         "-ss", str(start), "-t", str(seconds), "-i", cam,
         "-loop", "1", "-t", str(seconds), "-i", frame,
         "-i", rail,
         "-filter_complex", fc, "-map", "[v]", "-map", "1:a",
         "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-c:a", "aac", "-ar", "48000", "-ac", "2", "-t", str(seconds), out])
    return out


def build_card_segment(name, out, lecture_dir, work, q_text, q_start, card):
    """A question card as a clip: full frame, eased in and out, sting mixed.

    Audio is the lecture's own track across the question -- which the audio
    stage has muted, by design -- with the sting over it, delayed so it lands
    on the card rather than on the cut.
    """
    from src.audio.cards import render_card

    cam = os.path.join(lecture_dir, "camera_muted_anon_tracked.mp4")
    png = os.path.join(work, f"{name}-card-{int(q_start)}.png")
    render_card(q_text, out_path=png)
    sound = os.path.join(ASSETS, name, "question-card-sound.mp3")

    fade_v, sting_delay = 0.40, 800
    fc = (f"[1:a]volume=0.22,adelay={sting_delay}:all=1[s];"
          f"[2:a][s]amix=inputs=2:duration=first:normalize=0[aout];"
          f"[0:v]fps={FPS},fade=t=in:st=0:d={fade_v},"
          f"fade=t=out:st={card - fade_v:.2f}:d={fade_v},format=yuv420p[v]")
    run(["ffmpeg", "-y", "-v", "error",
         "-loop", "1", "-t", str(card), "-i", png,
         "-i", sound,
         "-ss", str(q_start), "-t", str(card), "-i", cam,
         "-filter_complex", fc, "-map", "[v]", "-map", "[aout]",
         "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-c:a", "aac", "-ar", "48000", "-ac", "2", "-t", str(card), out])
    return out


def build_full_demo(name, t, out_dir, lecture_dir, work):
    """The whole story for one lecture, in one file.

    intro -> opening of the lecture -> a REAL student question (card + sting)
    -> the lecture resuming. Question and timings come from lecture 12's own
    transcript, not from invented content.
    """
    Q_START, Q_TEXT = 645.6, ("After two additions, do new versions point "
                              "directly to unchanged nodes, or through the "
                              "previous version?")
    parts = []

    if name == "fun":
        a, intro_dur = build_intro_fun(os.path.join(work, f"{name}-f-a.mp4"), work)
    else:
        a, intro_dur = build_intro_pro(os.path.join(work, f"{name}-f-a.mp4"), 6.0, work)
    parts.append(a)

    # the lecture's actual opening
    b = os.path.join(work, f"{name}-f-b.mp4")
    build_layout(name, t, b, 12.0, 0.0, lecture_dir, work)
    parts.append(b)

    # run-up to the question, the card itself, then the lecture resuming
    c = os.path.join(work, f"{name}-f-c.mp4")
    build_layout(name, t, c, 5.0, Q_START - 5.0, lecture_dir, work)
    parts.append(c)

    parts.append(build_card_segment(name, os.path.join(work, f"{name}-f-d.mp4"),
                                    lecture_dir, work, Q_TEXT, Q_START, 7.0))

    e = os.path.join(work, f"{name}-f-e.mp4")
    build_layout(name, t, e, 6.0, Q_START + 20.0, lecture_dir, work)
    parts.append(e)

    lst = os.path.join(work, f"{name}-full.txt")
    with open(lst, "w") as f:
        for s in parts:
            f.write(f"file '{os.path.abspath(s)}'\n")
    out = os.path.join(out_dir, f"15210-lecture12-{name}.mp4")
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
         "-i", lst, "-c", "copy", "-movflags", "+faststart", out])
    return out, intro_dur + 12.0 + 5.0 + 7.0 + 6.0


def build_question_demo(name, t, out_dir, lecture_dir, work,
                        q_start=645.6, before=5.0, card=6.0, after=4.0):
    """Layout -> full-frame question card -> layout, with the sting mixed in.

    The card takes the WHOLE frame, not the slide region. It is authored as a
    complete 1920x1080 with its own lockup and rail; dropping it into the
    1380px slide slot would shrink its type and show two sets of CMU branding
    at once. This is also what cards.py does upstream -- it substitutes whole
    screen frames -- so the card genuinely replaces the picture.

    Defaults straddle a REAL flagged question in lecture 12 (645.6-664.0s).
    """
    from src.audio.cards import render_card

    cam = os.path.join(lecture_dir, "camera_muted_anon_tracked.mp4")
    seg = []

    a = os.path.join(work, f"{name}-q-a.mp4")
    build_layout(name, t, a, before, q_start - before, lecture_dir, work)
    seg.append(a)

    # The card itself. Audio is the lecture's own track across the question --
    # which the audio stage has muted, by design -- with the sting over it.
    png = os.path.join(work, f"{name}-q-card.png")
    render_card("After two additions, do new versions point directly to "
                "unchanged nodes, or through the previous version?",
                out_path=png)
    sound = os.path.join(ASSETS, name, "question-card-sound.mp3")
    b = os.path.join(work, f"{name}-q-b.mp4")
    # Ease in and out rather than cutting. A hard cut from a dark layout to a
    # white full-frame card with the sting landing on the same frame reads as
    # whiplash; a short dip plus a beat before the sound lets it arrive.
    fade_v, sting_delay = 0.40, 800
    fc = (f"[1:a]volume=0.22,adelay={sting_delay}:all=1[s];"
          f"[2:a][s]amix=inputs=2:duration=first:normalize=0[aout];"
          f"[0:v]fps={FPS},fade=t=in:st=0:d={fade_v},"
          f"fade=t=out:st={card - fade_v:.2f}:d={fade_v},format=yuv420p[v]")
    run(["ffmpeg", "-y", "-v", "error",
         "-loop", "1", "-t", str(card), "-i", png,
         "-i", sound,
         "-ss", str(q_start), "-t", str(card), "-i", cam,
         "-filter_complex", fc,
         "-map", "[v]", "-map", "[aout]",
         "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-c:a", "aac", "-ar", "48000", "-ac", "2", "-t", str(card), b])
    seg.append(b)

    c = os.path.join(work, f"{name}-q-c.mp4")
    build_layout(name, t, c, after, q_start + 20, lecture_dir, work)
    seg.append(c)

    lst = os.path.join(work, f"{name}-q.txt")
    with open(lst, "w") as f:
        for s in seg:
            f.write(f"file '{os.path.abspath(s)}'\n")
    out = os.path.join(out_dir, "demo-question.mp4")
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
         "-i", lst, "-c", "copy", out])
    return out, before + card + after


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theme", choices=list(THEMES), default=None)
    ap.add_argument("--lecture-dir", default="data/15210-lecture12")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--intro-seconds", type=float, default=6.0,
                    help="Professional only; the fun intro runs its clip's length")
    # t=0 of screen_sync/camera IS the start of the lecture: sync aligned the
    # two streams and dropped the pre-lecture black, so the title slide is up
    # at 0.00s. The demo therefore opens on the real opening rather than a
    # sample taken from the middle.
    ap.add_argument("--start", type=float, default=0.0,
                    help="Seconds into the aligned lecture for the live section "
                         "(0 = the lecture's actual start)")
    args = ap.parse_args()

    work = os.path.join(ROOT, ".demo-work")
    os.makedirs(work, exist_ok=True)

    for name in ([args.theme] if args.theme else list(THEMES)):
        t = THEMES[name]
        out_dir = os.path.join(ROOT, "assets", "themes", name, "samples")
        os.makedirs(out_dir, exist_ok=True)

        if name == "fun":
            a, intro_dur = build_intro_fun(os.path.join(work, f"{name}-a.mp4"), work)
        else:
            a, intro_dur = build_intro_pro(os.path.join(work, f"{name}-a.mp4"),
                                           args.intro_seconds, work)
        live = max(1.0, args.seconds - intro_dur)
        b = build_layout(name, t, os.path.join(work, f"{name}-b.mp4"),
                         live, args.start, args.lecture_dir, work)

        lst = os.path.join(work, f"{name}.txt")
        with open(lst, "w") as f:
            f.write(f"file '{os.path.abspath(a)}'\nfile '{os.path.abspath(b)}'\n")
        out = os.path.join(out_dir, "demo-20s.mp4")
        run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
             "-i", lst, "-c", "copy", out])

        dur = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", out], capture_output=True, text=True).stdout.strip()
        print(f"  {name}/demo-20s.mp4  {float(dur):.2f}s "
              f"({intro_dur:.2f}s intro + {live:.2f}s live), "
              f"{os.path.getsize(out) / 1e6:.1f} MB")

        qout, qdur = build_question_demo(name, t, out_dir, args.lecture_dir, work)
        print(f"  {name}/demo-question.mp4  {qdur:.1f}s "
              f"(layout -> question card -> layout, sting mixed), "
              f"{os.path.getsize(qout) / 1e6:.1f} MB")

        fout, fdur = build_full_demo(name, t, out_dir, args.lecture_dir, work)
        real = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", fout], capture_output=True, text=True).stdout.strip()
        print(f"  {name}/{os.path.basename(fout)}  {float(real):.1f}s  "
              f"intro -> lecture open -> student question -> lecture, "
              f"{os.path.getsize(fout) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
