"""Render sample cards for each theme, so a look can be judged before it is wired in.

    python scripts/render_theme_samples.py
    python scripts/render_theme_samples.py --theme professional

Writes assets/themes/<theme>/samples/{intro,question,layout}.png at 1920x1080.

Geometry is not decorative. screen.mp4 is exactly 1920x1080 and a card *replaces*
the frame rather than compositing over it, so every card must be a complete,
fully opaque 1080p image -- see the note in cards.py about not reintroducing an
ffmpeg overlay chain. The question card additionally has to reserve a band for
text that cards.py draws at render time; TEXT_BAND below is that contract.
"""

import argparse
import os
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_DIR = os.path.join(ROOT, "assets", "fonts", "static")

# Sample content, taken from the Figma so the samples show real strings rather
# than lorem ipsum -- long questions are exactly where these designs break.
COURSE_NUM = "15-210"
COURSE_TITLE = "Parallel and Sequential Data Structures and Algorithms"
LECTURE = "Lecture 12: Binary Search Trees"
TERM = "Spring 2026"
QUESTION = ("If you do two additions in a row, does the second one point back "
            "to the version from two additions ago — or to the node "
            "created one addition ago?")


def font(name, size):
    """Open Sans by weight. The family lives in assets/fonts/static/."""
    path = os.path.join(FONT_DIR, f"OpenSans-{name}.ttf")
    if not os.path.exists(path):
        raise SystemExit(f"missing font {path}")
    return ImageFont.truetype(path, size)


def serif(size):
    """A high-contrast serif for the CMU wordmark. Bodoni is the closest thing
    macOS ships to the real wordmark face; Georgia is the fallback."""
    for path, idx in (("/System/Library/Fonts/Supplemental/Bodoni 72.ttc", 1),
                      ("/System/Library/Fonts/Supplemental/Georgia Bold.ttf", 0)):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size, index=idx)
            except Exception:
                continue
    return font("Bold", size)


def tracked(draw, xy, text, fnt, fill, tracking):
    """Draw text with letter spacing. PIL has no tracking, and the design's
    'STUDENT QUESTION' label depends on it."""
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=fnt, fill=fill)
        x += draw.textlength(ch, font=fnt) + tracking
    return x


def wrap(draw, text, fnt, max_w):
    words, lines, cur = text.split(), [], ""
    for w_ in words:
        trial = f"{cur} {w_}".strip()
        if draw.textlength(trial, font=fnt) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w_
    if cur:
        lines.append(cur)
    return lines


def fit_block(draw, text, weight, max_w, max_h, start, floor=30, leading=1.34):
    """Largest size at which the wrapped text fits the band. Mirrors what
    cards.py does, so the sample shows the real worst case."""
    size = start
    while size >= floor:
        fnt = font(weight, size)
        lines = wrap(draw, text, fnt, max_w)
        if len(lines) * int(size * leading) <= max_h:
            return fnt, lines, int(size * leading)
        size -= 2
    fnt = font(weight, floor)
    return fnt, wrap(draw, text, fnt, max_w), int(floor * leading)


# ---------------------------------------------------------------------------
# themes
# ---------------------------------------------------------------------------
THEMES = {
    "professional": dict(
        bg=(255, 255, 255), ink=(26, 26, 26), muted=(90, 90, 95),
        accent=(196, 18, 48),            # Carnegie Red #C41230
        accent_dark=(140, 11, 30),
        panel=(196, 18, 48), panel_ink=(255, 255, 255),
        rail=33, margin=120, radius=0,
    ),
    "fun": dict(
        bg=(18, 19, 26), ink=(255, 255, 255), muted=(168, 172, 190),
        accent=(255, 184, 28),           # gold
        accent_dark=(196, 18, 48),
        panel=(196, 18, 48), panel_ink=(255, 255, 255),
        rail=0, margin=130, radius=44,
    ),
}


def wordmark(draw, t, x, y, on_dark=False):
    """Carnegie Mellon University / School of Computer Science lockup."""
    red = (255, 255, 255) if on_dark else t["accent"]
    sub = (255, 255, 255) if on_dark else (26, 26, 26)
    draw.text((x, y), "Carnegie Mellon University", font=serif(40), fill=red)
    draw.text((x, y + 52), "School of Computer Science",
              font=font("Light", 38), fill=sub)


# ---------------------------------------------------------------------------
def render_intro(name, t):
    img = Image.new("RGB", (W, H), t["bg"])
    d = ImageDraw.Draw(img)
    m = t["margin"]

    if name == "fun":
        # Chunky diagonal ribbon + dot confetti, so the two demos read as
        # different at a glance rather than as a recolour. The ribbon is pinned
        # to the bottom strip: at its first height it ran up under the subtitle,
        # putting grey body text on a red field.
        d.polygon([(0, H - 250), (W, H - 360), (W, H), (0, H)], fill=t["accent_dark"])
        d.polygon([(0, H - 150), (W, H - 258), (W, H), (0, H)], fill=t["accent"])
        for i, (cx, cy, r) in enumerate([(1620, 180, 15), (1720, 260, 9),
                                         (1540, 300, 11), (1780, 150, 7)]):
            d.ellipse([cx - r, cy - r, cx + r, cy + r],
                      fill=t["accent"] if i % 2 else t["panel"])
        # The red rail, matching the professional intro and the question card.
        # Drawn last so the ribbon cannot run over it at the bottom edge.
        d.rectangle([0, 0, 33, H], fill=t["panel"])
    else:
        d.rectangle([0, 0, t["rail"], H], fill=t["accent"])

    wordmark(d, t, m, 100, on_dark=(name == "fun"))

    title_w = "ExtraBold" if name == "fun" else "Bold"
    fnt, lines, lead = fit_block(d, LECTURE, title_w, W - 2 * m - 60, 240,
                                 start=104 if name == "fun" else 96)
    y = 470 if name == "fun" else 520
    for ln in lines:
        d.text((m, y), ln, font=fnt, fill=t["accent"] if name == "professional"
               else t["ink"])
        y += lead

    if name == "fun":
        d.rounded_rectangle([m, y + 14, m + 240, y + 32], radius=9, fill=t["accent"])
        y += 40

    sub = f"{COURSE_NUM}: {COURSE_TITLE}"
    sf, slines, slead = fit_block(d, sub, "SemiBold", W - 2 * m, 140, start=46)
    for ln in slines:
        d.text((m, y), ln, font=sf, fill=t["ink"] if name == "professional"
               else t["muted"])
        y += slead

    # On the fun theme this sits on the gold ribbon, so it takes the dark ink --
    # white on #FFB81C is about 1.8:1 and unreadable.
    d.text((m, 930), TERM, font=font("Regular", 40),
           fill=t["muted"] if name == "professional" else (18, 19, 26))
    return img


def render_question(name, t, text=QUESTION):
    """text=None renders the blank template cards.py draws the question into."""
    img = Image.new("RGB", (W, H), t["bg"])
    d = ImageDraw.Draw(img)
    m = t["margin"]
    blank = text is None
    if blank:
        # Lay the bubble out for a typical three-line question so the template
        # is a fixed shape; cards.py only fills the band, it cannot resize art.
        text = QUESTION

    # The band cards.py will draw into. Kept clear of the label above and the
    # bottom edge below; a long question is what used to collide with the
    # heading (commit 135934a).
    top, bottom = 371, 1010
    tx = m
    if name == "fun":
        tx, top, bottom = m + 10, 380, 880
    fnt, lines, lead = fit_block(d, text, "SemiBold" if name == "fun" else "Regular",
                                 W - 2 * m - (60 if name == "fun" else 0),
                                 bottom - top, start=64)

    if name == "fun":
        # Size the bubble to the text rather than to the band. A fixed bubble
        # left a third of it empty under a short question, which reads as a
        # rendering fault rather than a style.
        b_top, b_bot = 250, min(H - 150, top + len(lines) * lead + 60)
        d.rounded_rectangle([m - 40, b_top, W - m + 40, b_bot],
                            radius=t["radius"], fill=t["panel"])
        d.polygon([(m + 60, b_bot - 2), (m + 170, b_bot - 2), (m + 80, b_bot + 85)],
                  fill=t["panel"])
    else:
        d.rectangle([0, 0, t["rail"], H], fill=t["accent"])

    wordmark(d, t, m, 62, on_dark=(name == "fun"))
    tracked(d, (tx, 300 if name == "fun" else 290), "STUDENT QUESTION",
            font("Bold", 34), t["accent"], 7)

    if not blank:
        y = top
        for ln in lines:
            d.text((tx, y), ln, font=fnt, fill=t["ink"] if name == "professional"
                   else t["panel_ink"])
            y += lead
    return img


def render_layout(name, t, slide=None, cam=None):
    """Slide left, camera + branding in a right rail. Replaces the corner PiP:
    at 480px in a corner the instructor is a few dozen pixels tall."""
    img = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(img)
    rail_x, rail_w = 1380, W - 1380

    # --- slide, letterboxed into the left region
    sw = rail_x
    sh = int(sw * 9 / 16)
    sy = (H - sh) // 2
    if slide is not None:
        img.paste(slide.resize((sw, sh), Image.LANCZOS), (0, sy))
    else:
        d.rectangle([0, sy, sw, sy + sh], fill=(235, 235, 238))
        d.text((60, sy + 40), "slide", font=font("Bold", 54), fill=(120, 120, 128))

    # --- rail
    if name == "professional":
        # Drawn into its own tile and pasted. Drawing the diagonals straight
        # onto the frame let them run the full 1920 and streak across the slide.
        rail = Image.new("RGB", (rail_w, H))
        rd = ImageDraw.Draw(rail)
        for i in range(rail_w):
            k = i / rail_w
            rd.line([(i, 0), (i, H)],
                    fill=(int(140 + 56 * k), int(11 + 7 * k), int(30 + 18 * k)))
        for k in range(-H, rail_w + H, 26):          # faint engraved lines
            rd.line([(k, 0), (k + H, H)], fill=(168, 22, 52), width=1)
        img.paste(rail, (rail_x, 0))
        d = ImageDraw.Draw(img)
    else:
        d.rectangle([rail_x, 0, W, H], fill=t["bg"])
        d.rectangle([rail_x, 0, rail_x + 10, H], fill=t["accent"])

    # --- camera inset, 16:9
    cm = 30
    cx0, cw = rail_x + cm, rail_w - 2 * cm
    ch = int(cw * 9 / 16)
    cy0 = 60
    if cam is not None:
        c = cam.resize((cw, ch), Image.LANCZOS)
        if name == "fun":
            mask = Image.new("L", (cw, ch), 0)
            ImageDraw.Draw(mask).rounded_rectangle([0, 0, cw, ch], radius=28, fill=255)
            img.paste(c, (cx0, cy0), mask)
        else:
            img.paste(c, (cx0, cy0))
            d.rectangle([cx0 - 3, cy0 - 3, cx0 + cw + 3, cy0 + ch + 3],
                        outline=(255, 255, 255), width=3)
    else:
        d.rounded_rectangle([cx0, cy0, cx0 + cw, cy0 + ch],
                            radius=28 if name == "fun" else 0, fill=(52, 54, 64))
        d.text((cx0 + 24, cy0 + 24), "camera", font=font("Bold", 40),
               fill=(150, 152, 165))

    y = cy0 + ch + 70
    d.text((cx0, y), COURSE_NUM, font=font("ExtraBold", 92),
           fill=(255, 255, 255) if name == "professional" else t["accent"])
    y += 118
    lf = font("SemiBold", 34)
    for ln in wrap(d, LECTURE, lf, cw):
        d.text((cx0, y), ln, font=lf, fill=(255, 255, 255))
        y += 44
    d.text((cx0, y + 6), TERM, font=font("Regular", 32),
           fill=(255, 235, 238) if name == "professional" else t["muted"])

    d.text((cx0, H - 110), "Carnegie Mellon", font=serif(36), fill=(255, 255, 255))
    d.text((cx0, H - 66), "University", font=serif(36), fill=(255, 255, 255))
    return img


# ---------------------------------------------------------------------------
# intro over a supplied clip
# ---------------------------------------------------------------------------
# The fun intro is a supplied motion card (intro-card-fun.mp4). The lecture
# details are composited ON TOP of it rather than drawn from scratch, so the
# clip is the art and this only contributes type. The intro segment therefore
# runs exactly as long as the clip -- the duration comes from the file.
INTRO_CLIP_NAMES = ("intro-card-fun.mp4", "intro-card.mp4", "intro.mov",
                    "intro-card-fun.mov")


def find_intro_clip(theme_dir):
    for n in INTRO_CLIP_NAMES:
        p = os.path.join(theme_dir, n)
        if os.path.exists(p):
            return p
    return None


def intro_overlay(name, t, scrim=True):
    """Transparent 1920x1080 layer of just the lecture details.

    Everything is drawn with alpha so it can sit over moving footage. A soft
    scrim goes behind the text block because the clip underneath is unknown --
    white type over a light frame is unreadable, and the whole point of the
    overlay is that it must survive whatever the card is doing.
    """
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    m = t["margin"]

    if scrim:
        # Vertical gradient, strongest at the bottom where the type sits.
        for y in range(430, H):
            a = int(180 * min(1.0, (y - 430) / 320.0))
            d.line([(0, y), (W, y)], fill=(0, 0, 0, a))

    d.rectangle([0, 0, 33, H], fill=t["panel"] + (255,))
    d.text((m, 100), "Carnegie Mellon University", font=serif(40),
           fill=(255, 255, 255, 255))
    d.text((m, 152), "School of Computer Science", font=font("Light", 38),
           fill=(255, 255, 255, 255))

    fnt, lines, lead = fit_block(d, LECTURE, "ExtraBold", W - 2 * m - 60, 240,
                                 start=104)
    y = 560
    for ln in lines:
        d.text((m, y), ln, font=fnt, fill=(255, 255, 255, 255))
        y += lead
    d.rounded_rectangle([m, y + 14, m + 240, y + 32], radius=9,
                        fill=t["accent"] + (255,))
    y += 46

    sub = f"{COURSE_NUM}: {COURSE_TITLE}"
    sf, slines, slead = fit_block(d, sub, "SemiBold", W - 2 * m, 140, start=46)
    for ln in slines:
        d.text((m, y), ln, font=sf, fill=(235, 236, 243, 255))
        y += slead

    d.text((m, 950), TERM, font=font("Regular", 40), fill=(255, 255, 255, 235))
    return lay


def compose_intro(clip, overlay_png, out_path, fade_in=0.6, hold_from=0.4):
    """Composite the details over the supplied clip, preserving its length.

    One image over one video -- a single overlay input, not the per-question
    looped-image chain that made the old cards.py O(questions x duration).
    """
    vf = (f"[1:v]format=rgba,fade=t=in:st={hold_from}:d={fade_in}:alpha=1[ov];"
          f"[0:v]scale={W}:{H}:force_original_aspect_ratio=decrease,"
          f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1[bg];"
          f"[bg][ov]overlay=0:0:format=auto")
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", clip, "-i", overlay_png,
           "-filter_complex", vf, "-c:v", "libx264", "-preset", "medium",
           "-crf", "18", "-pix_fmt", "yuv420p"]
    # Carry the clip's own audio through if it has any; a motion card often
    # comes with a sting.
    has_audio = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=codec_type", "-of", "csv=p=0", clip],
        capture_output=True, text=True).stdout.strip()
    cmd += ["-c:a", "aac", "-map", "0:v", "-map", "0:a"] if has_audio else ["-an"]
    cmd.append(out_path)
    subprocess.run(cmd, check=True)
    return out_path


def grab(video, t_sec):
    """One frame from a local video, for a sample that shows real content."""
    if not os.path.exists(video):
        return None
    out = "/tmp/_frame.png"
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t_sec), "-i", video,
                        "-frames:v", "1", out, "-y"], check=True, timeout=60)
        return Image.open(out).convert("RGB")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theme", choices=list(THEMES), default=None)
    ap.add_argument("--lecture-dir", default="data/15210-lecture12")
    ap.add_argument("--animate", action="store_true", help="(kept for compat; the overlay always renders)")
    ap.add_argument("--intro-seconds", type=float, default=5.0)
    args = ap.parse_args()

    slide = grab(os.path.join(args.lecture_dir, "screen_sync.mp4"), 2100)
    cam = grab(os.path.join(args.lecture_dir, "camera_muted_anon_tracked.mp4"), 2100)
    print(f"slide frame: {'yes' if slide else 'placeholder'} | "
          f"camera frame: {'yes' if cam else 'placeholder'}")

    for name in ([args.theme] if args.theme else list(THEMES)):
        t = THEMES[name]
        out = os.path.join(ROOT, "assets", "themes", name, "samples")
        os.makedirs(out, exist_ok=True)
        for label, im in (("intro", render_intro(name, t)),
                          ("question", render_question(name, t)),
                          ("layout", render_layout(name, t, slide, cam))):
            p = os.path.join(out, f"{label}.png")
            im.save(p)
            print(f"  {name}/{label}.png  {im.size}")

        # The blank template cards.py consumes. Lives beside the theme, not in
        # samples/, because this one is an input to the pipeline rather than a
        # picture of the output. It replaces the deleted
        # assets/student-question-card-template.png.
        tpl = os.path.join(ROOT, "assets", "themes", name, "question-card.png")
        render_question(name, t, text=None).save(tpl)
        print(f"  {name}/question-card.png  (blank template for cards.py)")

        # The details layer that gets composited over a supplied motion card.
        ov = os.path.join(out, "intro-overlay.png")
        intro_overlay(name, t).save(ov)
        print(f"  {name}/intro-overlay.png  (RGBA, composites over the clip)")

        theme_dir = os.path.join(ROOT, "assets", "themes", name)
        clip = find_intro_clip(theme_dir)
        if clip:
            mp4 = compose_intro(clip, ov, os.path.join(out, "intro.mp4"))
            dur = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", mp4], capture_output=True, text=True).stdout.strip()
            print(f"  {name}/intro.mp4  over {os.path.basename(clip)}, "
                  f"{float(dur):.2f}s (the clip sets the length)")
        else:
            print(f"  {name}: no intro clip yet -- drop one at "
                  f"assets/themes/{name}/intro-card-fun.mp4 and re-run")


if __name__ == "__main__":
    main()
