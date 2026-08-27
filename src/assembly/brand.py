"""The SCS Open Courseware scene contract, as code.

Everything in this module is a transcription of
`assets/brand/plates/README.md`, which is the contract Phillip handed over with
the Brand Assets folder. Nothing here is a design decision of ours: the numbers
are his, and where this file disagrees with that README the README wins.

Two scenes:

    scene_a   slide-led. Slide on the left, professor in the right rail, course
              metadata under it, SCS lockup and the Continue Learning prompt in
              the footer. Everything but the two live windows is the plate.
    scene_b   full-bleed camera, carrying nothing but the 60% unitmark
              watermark. For walking, board work, demonstrations.

The plate is a transparent PNG that goes ON TOP of the live video. The two
Scene A windows are the only transparent parts, so the pipeline's whole job is:
put the slide at one rectangle, the professor at the other, lay the plate over
it, and draw the per-lecture type into the rail.

Why the plate is not just baked into the compositor
---------------------------------------------------
Because then a design change would be a code change. The contract is the two
rectangles; the plate is art. `verify_plate` checks that a plate really does
leave those rectangles transparent, which is what makes "redesign it in Figma,
export 1920x1080 with alpha, drop it in" true rather than aspirational.

Why the per-lecture type is not in the plate
--------------------------------------------
Course code, course title, lecture title and term change per video, so they are
drawn at render time. TEXT_SPEC below is the baseline/font/size/colour table
from the README, verbatim.
"""

import datetime
import itertools
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1920, 1080

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BRAND_DIR = os.path.join(ROOT, "assets", "brand")
PLATE_DIR = os.path.join(BRAND_DIR, "plates")
FONT_DIR = os.path.join(ROOT, "assets", "fonts", "static")
COURSES_JSON = os.path.join(BRAND_DIR, "courses.json")

SCENE_A_PLATE = os.path.join(PLATE_DIR, "scene-a-overlay.png")
SCENE_B_PLATE = os.path.join(PLATE_DIR, "scene-b-overlay.png")

# The field colour behind everything, and Carnegie Red. From the handoff README.
FIELD = (0x0F, 0x0F, 0x10)              # RGB
CARNEGIE_RED = (0xC4, 0x12, 0x30)
# Top of the footer band. The rail's type has to stay above it, and the plate
# builder copies everything below it out of the handoff art unchanged.
FOOTER_Y = 972

# --- THE CONTRACT ---------------------------------------------------------
# "As long as those two rectangles stay transparent and stay at these
# coordinates, the pipeline needs no code change."
#
# There are now two geometries, and `handoff` is still the authority on what
# the contract SAYS. `wide` is a retune of the same two-window structure, made
# because the handoff frame leaves 100px of empty field above the windows and
# another 96 below the slide -- a fifth of the height carrying nothing -- while
# the slide, the only thing a viewer reads, sits at 1380x776. It tightens the
# margins, narrows the rail, and puts every reclaimed pixel into the slide.
#
# What the retune must not break, and why these numbers and not larger ones:
#
#   * The slide window stays EXACTLY 16:9 (1472 = 16x92, 828 = 9x92), so a 16:9
#     deck fills it with no letterbox. Round numbers that are merely close --
#     1470x827 -- put a one-pixel bar of field down one edge.
#   * The speaker window's bottom edge stays at y=676 in both, so TEXT_SPEC's
#     baselines are shared and the rail type did not have to be re-spec'd.
#   * The rail cannot narrow indefinitely, and what stops it is the camera,
#     not the type. layout.crop_size derives the crop from the speaker
#     window's aspect, so a narrower rail filling the same column height is a
#     narrower crop of the same 1280x720 source. Measured against the 89,576
#     tracked frames of lecture 12, the share of frames whose gesture box is
#     wider than the crop -- i.e. where an outstretched hand leaves the rail --
#     goes:
#
#         rail 432 -> crop 467px -> 3.9%      rail 342 -> crop 378px -> 7.6%
#         rail 400 -> crop 432px -> 4.9%      rail 320 -> crop 346px -> 10.1%
#         rail 360 -> crop 389px -> 6.9%      rail 300 -> crop 324px -> 12.2%
#
#     The curve is shallow because the width distribution is long-tailed: the
#     median box is 210px and p90 is 346px, so most of what a wider crop buys
#     is headroom around a rare fully-extended arm. 342 costs about four
#     frames in a hundred against the handoff rail and buys 21% more slide,
#     which is the trade this layout exists to make. Below ~320 the curve
#     steepens and the crop starts cutting into the p75 box, which is his
#     ordinary gesturing rather than a rare one.
#   * The slide is WIDTH-limited here, not height-limited -- the rail floor
#     sets its width and 16:9 sets its height -- so the leftover vertical
#     space is split evenly above and below it (58 and 59) rather than banked
#     at one edge, where it would read as the frame sitting crooked.
# The rail's type, per layout: element -> (baseline y, weight, size px, colour).
#
# `handoff` is the README's table verbatim, two lines for the lecture title.
# `wide` re-stacks the same elements to free a THIRD lecture line, which is
# what a 360px rail needs and is worth more than any font size. Measured over
# the 693 lecture names in the Spring manifests, at 360px:
#
#     2 lines, 22px floor   299 shrank,  78 truncated with an ellipsis
#     3 lines, 22px floor   103 shrank,   5 truncated
#
# A third line does not merely rescue the long titles -- it lets most of them
# stay at the spec's 32px, because the wrapper stops having to shrink to force
# a two-line fit. Dropping the shrink floor instead was tried and is strictly
# worse: it keeps all 299 shrinks and still truncates 14 at 17px type.
_HANDOFF_TEXT = {
    "course_code":  (718, "Bold", 19, (0xC4, 0x12, 0x30)),
    "course_title": [(752, "Regular", 17, (0xA5, 0xA5, 0xAA)),
                     (777, "Regular", 17, (0xA5, 0xA5, 0xAA))],
    "rule":         (800, (0x33, 0x33, 0x36)),
    "lecture":      [(839, "SemiBold", 32, (0xFF, 0xFF, 0xFF)),
                     (879, "SemiBold", 32, (0xFF, 0xFF, 0xFF))],
    "term":         (916, "Regular", 18, (0x8E, 0x8E, 0x93)),
}
# Same elements, same sizes and colours, re-spaced between the speaker
# window's bottom edge (676) and the footer (972) to hold three lecture lines.
# The gaps above and below the block absorb it: 42->32 at the top, 56->38 at
# the bottom, and the lecture leading tightens 40->37.
# The code and the term share a line -- "15-210, Spring 2026" -- rather than
# bracketing the block top and bottom. They are the same KIND of fact, both
# one short string, and giving each its own line spent 39px of rail to say
# what fits comfortably on one. The line it frees goes to the speaker window,
# which is the only element in the rail a viewer actually watches.
# Order, top to bottom: course title, rule, lecture title, then the code and
# term together on the last line. The identifiers sit UNDER the two things a
# viewer is actually reading, which is the order they matter in -- you look at
# a lecture to find out what it covers, not to find out its course number.
#
# The course title is specced at 24px rather than the handoff's 17px because
# 24 is the largest size at which this repo's longest real course title,
# "Parallel and Sequential / Data Structures and Algorithms", still holds two
# 360px lines (351px on the wider of them). It is a ceiling, not a promise:
# fit_lines shrinks per lecture, so a longer title simply comes down from it.
_WIDE_TEXT = {
    "course_title": [(744, "Regular", 24, (0xC8, 0xC8, 0xCE)),
                     (774, "Regular", 24, (0xC8, 0xC8, 0xCE))],
    "rule":         (794, (0x33, 0x33, 0x36)),
    "lecture":      [(830, "SemiBold", 32, (0xFF, 0xFF, 0xFF)),
                     (867, "SemiBold", 32, (0xFF, 0xFF, 0xFF)),
                     (904, "SemiBold", 32, (0xFF, 0xFF, 0xFF))],
    "course_meta":  (936, "Bold", 19, (0xC4, 0x12, 0x30),
                     "Regular", (0x8E, 0x8E, 0x93)),
}

# The 4:3 rail is 722px rather than 342, so the type scales with it. Same
# elements in the same order; the sizes are the largest that keep the lecture
# title to two lines on the corpus's median name at that measure.
_WIDE43_TEXT = {
    "course_title": [(715, "Regular", 28, (0xC8, 0xC8, 0xCE)),
                     (750, "Regular", 28, (0xC8, 0xC8, 0xCE))],
    "rule":         (772, (0x33, 0x33, 0x36)),
    "lecture":      [(812, "SemiBold", 38, (0xFF, 0xFF, 0xFF)),
                     (856, "SemiBold", 38, (0xFF, 0xFF, 0xFF)),
                     (900, "SemiBold", 38, (0xFF, 0xFF, 0xFF))],
    "course_meta":  (932, "Bold", 22, (0xC4, 0x12, 0x30),
                     "Regular", (0x8E, 0x8E, 0x93)),
}

LAYOUTS = {
    "handoff": {
        "slide": (40, 100, 1380, 776),        # x, y, w, h
        "speaker": (1448, 100, 432, 576),
        "rule_x1": 1880,
        "plate": "scene-a-overlay.png",
        "text": _HANDOFF_TEXT,
        "aspect": 16 / 9,
    },
    "wide": {
        "slide": (20, 58, 1520, 855),
        "speaker": (1558, 58, 342, 652),
        "rule_x1": 1900,
        "plate": "scene-a-overlay-wide.png",
        "text": _WIDE_TEXT,
        "aspect": 16 / 9,
    },
    # For a 4:3 deck. "Create and test separate layouts for 4:3 and 16:9
    # slides."
    #
    # The 16:9 layout does not crop a 4:3 deck -- _slide_filter pads it into
    # the field rather than cutting the title off -- so the deck is safe
    # either way. What it does is waste the window: a 4:3 slide fitted into
    # 1520x855 uses 1140x855 of it and leaves 380px of field in two bars, and
    # the slide is no bigger than it would have been in a window built for it.
    #
    # So the window is built for it. The slide is height-limited here rather
    # than width-limited, at the same 855px tall, and the 380px the bars were
    # wasting goes to the rail -- which turns the speaker window from a narrow
    # portrait strip into a 1.16:1 medium shot on an 837px crop. A 4:3 lecture
    # ends up with a SMALLER slide and a much larger instructor than a 16:9
    # one, which is the right trade: there is simply less slide to show.
    "wide43": {
        "slide": (20, 58, 1140, 855),
        "speaker": (1178, 58, 722, 621),
        "rule_x1": 1900,
        "plate": "scene-a-overlay-wide43.png",
        "text": _WIDE43_TEXT,
        "aspect": 4 / 3,
    },
}
DEFAULT_LAYOUT = "wide"

# Set by set_layout() below, which runs at import. Read these through the
# module (`brand.SLIDE_WINDOW`) rather than from-importing them: a
# from-import binds the value at import time and would not follow a
# --layout switch.
SLIDE_WINDOW = SPEAKER_WINDOW = None
WINDOWS = {}
RAIL_X = RAIL_W = RULE_X1 = 0
LAYOUT = TEXT_SPEC = None


def set_layout(name=DEFAULT_LAYOUT):
    """Point the module's geometry at one of LAYOUTS, and check it adds up.

    The arithmetic check is not ceremony. Every one of these numbers has to
    agree with three others -- the margins and the gutter have to sum to 1920,
    the slide has to stay 16:9, the rail has to start where the speaker window
    does -- and a geometry that is out by a few pixels does not fail, it
    renders a frame with a sliver of field down one side that nobody notices
    until it is in a published video.
    """
    global SLIDE_WINDOW, SPEAKER_WINDOW, WINDOWS, RAIL_X, RAIL_W, RULE_X1
    global SCENE_A_PLATE, LAYOUT, TEXT_SPEC
    if name not in LAYOUTS:
        raise SystemExit(f"unknown layout {name!r}; expected one of "
                         f"{sorted(LAYOUTS)}")
    spec = LAYOUTS[name]
    sx, sy, sw, sh = spec["slide"]
    cx, cy, cw, ch = spec["speaker"]
    # Not fatal, because the handoff geometry itself is off by four pixels --
    # 1380x776 is 1.7784:1, not 1.7778:1 -- and that is Phillip's number, not a
    # mistake to reject. It is worth SAYING, though: an inexact window is where
    # the hairline of field down the edge of a 16:9 deck comes from, and `wide`
    # is exact on purpose.
    # Checked against the layout's OWN declared aspect, not against 16:9. A
    # 4:3 layout has a 4:3 window on purpose, and warning that it is not 16:9
    # would be the check crying wolf on the one layout built to be different.
    want = spec.get("aspect", 16 / 9)
    exact = sw / want
    if abs(sh - exact) >= 1.0:
        print(f"[brand] layout {name}: slide window {sw}x{sh} is "
              f"{sw / sh:.4f}:1, not the {want:.4f}:1 it declares "
              f"({sw}x{exact:.1f}) -- a matching deck loses "
              f"{abs(sh - exact):.1f}px to a bar or a crop down one edge.")
    if cx <= sx + sw:
        raise SystemExit(f"layout {name}: speaker window overlaps the slide")
    if cx + cw > W or sy < 0 or cy < 0:
        raise SystemExit(f"layout {name}: window falls outside the frame")
    LAYOUT = name
    SLIDE_WINDOW, SPEAKER_WINDOW = spec["slide"], spec["speaker"]
    WINDOWS = {"slide": SLIDE_WINDOW, "speaker": SPEAKER_WINDOW}
    # The metadata rail: left-aligned at the speaker window's x, and no wider
    # than it. "Check the longest expected lecture title before batch
    # rendering." fit_lines() is that check; a narrower rail simply makes it
    # shrink the type sooner.
    RAIL_X, RAIL_W = cx, cw
    RULE_X1 = spec["rule_x1"]
    SCENE_A_PLATE = os.path.join(PLATE_DIR, spec["plate"])
    TEXT_SPEC = spec["text"]
    _check_rail_stack(name, spec)
    return spec


def _check_rail_stack(name, spec):
    """Every rail baseline must clear the speaker window and the footer.

    The rail's vertical budget is whatever is left between the bottom of the
    speaker window and the top of the footer, and `wide` spends nearly all of
    it to fit a third lecture line. That leaves no room for a later edit to be
    approximately right: a baseline eight pixels low puts the term's
    descenders through the footer's red rule, which looks like a rendering bug
    and is a number in a table.
    """
    cy, ch = spec["speaker"][1], spec["speaker"][3]
    top, bottom = cy + ch, FOOTER_Y
    ys = []
    for key, val in spec["text"].items():
        if key == "rule":
            ys.append((val[0], key))
        elif isinstance(val, list):
            ys.extend((v[0], f"{key}[{i}]") for i, v in enumerate(val))
        else:
            ys.append((val[0], key))
    for y, what in sorted(ys):
        if y <= top:
            raise SystemExit(
                f"layout {name}: rail baseline {what} at y{y} is inside the "
                f"speaker window, which ends at y{top}")
        if y >= bottom:
            raise SystemExit(
                f"layout {name}: rail baseline {what} at y{y} is in the "
                f"footer, which starts at y{bottom}")


set_layout()


# scenes.json speaks pip/full -- the names the cut list has used since before
# the brand assets existed, and what every scenes.json already on disk contains.
# Phillip's package calls the same two things Scene A and Scene B. One table,
# so the correspondence is written down once rather than assumed at four call
# sites.
SCENE_ALIASES = {
    "pip": "scene_a", "scene_a": "scene_a", "a": "scene_a", "slide": "scene_a",
    "full": "scene_b", "scene_b": "scene_b", "b": "scene_b", "camera": "scene_b",
}


def scene_name(raw):
    """Canonical scene id for whatever a cut list calls it."""
    key = str(raw).strip().lower().replace("-", "_")
    if key not in SCENE_ALIASES:
        raise SystemExit(
            f"unknown scene {raw!r} in the cut list; expected one of "
            f"{sorted(set(SCENE_ALIASES))}")
    return SCENE_ALIASES[key]


def font(weight, size):
    p = os.path.join(FONT_DIR, f"OpenSans-{weight}.ttf")
    if not os.path.exists(p):
        raise SystemExit(f"missing font {p}")
    return ImageFont.truetype(p, size)


# ---------------------------------------------------------------------------
# per-lecture metadata
# ---------------------------------------------------------------------------
def term_from_panopto(start):
    """"Spring 2026" from a Panopto SessionStartTime.

    Panopto reports the start as SECONDS SINCE 1601-01-01 -- the Windows
    FILETIME epoch, not Unix. Read as Unix it lands in 2395, and read as .NET
    ticks-in-seconds it lands in year 426; both are wrong quietly, which is why
    this is a named function with the epoch spelled out rather than an
    expression somewhere.
    """
    if not start:
        return None
    try:
        d = datetime.datetime(1601, 1, 1) + datetime.timedelta(seconds=float(start))
    except (OverflowError, ValueError, TypeError):
        return None
    # CMU terms: Spring is Jan-May, Summer Jun-Jul, Fall Aug-Dec.
    season = "Spring" if d.month <= 5 else ("Summer" if d.month <= 7 else "Fall")
    return f"{season} {d.year}"


def course_title(code):
    """Full course title for a course number, from the registry.

    Panopto gives us the course NUMBER and the lecture name, and nothing else --
    there is no full title in the manifest to draw from. So the titles live in
    assets/brand/courses.json, which is a two-line-per-course file someone fills
    in once per course rather than something derived.
    """
    if not code or not os.path.exists(COURSES_JSON):
        return None
    with open(COURSES_JSON) as f:
        reg = json.load(f)
    entry = reg.get(code) or reg.get(code.replace("-", ""))
    if isinstance(entry, dict):
        return entry.get("title")
    return entry


def lecture_meta(metadata_path, term=None, title=None):
    """The four strings the rail draws, resolved from every available source.

    Precedence is explicit-flag, then anything the lecture's own metadata.json
    carries, then the derivations. A lecture can therefore override the
    registry without editing it, which matters for one-offs and guest lectures.
    """
    md = {}
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            md = json.load(f)
    code = md.get("course") or ""
    resolved_title = title or md.get("course_title") or course_title(code)
    resolved_term = term or md.get("term") or term_from_panopto(md.get("start"))
    if not resolved_title:
        print(f"[brand] no course title for {code!r} -- add it to "
              f"{os.path.relpath(COURSES_JSON, ROOT)} or pass --course-title. "
              f"The rail will show the code and lecture only.")
    if not resolved_term:
        print("[brand] no term could be resolved; pass --term")
    return {
        "course_code": code,
        "course_title": resolved_title or "",
        "lecture": md.get("name") or md.get("lecture") or "Lecture",
        "term": resolved_term or "",
    }


# ---------------------------------------------------------------------------
# fitting type into a 432px rail
# ---------------------------------------------------------------------------
def wrap_to(draw, text, fnt, max_w, max_lines):
    """Break `text` into at most max_lines lines that each fit max_w.

    NOT a greedy wrap. Greedy is what a paragraph wants and the wrong thing for
    a two-line title in a narrow rail: it packs the first line to the margin and
    leaves an orphan below it, so "Lecture 12: Binary Search Trees" comes out as
    "Lecture 12: Binary Search" / "Trees" where the proof render breaks it
    "Lecture 12:" / "Binary Search Trees". Three rules, in order:

      1. An explicit newline in the string wins. The course registry and a
         lecture's metadata.json can both carry one, which is the only way to
         get a break that depends on meaning we cannot see.
      2. Break after a colon when both halves fit. Lecture names are overwhelm-
         ingly "Lecture N: Subject", and that colon is the intended break.
      3. Otherwise balance: of all the ways to split the words across the lines,
         take the one whose widest line is narrowest.

    Returns (lines, overran).
    """
    if not text:
        return [], False
    if "\n" in text:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        return lines[:max_lines], len(lines) > max_lines

    words = text.split()
    if not words:
        return [], False

    def widths(lines):
        return [draw.textlength(l, font=fnt) for l in lines]

    def split_at(cuts):
        out, prev = [], 0
        for c in list(cuts) + [len(words)]:
            out.append(" ".join(words[prev:c]))
            prev = c
        return [l for l in out if l]

    def balanced(k):
        """The k-way split whose widest line is narrowest, or None if none fit.

        Brute force over the cut positions. A lecture title is a dozen words
        and k is 2 or 3, so this is a couple of hundred candidates -- cheap
        next to the font metrics it is already paying for, and it does not
        have greedy's failure mode of packing the first line and orphaning the
        last.
        """
        if k > len(words):
            return None
        best = None
        for cuts in itertools.combinations(range(1, len(words)), k - 1):
            cand = split_at(cuts)
            if len(cand) != k:
                continue
            w = max(widths(cand))
            if w <= max_w and (best is None or w < best[0]):
                best = (w, cand)
        return best[1] if best else None

    # rule 2 -- the colon, if it lands anywhere but the very end. Two lines
    # only: "Lecture 12:" / "Binary Search Trees" is the break the proof
    # render uses, and it is worth more than any three-line arrangement.
    #
    # Guarded on max_lines, because a caller asking for ONE line is asking for
    # one: without this the rule happily returned a two-line split and the
    # caller drew lines[0], so the end card said "Lecture 12:" and dropped
    # "Binary Search Trees" with no warning of any kind.
    for i, word in enumerate(words[:-1] if max_lines >= 2 else []):
        if word.endswith(":"):
            cand = split_at([i + 1])
            if len(cand) == 2 and max(widths(cand)) <= max_w:
                return cand, False

    # rule 3 -- balance, over as few lines as will hold it. Fewer lines first,
    # so a title that fits on one never spreads onto two, and one that fits on
    # two never spreads onto three, just because the line is available.
    for k in range(1, max_lines + 1):
        cand = balanced(k)
        if cand:
            return cand, False

    if len(words) == 1:
        return words, draw.textlength(words[0], font=fnt) > max_w
    # Nothing fits. Hand back an even split at the full line count so the
    # caller can shrink the type and try again; if it runs out of sizes it
    # truncates this.
    n = max(1, min(max_lines, len(words)))
    step = len(words) / n
    return split_at([round(step * i) for i in range(1, n)]), True


def fit_lines(draw, text, weight, size, max_w, max_lines, min_size=None):
    """Wrap `text` into at most max_lines, shrinking the type if it will not go.

    "Check the longest expected lecture title before batch rendering. The rail
    is only 432 px wide." This is that check, at render time and per lecture,
    because the alternative -- discovering it in the finished video -- is a
    re-encode. Shrinking stops at min_size and then truncates with an ellipsis,
    so a pathological title degrades visibly rather than running off the frame.
    """
    min_size = min_size or max(11, int(size * 0.7))
    for sz in range(size, min_size - 1, -1):
        fnt = font(weight, sz)
        lines, overran = wrap_to(draw, text, fnt, max_w, max_lines)
        if not overran and all(draw.textlength(l, font=fnt) <= max_w for l in lines):
            if sz != size:
                print(f"[brand] {text!r} needed {sz}px (spec says {size}px) to "
                      f"fit {max_lines} line(s) in the {max_w}px rail")
            return lines, sz
    fnt = font(weight, min_size)
    lines, _ = wrap_to(draw, text, fnt, max_w, max_lines)
    if lines:
        while lines[-1] and draw.textlength(lines[-1] + "…", font=fnt) > max_w:
            lines[-1] = lines[-1][:-1]
        lines[-1] = lines[-1].rstrip() + "…"
    print(f"[brand] WARNING: {text!r} does not fit {max_lines} line(s) in the "
          f"{max_w}px rail even at {min_size}px -- truncated. Shorten it in "
          f"metadata.json, or ask for a taller rail.")
    return lines, min_size


# ---------------------------------------------------------------------------
# the plate
# ---------------------------------------------------------------------------
def verify_plate(rgba, path, windows=None, edge=2):
    """Check a plate really leaves the contract's windows transparent.

    The windows are antialiased -- the fully-transparent core measured off the
    supplied plate is inset one pixel from the stated rectangle, and the corners
    are rounded -- so this tests the interior rather than the exact border:
    every pixel `edge` px inside each window must be fully transparent, and the
    plate must be opaque just outside it. That is enough to catch a window that
    moved, resized, or was flattened away, and loose enough that a redesign with
    different corner rounding still passes.
    """
    # Resolved here, not as a default argument. A default binds once, at def
    # time, so `windows=WINDOWS` captured whichever geometry was active at
    # import and then checked every later plate against it -- --layout handoff
    # loaded the handoff plate and verified it against the wide rectangles.
    windows = WINDOWS if windows is None else windows
    alpha = rgba[..., 3]
    problems = []
    for name, (x, y, w, h) in windows.items():
        inner = alpha[y + edge:y + h - edge, x + edge:x + w - edge]
        # Corners: a rounded window's corner pixels are opaque plate, so sample
        # a cross through the middle instead of the full inner rectangle.
        band_h = inner[inner.shape[0] // 2 - 4:inner.shape[0] // 2 + 4, :]
        band_v = inner[:, inner.shape[1] // 2 - 4:inner.shape[1] // 2 + 4]
        for band, what in ((band_h, "horizontal"), (band_v, "vertical")):
            if band.size and band.max() != 0:
                problems.append(
                    f"{name} window at x{x} y{y} {w}x{h} is not transparent "
                    f"along its {what} centre line (max alpha {band.max()})")
        ring = []
        if y - edge - 1 >= 0:
            ring.append(alpha[y - edge - 1, x + w // 2])
        if y + h + edge < H:
            ring.append(alpha[y + h + edge, x + w // 2])
        if x - edge - 1 >= 0:
            ring.append(alpha[y + h // 2, x - edge - 1])
        if x + w + edge < W:
            ring.append(alpha[y + h // 2, x + w + edge])
        if ring and min(ring) < 250:
            problems.append(
                f"{name} window at x{x} y{y} {w}x{h} is transparent OUTSIDE its "
                f"stated edge (min alpha {min(ring)}) -- the window has moved "
                f"or grown")
    if problems:
        raise SystemExit(
            f"{os.path.basename(path)} breaks the plate contract:\n  "
            + "\n  ".join(problems)
            + f"\n\nThe contract is in {os.path.relpath(PLATE_DIR, ROOT)}/"
              f"README.md, as retuned by the {LAYOUT!r} layout in brand.py: "
              f"slide at x{windows['slide'][0]} y{windows['slide'][1]} "
              f"{windows['slide'][2]}x{windows['slide'][3]}, speaker at "
              f"x{windows['speaker'][0]} y{windows['speaker'][1]} "
              f"{windows['speaker'][2]}x{windows['speaker'][3]}, transparent, "
              f"on a {W}x{H} RGBA export. Rebuild it with "
              f"`python -m src.assembly.brand --build-plate`.")


def check_footer_unitmark(rgba, path, box=(40, 985, 420, 1065)):
    """Confirm the footer lockup actually rendered.

    Phillip's warning, verbatim: "rsvg-convert silently drops the linked
    unitmark if you rebuild the PNGs from the SVGs, so confirm the footer logo
    is actually there rather than trusting the render." A dropped <image> leaves
    the footer a flat field colour, which looks like a deliberately clean footer
    and is instead a missing university mark on a published video. So: count
    pixels in the footer-left box that differ from the field colour. There is no
    threshold to tune -- a real lockup is thousands of white pixels, a dropped
    one is zero.
    """
    x0, y0, x1, y1 = box
    region = rgba[y0:y1, x0:x1, :3].astype(np.int16)
    ink = int((np.abs(region - np.array(FIELD, dtype=np.int16)).max(axis=2) > 40).sum())
    if ink < 500:
        print(f"[brand] WARNING: {os.path.basename(path)} has only {ink} non-field "
              f"pixels where the footer unitmark should be. If this plate was "
              f"rebuilt from the SVG with rsvg-convert, the linked unitmark was "
              f"dropped -- re-composite it before mastering.")
        return False
    return True


class Plate:
    """One scene's overlay, pre-composited for the render loop.

    Held as BGR plus a float alpha because the compositor works in OpenCV
    order, and because the per-lecture type is drawn ONCE here rather than per
    frame -- the rail text is constant for the whole lecture, so a 90-minute
    encode should pay for it once.

    `ink_box` is the bounding box of everything the plate actually draws.
    Scene B is 99.7% transparent, so blending the full frame for a watermark in
    one corner would be 2 megapixels of float maths per frame for 20,000 pixels
    of watermark.
    """

    # `windows` defaults to the sentinel, not to WINDOWS, for the reason
    # verify_plate spells out: a default argument binds once at def time, and
    # the active geometry changes with --layout. Scene B passes windows={} to
    # say "this plate has no live windows", which is why the sentinel is a
    # distinct object rather than None.
    _ACTIVE = object()

    def __init__(self, path, meta=None, verify=True, windows=_ACTIVE):
        if windows is Plate._ACTIVE:
            windows = WINDOWS
        if not os.path.exists(path):
            raise SystemExit(
                f"missing plate {path}. The brand assets live in "
                f"{os.path.relpath(BRAND_DIR, ROOT)}; see its README.")
        img = Image.open(path).convert("RGBA")
        if img.size != (W, H):
            raise SystemExit(f"{os.path.basename(path)} is {img.size}, "
                             f"expected {(W, H)} -- export 1920x1080 with alpha")
        self.path = path
        self.windows = windows

        arr = np.array(img)
        if verify and windows:
            verify_plate(arr, path, windows)
        if windows:                     # only Scene A has a footer
            check_footer_unitmark(arr, path)

        if meta:
            img = self._draw_meta(img, meta)
            arr = np.array(img)

        self.bgr = arr[..., :3][:, :, ::-1].copy()
        self.alpha = (arr[..., 3].astype(np.float32) / 255.0)[..., None]
        ys, xs = np.where(arr[..., 3] > 0)
        self.ink_box = ((int(xs.min()), int(ys.min()),
                         int(xs.max()) + 1, int(ys.max()) + 1)
                        if len(xs) else (0, 0, 0, 0))

    # --- the per-lecture rail ---------------------------------------------
    def _draw_meta(self, img, meta):
        """Course code, course title, rule, lecture title, term.

        Drawn on the BASELINES in TEXT_SPEC (PIL's anchor="ls"), not on a
        top-left corner, because that is how the spec states them and how the
        proof render was produced. Getting that wrong shifts every line down by
        roughly its ascender -- about 25px on the lecture title -- which reads
        as "close enough" next to the proof and is not.
        """
        d = ImageDraw.Draw(img)
        max_w = RULE_X1 - RAIL_X        # the rule's own width; the rail's usable one

        code = meta.get("course_code")
        term = meta.get("term")
        if "course_code" in TEXT_SPEC and code:
            y, wt, sz, col = TEXT_SPEC["course_code"]
            d.text((RAIL_X, y), code, font=font(wt, sz), fill=col, anchor="ls")

        spec = TEXT_SPEC["course_title"]
        lines, sz = fit_lines(d, meta.get("course_title", ""), spec[0][1],
                              spec[0][2], max_w, len(spec))
        for (y, wt, _, col), text in zip(spec, lines):
            d.text((RAIL_X, y), text, font=font(wt, sz), fill=col, anchor="ls")

        y, col = TEXT_SPEC["rule"]
        d.line([(RAIL_X, y), (RULE_X1, y)], fill=col, width=1)

        spec = TEXT_SPEC["lecture"]
        lines, sz = fit_lines(d, meta.get("lecture", ""), spec[0][1],
                              spec[0][2], max_w, len(spec))
        for (y, wt, _, col), text in zip(spec, lines):
            d.text((RAIL_X, y), text, font=font(wt, sz), fill=col, anchor="ls")

        if "course_meta" in TEXT_SPEC and (code or term):
            # One line: the code in Carnegie Red, the term after it in the
            # secondary grey. Drawn as two runs on a shared baseline rather
            # than one string, because the colour change is the whole point --
            # "15-210" is the identifier and "Spring 2026" is a qualifier, and
            # flattening both to one colour loses that.
            #
            # Drawn AFTER the lecture title, and following it: the lecture
            # block reserves three lines and most titles use two, so a fixed
            # baseline here leaves an empty line's worth of gap above this one
            # on the common case. It keeps its spec gap from whichever line
            # was actually the last one drawn.
            y, wt, sz, col, term_wt, term_col = TEXT_SPEC["course_meta"]
            if lines:
                y = spec[len(lines) - 1][0] + (y - spec[-1][0])
            x = RAIL_X
            if code:
                f = font(wt, sz)
                d.text((x, y), code, font=f, fill=col, anchor="ls")
                x += d.textlength(code, font=f)
            if term:
                f = font(term_wt, sz)
                sep = ", " if code else ""
                d.text((x, y), f"{sep}{term}", font=f, fill=term_col,
                       anchor="ls")
            term = None                 # drawn; do not draw it again below

        if term and "term" in TEXT_SPEC:
            y, wt, sz, col = TEXT_SPEC["term"]
            # The lecture block reserves as many lines as the layout allows,
            # but most titles do not use them all. Pinning the term to a fixed
            # baseline then leaves it floating a whole empty line below a
            # two-line title -- which is what reserving a third line for the
            # long ones would otherwise cost every short one. So the term
            # follows the last line actually drawn, keeping its spec gap.
            if LAYOUTS[LAYOUT].get("reflow_term") and lines:
                gap = y - spec[-1][0]
                y = spec[len(lines) - 1][0] + gap
            d.text((RAIL_X, y), term, font=font(wt, sz), fill=col, anchor="ls")
        return img

    # --- compositing -------------------------------------------------------
    def fill(self, canvas):
        """Copy the plate wholesale. For Scene A, whose plate is opaque
        everywhere except the two windows -- a memcpy is cheaper than a blend,
        and the windows are overwritten by video immediately after."""
        canvas[:, :] = self.bgr

    def legibility(self, canvas):
        """How well this plate's ink reads against what is under it, 0..1.

        Scene B's only mark is a 60%-white unitmark in the top right, and the
        top right of this camera framing is the projection screen -- a white
        rectangle. Measured on lecture 12 the watermark changes those pixels by
        at most 42/255 against a background whose mean is 249: it is applied,
        and it is invisible. A watermark nobody can see is not a watermark, and
        nothing about a successful encode would have said so.

        Reported rather than corrected: the mark's presentation is Phillip's
        design decision, not ours. What the pipeline owes him is the number.
        """
        x0, y0, x1, y1 = self.ink_box
        if x1 <= x0 or y1 <= y0:
            return 1.0
        a = self.alpha[y0:y1, x0:x1, 0]
        ink = a > 0.05
        if not ink.any():
            return 1.0
        under = canvas[y0:y1, x0:x1].astype(np.float32).mean(axis=2)[ink]
        mark = self.bgr[y0:y1, x0:x1].astype(np.float32).mean(axis=2)[ink]
        # Weber-ish: the contrast that actually survives the alpha blend.
        delta = np.abs(mark - under) * a[ink]
        return float(delta.mean() / 255.0)

    def blend(self, canvas, box=None):
        """Alpha-composite the plate over `canvas`, in place, within `box`."""
        x0, y0, x1, y1 = box or self.ink_box
        if x1 <= x0 or y1 <= y0:
            return
        a = self.alpha[y0:y1, x0:x1]
        reg = canvas[y0:y1, x0:x1].astype(np.float32)
        canvas[y0:y1, x0:x1] = (
            reg * (1.0 - a) + self.bgr[y0:y1, x0:x1] * a).astype(np.uint8)


# ---------------------------------------------------------------------------
# building a plate for a retuned geometry
# ---------------------------------------------------------------------------
# The handoff plate is art: Phillip's SVG, exported to PNG. Moving a window
# means the art has to move with it, and the obvious route -- edit the SVG,
# re-export -- is not available here for two reasons worth writing down rather
# than rediscovering. There is no SVG renderer on this machine (no
# rsvg-convert, no inkscape, no cairosvg), and scene-a-overlay.svg links its
# unitmark by ABSOLUTE PATH into Phillip's home directory, so an export
# anywhere else would silently drop the logo. check_footer_unitmark exists
# because that has already happened once.
#
# So the footer is not redrawn at all. FOOTER_Y down is copied out of the
# handoff plate verbatim -- unitmark, bell, rules, type, at their exact
# pixels -- and only the field and the two windows above it are rebuilt. That
# keeps the part of the plate that is genuinely art untouched, and rebuilds
# only the part that is geometry.
WINDOW_RADIUS = 12
SLIDE_STROKE = (0xD7, 0xD7, 0xDA, int(round(255 * 0.55)))
SPEAKER_STROKE = (0xD7, 0xD7, 0xDA, int(round(255 * 0.42)))

# The glow around each live window. "Adjust the bounding box, background
# colour and glow treatment to create a SUBTLER transition between slides and
# the surrounding video" -- the 1px hairline the handoff plate uses is a hard
# edge between a bright slide and a near-black field, and a hard edge is
# exactly the thing that reads as a pasted-on rectangle.
#
# A halo, not an outline: the mask is the window blurred outward, with the
# window itself cut back out, so brightness falls off over GLOW_RADIUS px
# instead of stopping dead. The colour is the same neutral as the hairline, so
# the two read as one treatment rather than two.
#
# Tuned to be SEEN. The first pass spread 0.18 alpha over a 22px blur, which
# lifted the field from 15 to 32 at the edge -- present in the pixels and
# invisible on a screen. Concentrating the same idea into a 10px falloff at
# 0.55 puts a band about 4px wide well clear of the field before it decays,
# which is what reads as a glow rather than as a slightly lighter background.
GLOW_COLOUR = (0xD7, 0xD7, 0xDA)
GLOW_RADIUS = 6
GLOW_ALPHA = 0.55


# How far to lift the footer's ink clear of a player's control bar.
#
# Measured on the built plate: the footer band is y972..1080 and its ink --
# lockup, rule, bell, "CONTINUE LEARNING" -- runs y972..1056, i.e. to within
# 24px of the bottom edge. Against the control bars that actually sit over it:
#
#     YouTube, controls shown        covers y>1032   24px of ink hidden
#     YouTube, controls + hover      covers y>1012   44px hidden
#     Vimeo                          covers y>1028   28px hidden
#     Panopto                        covers y>1036   20px hidden
#
# At REST none of them cover anything -- YouTube hides its controls after a
# few seconds -- so this is not a permanent loss, and that is why the default
# is 0 rather than a lift somebody would have to notice and undo. It is also
# why the answer is Phillip's and not ours: whether the mark should survive a
# hover is a brand decision, and what this owes him is the number.
#
# A lift extends the band UPWARD and moves the ink with it, so the band still
# reaches the bottom edge; it does not leave a stripe of field under a
# floating bar. 24 clears a resting YouTube control bar, 44 clears a hover.
FOOTER_LIFT = 0


def build_plate(name=None, source=None, out=None, footer_lift=FOOTER_LIFT):
    """Render the Scene A plate for a layout, reusing the handoff footer.

    Returns the path written. The result is checked by the same verify_plate
    the renderer runs, so a plate that does not match its own geometry fails
    here rather than in a finished video.
    """
    name = name or LAYOUT
    spec = LAYOUTS[name]
    source = source or os.path.join(PLATE_DIR, LAYOUTS["handoff"]["plate"])
    out = out or os.path.join(PLATE_DIR, spec["plate"])
    if not os.path.exists(source):
        raise SystemExit(f"no source plate to lift the footer from: {source}")

    src = Image.open(source).convert("RGBA")
    if src.size != (W, H):
        raise SystemExit(f"{source} is {src.size}, expected {(W, H)}")

    plate = Image.new("RGBA", (W, H), FIELD + (255,))
    strip = src.crop((0, FOOTER_Y, W, H))
    if footer_lift:
        # Extend the band upward by `footer_lift` and move the ink with it.
        # The extension is filled from the strip's own top row, so it is the
        # footer's background colour rather than a guess at it.
        top = strip.crop((0, 0, W, 1)).resize((W, footer_lift))
        plate.paste(top, (0, FOOTER_Y - footer_lift))
    plate.paste(strip, (0, FOOTER_Y - footer_lift))

    # The window shapes, as one mask. Used twice: blurred for the glow, hard
    # for the holes.
    shape = Image.new("L", (W, H), 0)
    sd = ImageDraw.Draw(shape)
    for x, y, w, h in (spec["slide"], spec["speaker"]):
        sd.rounded_rectangle([x, y, x + w - 1, y + h - 1],
                             radius=WINDOW_RADIUS, fill=255)

    # The glow, before the holes are punched -- it is plate pixels lying
    # OUTSIDE each window, so it has to be painted while there is still plate
    # there to paint on. Cutting the hard shape back out is what makes it a
    # halo rather than a wash over the whole window.
    if GLOW_ALPHA > 0:
        halo = shape.filter(ImageFilter.GaussianBlur(GLOW_RADIUS))
        halo = Image.composite(Image.new("L", (W, H), 0), halo, shape)
        halo = halo.point(lambda v: int(v * GLOW_ALPHA))
        plate.paste(Image.new("RGBA", (W, H), GLOW_COLOUR + (255,)),
                    (0, 0), halo)

    # The windows are holes, not shapes: punch the alpha to zero rather than
    # drawing anything, so whatever the compositor puts underneath shows
    # through unmodified.
    hole = Image.new("L", (W, H), 255)
    hd = ImageDraw.Draw(hole)
    for x, y, w, h in (spec["slide"], spec["speaker"]):
        hd.rounded_rectangle([x, y, x + w - 1, y + h - 1],
                             radius=WINDOW_RADIUS, fill=0)
    plate.putalpha(Image.composite(plate.getchannel("A"),
                                   Image.new("L", (W, H), 0), hole))

    # The 1px window borders, drawn on the half-pixel like the SVG does so the
    # stroke lands on the plate side of the hole instead of straddling it.
    bd = ImageDraw.Draw(plate)
    for (x, y, w, h), stroke in ((spec["slide"], SLIDE_STROKE),
                                 (spec["speaker"], SPEAKER_STROKE)):
        bd.rounded_rectangle([x, y, x + w - 1, y + h - 1],
                             radius=WINDOW_RADIUS, outline=stroke, width=1)

    arr = np.array(plate)
    verify_plate(arr, out, windows={"slide": spec["slide"],
                                    "speaker": spec["speaker"]})
    check_footer_unitmark(arr, out)
    plate.save(out)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--layout", default=DEFAULT_LAYOUT, choices=sorted(LAYOUTS))
    ap.add_argument("--build-plate", action="store_true",
                    help="Render the Scene A plate for --layout and verify it")
    ap.add_argument("--footer-lift", type=int, default=FOOTER_LIFT,
                    help="Raise the footer's ink this many pixels so a "
                         "player's control bar does not cover it. 24 clears a "
                         "resting YouTube bar, 44 clears a hover. Default 0 "
                         "-- see FOOTER_LIFT for the measurements.")
    args = ap.parse_args()
    spec = set_layout(args.layout)
    sx, sy, sw, sh = spec["slide"]
    cx, cy, cw, ch = spec["speaker"]
    print(f"layout   : {args.layout}")
    print(f"slide    : x{sx} y{sy} {sw}x{sh}  ({sw * sh / 1e6:.3f} Mpx, "
          f"{sw / sh:.4f}:1)")
    print(f"speaker  : x{cx} y{cy} {cw}x{ch}  ({cw / ch:.4f}:1, camera "
          f"crop {int(720 * cw / ch)}px wide of 1280)")
    print(f"rail     : x{RAIL_X} w{RULE_X1 - RAIL_X}")
    print(f"margins  : left {sx}, gutter {cx - (sx + sw)}, right "
          f"{W - (cx + cw)}, top {sy}, slide-to-footer "
          f"{FOOTER_Y - (sy + sh)}")
    if args.build_plate:
        print(f"wrote    : {build_plate(args.layout, footer_lift=args.footer_lift)}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# the closing card
# ---------------------------------------------------------------------------
# "A simple closing card for demos, including a thank-you message and
# subscription reminder."
#
# Built from the plate rather than designed separately. The footer band --
# lockup, red rule, bell, CONTINUE LEARNING, Subscribe for new lectures -- is
# lifted out of the Scene A art unchanged, so the card ends the video with the
# exact same furniture the video carried throughout, and the subscription
# reminder Phillip asked for is already in it. Everything above the footer is
# the brand field with three lines of type on it.
#
# Which also answers "consistent branding across scenes" for the one place it
# would otherwise be easiest to get wrong: the card cannot drift from the
# plate, because it IS the plate's footer.
END_CARD_TEXT = {
    "thanks":  (470, "SemiBold", 64, (0xFF, 0xFF, 0xFF)),
    "rule_y":  520,
    "lecture": (582, "Regular", 30, (0xC8, 0xC8, 0xCE)),
    "course":  (628, "Bold", 24, (0xC4, 0x12, 0x30),
                "Regular", (0x8E, 0x8E, 0x93)),
}
END_CARD_MESSAGE = "Thank you for watching"
END_CARD_RULE_W = 220


def build_end_card(meta, out, plate=None, message=END_CARD_MESSAGE):
    """Render the closing card for one lecture. Returns the path written."""
    plate = plate or SCENE_A_PLATE
    src = Image.open(plate).convert("RGBA")
    if src.size != (W, H):
        raise SystemExit(f"{plate} is {src.size}, expected {(W, H)}")

    card = Image.new("RGBA", (W, H), FIELD + (255,))
    card.paste(src.crop((0, FOOTER_Y, W, H)), (0, FOOTER_Y))
    d = ImageDraw.Draw(card)
    cx = W // 2

    y, wt, sz, col = END_CARD_TEXT["thanks"]
    d.text((cx, y), message, font=font(wt, sz), fill=col, anchor="ms")

    ry = END_CARD_TEXT["rule_y"]
    d.line([(cx - END_CARD_RULE_W // 2, ry), (cx + END_CARD_RULE_W // 2, ry)],
           fill=CARNEGIE_RED, width=3)

    lecture = (meta or {}).get("lecture") or ""
    if lecture:
        y, wt, sz, col = END_CARD_TEXT["lecture"]
        # Centred, so it is fitted against a generous width rather than the
        # rail's -- this card has the whole frame and no reason to shrink.
        lines, sz2 = fit_lines(d, lecture, wt, sz, int(W * 0.7), 2)
        for n, text in enumerate(lines):
            d.text((cx, y + n * int(sz2 * 1.35)), text,
                   font=font(wt, sz2), fill=col, anchor="ms")
        if len(lines) > 1:                # the course line moves down with it
            END_CARD_SHIFT = int(sz2 * 1.35)
        else:
            END_CARD_SHIFT = 0

    else:
        END_CARD_SHIFT = 0

    code = (meta or {}).get("course_code") or ""
    term = (meta or {}).get("term") or ""
    if code or term:
        y, wt, sz, col, term_wt, term_col = END_CARD_TEXT["course"]
        y += END_CARD_SHIFT
        f, tf = font(wt, sz), font(term_wt, sz)
        sep = ", " if code and term else ""
        total = (d.textlength(code, font=f)
                 + d.textlength(f"{sep}{term}", font=tf))
        x = cx - total / 2
        if code:
            d.text((x, y), code, font=f, fill=col, anchor="ls")
            x += d.textlength(code, font=f)
        if term:
            d.text((x, y), f"{sep}{term}", font=tf, fill=term_col, anchor="ls")

    card.save(out)
    return out


# ---------------------------------------------------------------------------
# the adaptive footer
# ---------------------------------------------------------------------------
# "Retain consistent branding across instructor and slide views by moving the
# watermark and subscription messaging into an adaptive footer, using
# contrasting colours and transparency appropriate to the underlying
# classroom or slide background."
#
# Scene A had a footer and Scene B had a 60%-white unitmark in the top right.
# Two different treatments for the same job, and Plate.legibility already
# reported the Scene B one as invisible: the top right of this camera framing
# is the projection screen, and the mark moved those pixels by at most 42/255
# against a background whose mean was 249.
#
# So Scene B gets the SAME footer strip, lifted from the Scene A plate rather
# than drawn again, over a scrim whose opacity is set from what is actually
# under it. Constant-alpha does not work here: the strip has to sit on a dark
# lecture hall in one shot and a lit whiteboard in the next.
FOOTER_SCRIM = (0x09, 0x09, 0x0A)       # the footer band's own colour
# Scrim opacity at a black background and at a white one. Never zero: the
# footer's own type is light, and over mid-grey with no scrim at all it has
# nothing to sit against.
FOOTER_SCRIM_MIN = 0.35
FOOTER_SCRIM_MAX = 0.92
# Luma at which the scrim reaches FOOTER_SCRIM_MAX. Below the strip is already
# dark enough for light type; above it the scrim carries the contrast.
FOOTER_SCRIM_FULL_AT = 150.0


def footer_strip(plate=None):
    """The footer band of a Scene A plate as (bgr, alpha) float arrays.

    Returned split and BGR because that is what the compositor wants; it is
    the one place in this module that speaks OpenCV's channel order.
    """
    plate = plate or SCENE_A_PLATE
    arr = np.array(Image.open(plate).convert("RGBA"))[FOOTER_Y:]
    bgr = arr[..., 2::-1].astype(np.float32)
    alpha = (arr[..., 3].astype(np.float32) / 255.0)[..., None]
    return bgr, alpha


def scrim_alpha(under_luma, lo=FOOTER_SCRIM_MIN, hi=FOOTER_SCRIM_MAX,
                full_at=FOOTER_SCRIM_FULL_AT):
    """How opaque the scrim must be for light type to read over `under_luma`."""
    t = min(1.0, max(0.0, float(under_luma) / full_at))
    return lo + (hi - lo) * t


# The opening card. Same construction as the closing one -- plate footer
# lifted verbatim, type centred above it -- so a lecture opens and closes on
# the same furniture and neither can drift from the other.
TITLE_CARD_TEXT = {
    "course":  (404, "Bold", 26, (0xC4, 0x12, 0x30),
                "Regular", (0x8E, 0x8E, 0x93)),
    "title":   [(492, "SemiBold", 56, (0xFF, 0xFF, 0xFF)),
                (556, "SemiBold", 56, (0xFF, 0xFF, 0xFF))],
    "rule_y":  606,
    "course_title": (652, "Regular", 26, (0xC8, 0xC8, 0xCE)),
}
TITLE_CARD_RULE_W = 220


def build_title_card(meta, out, plate=None):
    """Render the opening card for one lecture. Returns the path written."""
    plate = plate or SCENE_A_PLATE
    src = Image.open(plate).convert("RGBA")
    card = Image.new("RGBA", (W, H), FIELD + (255,))
    card.paste(src.crop((0, FOOTER_Y, W, H)), (0, FOOTER_Y))
    d = ImageDraw.Draw(card)
    cx = W // 2
    meta = meta or {}

    code, term = meta.get("course_code") or "", meta.get("term") or ""
    if code or term:
        y, wt, sz, col, term_wt, term_col = TITLE_CARD_TEXT["course"]
        f, tf = font(wt, sz), font(term_wt, sz)
        sep = ", " if code and term else ""
        x = cx - (d.textlength(code, font=f)
                  + d.textlength(f"{sep}{term}", font=tf)) / 2
        if code:
            d.text((x, y), code, font=f, fill=col, anchor="ls")
            x += d.textlength(code, font=f)
        if term:
            d.text((x, y), f"{sep}{term}", font=tf, fill=term_col, anchor="ls")

    spec = TITLE_CARD_TEXT["title"]
    lines, sz = fit_lines(d, meta.get("lecture", ""), spec[0][1], spec[0][2],
                          int(W * 0.78), len(spec))
    for (y, wt, _, col), text in zip(spec, lines):
        d.text((cx, y), text, font=font(wt, sz), fill=col, anchor="ms")

    ry = TITLE_CARD_TEXT["rule_y"]
    if len(lines) < 2:                  # pull the rule up under a one-liner
        ry -= spec[1][0] - spec[0][0]
    d.line([(cx - TITLE_CARD_RULE_W // 2, ry),
            (cx + TITLE_CARD_RULE_W // 2, ry)], fill=CARNEGIE_RED, width=3)

    ct = meta.get("course_title") or ""
    if ct:
        y, wt, sz, col = TITLE_CARD_TEXT["course_title"]
        if len(lines) < 2:
            y -= spec[1][0] - spec[0][0]
        for n, line in enumerate(ct.split("\n")):
            d.text((cx, y + n * int(sz * 1.35)), line.strip(),
                   font=font(wt, sz), fill=col, anchor="ms")

    card.save(out)
    return out
