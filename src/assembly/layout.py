"""Render the two brand scenes onto a lecture: Scene A slide-led, Scene B the room.

    python -m src.assembly.layout --lecture-dir data/15210-lecture12 \
        --start 4080 --duration 300 --out sample.mp4

Consumes `scenes.json` (src/video/scenes.py) and composites, per frame, one of

    scene_a   slide on the left, professor in the right rail, course metadata
              under it, SCS lockup and the Continue Learning prompt in the
              footer. Everything but those two windows is the brand plate.
    scene_b   full-bleed camera carrying nothing but the 60% unitmark
              watermark -- walking, board work, demonstrations.

The two are deliberately different scenes rather than one layout with the big
source swapped. Scene B drops the whole plate furniture: that furniture exists
to label a composited frame, and a cut to the room is not a composited frame --
it is the room.

Where the design lives
----------------------
Not here. `src/assembly/brand.py` holds the contract from
assets/brand/plates/README.md, and the plate PNGs hold the art. This module
places two rectangles of video and lays a plate over them; the windows are
x40 y100 1380x776 for the slide and x1448 y100 432x576 for the professor, and
they do not move. Redesigning the plate is an export, not a code change --
brand.verify_plate is what makes that promise checkable.

Why the compositing is in Python rather than one big filter_complex
-------------------------------------------------------------------
The plates README gives a working ffmpeg one-liner, and it is the right thing
for a still or a fixed crop. It cannot do this one: the professor's crop pans --
he walks -- so the crop rectangle is a per-frame path, and ffmpeg's `crop`
cannot take an arbitrary one without an unusable expression. track_instructor.py
already solved this the same way: decode, crop in numpy, pipe raw frames back
out to a single libx264 encode. Everything stays on the CPU, which is the point;
there is no GPU encode path on PSC anyway.

Where the instructor is
-----------------------
From half a second of motion, not a detector and not a background. He is the
only thing in a fixed lecture-hall shot that moves; the per-pixel range over a
short window is him and nothing else. A few milliseconds per frame on the CPU,
no model, no GPU, no insightface -- so unlike track_instructor this runs
anywhere, including a laptop. See Tracker for why the two obvious alternatives
(median background, multi-second differencing) both fail on this footage.

Privacy
-------
Only the anonymized camera is ever opened -- never camera.mp4 or
camera_muted.mp4 -- so every student face in either scene has already been
pixelated by face_anon. Scene B is the wide shot by choice, which does put more
of the room on screen, at a larger size, than the rail crop does; that rests
entirely on face_anon having run, which is why the refusal in main() is not
optional.
"""

import argparse
import json
import os
import re
import subprocess
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from src.assembly import brand
from src.assembly.brand import FIELD, Plate, scene_name

# The two window rectangles are deliberately NOT from-imported. They are
# rebound by brand.set_layout(), and a from-import would capture whichever
# geometry happened to be active at import time -- so --layout would change the
# plate and not the windows, compositing the new art over the old rectangles.

from src.paths import LecturePaths, add_lecture_args
from src.sync import get_duration

W, H, FPS = brand.W, brand.H, 25
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Pan smoothing width, in FRAMES. 15 is 0.6s either side at 25fps. Measured on
# lecture 12: 90% of frames sit dead on him with the worst single-frame pan at
# 14.9px. Raising it calms the motion further but widens the excursions during
# fast movement -- 40 halves the pan and triples what gets cut off him.
PAN_SIGMA = 15.0

# The speaker crop is a FIXED size that only pans. Deriving it from his bounding
# box, which is what the first version did, means the box breathes with every
# gesture -- an arm goes out, the box widens, the crop widens, and he changes
# size on screen several times a second. Nothing about the shot should change
# except where it points. This is also what track_instructor does.
#
# The window is 432x576, a 3:4 portrait fixed by the plate contract, so a
# full-height crop of the 1280x720 camera is 540px wide -- comfortably more than
# the 346px his gesture box reaches on 90% of lecture-12 frames, and a 0.8x
# downscale rather than an upscale. Lowering PIP_CROP_H zooms in on him at the
# cost of cropping the top or bottom of the room; PIP_CROP_Y says which.
PIP_CROP_H = 1.00       # crop height as a fraction of the source frame height
PIP_CROP_Y = 0.00       # its top edge, same units


def video_encoder(prefer="auto", threads=2, crf=18, preset="veryfast"):
    """ffmpeg output args for the chunk encoders.

    Hardcoding libx264 here was leaving most of a laptop's throughput unused.
    cards.py already owns a smoke-tested picker -- it opens a real encode
    session rather than trusting `ffmpeg -encoders`, which lists h264_nvenc on
    GPUs that have no encoder block -- and on Apple Silicon that selects
    h264_videotoolbox. The win is not only that the media engine is faster: it
    takes the encode OFF the CPU, so the cores go to the per-frame compositing
    instead of competing with x264 for them. On PSC there is no VideoToolbox and
    no usable NVENC, so this falls through to libx264, which is correct there
    rather than a failure.
    """
    try:
        from src.audio.cards import pick_encoder
        name, extra = pick_encoder(prefer, threads=threads)
        return ["-c:v", name, *extra]
    except Exception as e:
        print(f"[layout] encoder probe failed ({e}); using libx264")
        return ["-c:v", "libx264", "-crf", str(crf), "-preset", preset,
                "-threads", str(threads)]


# ---------------------------------------------------------------------------
# where he is
# ---------------------------------------------------------------------------
class Tracker:
    """Instructor bounding box per frame, from short-window motion.

    A median background is the obvious primitive here and it does not work on
    this footage. Sampled locally, the median contains him because he barely
    leaves the podium over any few minutes; sampled across the whole lecture it
    contains him *more*, because he stands there for over half of it. Either way
    he stops registering as foreground exactly where he spends his time, and the
    box drifts onto whatever genuinely did change -- a marker set down on a
    desk, a bottle, a wall panel.

    What is reliable is that he moves and the room does not. The per-pixel range
    over a HALF SECOND of frames is his arms, head and shoulders and nothing
    else. The window has to be short: over a few seconds he has walked, and the
    range becomes the union of everywhere he has been (608x572 on this lecture,
    most of the room).

    Detection runs at half resolution -- the morphological close that joins a
    head to a torso is the expensive step, and at 640x360 it costs a quarter as
    much while the box, scaled back up, is accurate to a few pixels.

    This class only DETECTS. It carries just enough history to pick the right
    blob when several qualify; deciding where the camera points is plan_pan's
    job, offline, once the whole track is known.
    """

    def __init__(self, det_w=640, det_h=360, motion_frames=13, thresh=14,
                 hold=25, roi_top=0.18):
        self.dw, self.dh, self.thresh = det_w, det_h, thresh
        self.sx, self.sy = 1280 / det_w, 720 / det_h
        self.ring = deque(maxlen=motion_frames)
        self.prev = None        # previous frame, for ego-motion
        self.cum = 0.0          # cumulative camera pan, detection pixels
        # Only used to steer candidate selection frame to frame.
        self.hist = deque(maxlen=hold)
        self.current = None
        # The projection screen is IN this shot, and a slide change makes it the
        # largest thing that moves -- so the naive "biggest blob" locks onto the
        # screen at exactly the moments the layout cares about. Ignoring the top
        # of the frame is the cheap half of the fix; the shape, size and
        # continuity tests below are the rest.
        self.roi_y = int(det_h * roi_top)

    def detect(self, frame_bgr):
        """The instructor's box in source coordinates, or None for this frame."""
        small = cv2.cvtColor(cv2.resize(frame_bgr, (self.dw, self.dh)),
                             cv2.COLOR_BGR2GRAY).astype(np.int16)
        # Level each frame against its own median before differencing. The
        # projector lights this room: when a dark slide cuts to a bright one the
        # ambient level jumps and EVERY pixel changes, which surfaces as one
        # blob spanning the whole frame. That blob is rejected for being too
        # wide, so precisely when the slide changes -- the moments this layout
        # cares about -- he vanishes from the track entirely. Subtracting the
        # median cancels a uniform shift and leaves only things that actually
        # moved.
        small -= int(np.median(small))

        # THE CAMERA ITSELF PANS. This is a tracking PTZ, not the fixed camera
        # every earlier version of this assumed: measured on lecture 12, the
        # frame content swings 250 detection-pixels and back inside six seconds,
        # at up to 5.6px per frame, with a quarter of all frames showing global
        # translation. When it moves, every pixel changes, the whole frame comes
        # back as one blob too wide to be a person, and he drops out of the
        # track completely -- which is where the 33% of "gaps" came from, and
        # why they were NOT him standing still.
        #
        # Phase correlation recovers the shift for about 2ms a frame. Ring
        # frames are then rolled into a common reference before differencing, so
        # what is left is motion in the room rather than motion of the camera.
        f32 = small.astype(np.float32)
        if self.prev is not None:
            (dx, _), conf = cv2.phaseCorrelate(self.prev, f32)
            if conf > 0.25:
                self.cum += dx
        self.prev = f32
        self.ring.append((small, self.cum))
        if len(self.ring) < self.ring.maxlen:
            return None

        shifts = [int(round(self.cum - c)) for _, c in self.ring]
        stack = np.stack([np.roll(f, sft, axis=1)
                          for (f, _), sft in zip(self.ring, shifts)])
        fg = ((stack.max(axis=0) - stack.min(axis=0)) > self.thresh
              ).astype(np.uint8) * 255
        # Columns rolled in from the far edge are meaningless; blank them.
        edge = min(self.dw // 2, max(abs(s) for s in shifts) + 2)
        if edge:
            fg[:, :edge] = 0
            fg[:, -edge:] = 0
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
        fg[:self.roi_y] = 0
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cands = [cv2.boundingRect(c) for c in cnts]
        # A standing person is taller than wide and takes up a plausible slice
        # of the frame. The projection screen and a row of seats are far wider
        # than tall; a marker left on a desk or a bottle set down mid-lecture is
        # permanently "foreground" against a whole-lecture median, and is
        # excluded by size rather than by shape -- it is upright too.
        # Motion blobs are looser than silhouettes -- a gesturing arm is wide --
        # so the shape test is gentler here than a background-subtraction one
        # would need to be.
        people = [b for b in cands
                  if b[3] > b[2] * 0.55
                  and 0.20 * self.dh < b[3] < 0.95 * self.dh
                  and b[2] * b[3] * self.sx * self.sy > 8000]
        if people:
            if self.current is not None:
                # Continuity has to be scale-free. Subtracting a per-pixel
                # penalty from an area lets displacement dominate by orders of
                # magnitude -- once the box latched onto a marker across the
                # room it could never afford to move back to him. Dividing means
                # a blob twice as far away simply has to be twice as big.
                pcx = self.current
                best = max(people, key=lambda b: (b[2] * b[3]) /
                           (1 + abs((b[0] + b[2] / 2) * self.sx - pcx) / 250))
            else:
                best = max(people, key=lambda b: b[2] * b[3])
            x, y, w, h = best
            box = (x * self.sx, y * self.sy, w * self.sx, h * self.sy)
            self.hist.append(box[0] + box[2] / 2)
            self.current = float(np.median(self.hist))
            return box
        return None


# ---------------------------------------------------------------------------
# where the camera points -- decided offline, over the whole track
# ---------------------------------------------------------------------------
# Chunks render in separate PROCESSES, and on macOS those are spawned, not
# forked: a worker re-imports brand.py from scratch and gets DEFAULT_LAYOUT,
# not whatever the parent selected. Without this the parent honoured --layout
# and every worker ignored it, so `--layout handoff` composited the handoff
# plate over the wide rectangles -- and it fails SILENTLY, because both
# geometries render a plausible-looking frame. The pool initializer is the
# only place that runs once per worker before any task does.
def _init_worker(layout, scene_b_footer=True):
    global SCENE_B_FOOTER
    brand.set_layout(layout)
    SCENE_B_FOOTER = scene_b_footer


def crop_size(sw, sh, out_w, out_h, h_frac=PIP_CROP_H):
    ch = min(sh, int(round(sh * h_frac)))
    return min(sw, int(round(ch * out_w / out_h))), ch


def _rolling_median(a, win=51):
    """Median over a centred window, ignoring gaps. For outlier rejection only."""
    n = len(a)
    out = np.full(n, np.nan)
    r = win // 2
    for i in range(n):
        seg = a[max(0, i - r):min(n, i + r + 1)]
        seg = seg[~np.isnan(seg)]
        if len(seg):
            out[i] = np.median(seg)
    return out


def _fill(a):
    """Linear interpolation across gaps, edge-held at the ends."""
    idx = np.arange(len(a))
    ok = ~np.isnan(a)
    if not ok.any():
        return np.zeros_like(a)
    return np.interp(idx, idx[ok], a[ok])


def _hold(a):
    """Carry the last known value across gaps; back-fill the leading ones."""
    out = a.copy()
    last = np.nan
    for i in range(len(out)):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    ok = ~np.isnan(out)
    if not ok.any():
        return np.zeros_like(out)
    out[:np.argmax(ok)] = out[np.argmax(ok)]
    return out


def _gauss(x, sigma):
    r = int(max(1, round(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, r, mode="edge"), k, mode="valid")


def plan_pan(boxes, sw, cw, margin=None, sigma=PAN_SIGMA, iters=None,
             reject=300.0):
    """Where the crop's left edge sits, for every frame, decided all at once.

    Causal following could not be smooth, and not for want of tuning. Seeing one
    frame at a time it cannot start moving until he is already at the edge, so
    it must move violently when he is: 0.84px of pan per frame on average
    against a worst case of 223px in a single frame, that worst case being a
    containment clamp firing.

    Offline the whole track is known, so the camera can start moving BEFORE he
    does, which is what a human operator does and what makes motion read as
    deliberate. A centred Gaussian over the target does exactly that: it is
    symmetric, so it leads as much as it lags, and its output is smooth by
    construction -- there is no step it can be made to take.

    On hard containment, which this tried first and abandoned. Treating "he must
    be inside the window" as a per-frame constraint and solving for the smoothest
    path through the resulting corridor is appealing, and infeasible: if he is at
    x=200 before a gap and x=600 after, the corridor is [.., 180] then [300, ..],
    two disjoint sets, and NO smooth path satisfies both. The solver duly
    produced 368px single-frame jumps -- worse than the causal version it
    replaced. Smoothness and hard containment genuinely cannot both hold, so
    smoothness wins and the excursion is bounded instead: at this rail width,
    90% of frames sit dead on him and the worst 1% lose about 75px, mostly a
    hand, during fast movement.

    Two things feed the target, both of which were bugs first:

    * Gaps are HELD, not interpolated. Motion detection loses a person the
      moment they stand still, so a gap means he stayed put. Interpolating
      across it glides the camera from his last position toward the next
      detection and back -- a perfectly smooth pan to nowhere, and why the
      window sat up to 576px off him.
    * Detections far from a rolling median are dropped. A student standing up is
      not him moving; the median is used as the reference so that a genuine walk,
      which the median follows, is not mistaken for one.
    """
    n = len(boxes)
    c = np.full(n, np.nan)
    for i, b in enumerate(boxes):
        if b is not None:
            c[i] = b[0] + b[2] / 2
    if not (~np.isnan(c)).any():
        print("[layout] nothing tracked; centring the rail for the whole run")
        return np.full(n, max(0.0, (sw - cw) / 2)), 1.0

    ref = _rolling_median(c, 51)
    bad = (~np.isnan(c)) & (np.abs(c - ref) > reject)
    c[bad] = np.nan
    gaps = int(np.isnan(c).sum())

    target = _hold(_fill(_rolling_median(c, 9)))
    x = np.clip(_gauss(target, sigma) - cw / 2, 0, sw - cw)

    d = np.abs(np.diff(x)) if n > 1 else np.array([0.0])
    print(f"[layout] pan planned over {n} frames: {int(bad.sum())} outlier "
          f"detection(s) dropped, {100 * gaps / n:.0f}% of frames held through "
          f"gaps")
    print(f"[layout] pan {d.mean():.2f}px/frame mean, {d.max():.2f}px worst")
    return x, 0.0


def crop_at(frame, x, out_w, out_h, h_frac=PIP_CROP_H, y_frac=PIP_CROP_Y):
    """Fixed-size crop whose left edge is `x`, scaled to the slot.

    The size is a constant of the shot, not a function of the detection, so his
    scale on screen never changes -- only where the window points.
    """
    sh, sw = frame.shape[:2]
    cw, ch = crop_size(sw, sh, out_w, out_h, h_frac)
    y = int(round(min(max(0.0, sh * y_frac), sh - ch)))
    x = int(round(max(0, min(sw - cw, x))))
    return cv2.resize(frame[y:y + ch, x:x + cw], (out_w, out_h),
                      interpolation=cv2.INTER_LANCZOS4)


# ---------------------------------------------------------------------------
# compositing
# ---------------------------------------------------------------------------
def compose_scene_a(canvas, plates, slide, cam, x, crop_h=PIP_CROP_H,
                    crop_y=PIP_CROP_Y):
    """Slide window, speaker window, plate over the top.

    Three steps in an order that matters. The plate is opaque everywhere except
    the two windows, so copying it wholesale first is a memcpy that lands the
    correct pixel value across ~37% of the frame; the windows are then filled
    with video; then the plate is blended back over each window so its 1px
    antialiased edge and rounded corners cut the video cleanly instead of
    leaving a hard rectangle with square corners.

    Only the two window boxes are blended, never the whole frame. Outside them
    the plate is fully opaque, so a full-frame alpha composite would be two
    megapixels of float arithmetic per frame to reproduce a memcpy.
    """
    sx, sy, sw, sh = brand.SLIDE_WINDOW
    cx, cy, cw, ch = brand.SPEAKER_WINDOW

    plates["scene_a"].fill(canvas)
    canvas[sy:sy + sh, sx:sx + sw] = slide
    canvas[cy:cy + ch, cx:cx + cw] = crop_at(cam, x, cw, ch, h_frac=crop_h,
                                             y_frac=crop_y)
    plates["scene_a"].blend(canvas, (sx, sy, sx + sw, sy + sh))
    plates["scene_a"].blend(canvas, (cx, cy, cx + cw, cy + ch))


def compose_scene_b(canvas, plates, slide, cam, x, footer=True):
    """The room, whole and uncropped, under the brand footer.

    No rail and no metadata: the slide would be the same information twice --
    the projection screen is IN this shot -- and there is nothing composited
    here to label. But the footer stays, which is the change Phillip asked
    for. It used to carry a 60%-white unitmark in the top right instead, and
    Plate.legibility measured that mark moving its pixels by at most 42/255
    against a background whose mean was 249: the top right of this framing is
    the projection screen, so the watermark was landing on the one white
    rectangle in the room.

    The footer's opacity is set per frame from the luma actually under it, so
    the same band reads on a dark lecture hall and on a lit whiteboard. That
    is the whole point of moving it here rather than pasting the Scene A strip
    on at constant alpha.
    """
    canvas[:, :] = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LANCZOS4)
    if not footer:
        plates["scene_b"].blend(canvas)
        return
    strip_bgr, strip_a = plates["footer"]
    band = canvas[brand.FOOTER_Y:]
    s = brand.scrim_alpha(band.mean())
    a = strip_a * s
    band[:, :] = (band.astype(np.float32) * (1.0 - a)
                  + strip_bgr * a).astype(np.uint8)


def load_cards(paths, work=None):
    """The question cards as full 1920x1080 images, with their spans.

    A card is NOT a slide, and putting it through the slide window is what the
    first version did: 1380px wide inside the plate, its type shrunk, the
    SCS lockup on the card sitting next to the SCS lockup in the footer, and
    the instructor rail competing with a question the viewer is meant to read.
    render_full_lecture.py had already written this down -- "the card is
    authored as a complete 1920x1080, so it replaces the whole frame" -- and
    this module never honoured it.

    One frame per card is enough because a card is a still: cards.py holds the
    same rendered PNG for the whole span. So this is two decodes, not two
    thousand, and each worker does its own rather than pickling megabytes of
    pixels across the pool.
    """
    if not os.path.exists(paths.cards_manifest):
        return []
    with open(paths.cards_manifest) as f:
        spans = json.load(f)
    src = paths.screen_with_cards
    if not spans or not os.path.exists(src):
        return []
    out = []
    for c in spans:
        mid = (c["start"] + c["end"]) / 2
        proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-ss", str(mid), "-i", src,
             "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        buf = proc.stdout.read(W * H * 3)
        proc.stdout.close()
        proc.wait()
        if len(buf) < W * H * 3:
            print(f"[layout] could not read the card frame at {mid:.1f}s; "
                  f"that span will render as the normal layout")
            continue
        out.append({"start": c["start"], "end": c["end"],
                    "img": np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()})
    if out:
        print(f"[layout] {len(out)} question card(s) will play FULL FRAME")
    return out


# "Default to a hard cut on a phrase boundary; use a 4 to 6 frame dissolve only
# to soften a visible framing or exposure jump." Every transition this pipeline
# makes IS such a jump -- a slide-led composite to a full-bleed room, or either
# of those to a full-screen card -- so the dissolve applies at every scene
# change rather than being conditioned on a measurement that would always come
# back true.
#
# 5 frames -- 0.2s, the middle of the range Phillip's package gives -- is what
# this used to do, and on 2026-08-20 he watched it and said the transitions
# were too fast. That range was written for softening a framing jump inside
# one continuous shot, which is not what these are: every cut here is between
# two layouts that share no structure at all, and 0.2s between them reads as a
# glitch rather than as an edit. 12 frames is 0.48s, which is long enough to
# be read as deliberate and short enough not to slow the lecture down.
#
# Overridable per render with --transition-frames, because "slow them down" is
# a judgement and the number should be easy to move.
#
# The audio is untouched by any of this. It is one continuous track muxed over
# the whole render, which is the other half of the same instruction.
DISSOLVE_FRAMES = 12
# Softness of a wipe's edge, as a fraction of the frame width. A hard edge
# reads as a tear; too soft and it is just a slow crossfade with extra steps.
WIPE_SOFT = 0.06
# A wipe is a visible movement rather than a way of hiding a jump, so it needs
# longer than a dissolve or it reads as a swipe rather than a transition.
WIPE_FRAMES = 20


def blend_transition(dst, nxt, alpha, style="dissolve"):
    """Mix `nxt` into `dst` in place, `alpha` of the way through a transition.

    dissolve  a crossfade over the whole frame. What the brand spec asks for.
    wipe      a soft-edged vertical edge travelling left to right. Not in the
              spec -- it is a visible transition rather than a way of hiding a
              jump -- so it is opt-in and the default stays the spec's.
    """
    if style == "dissolve":
        cv2.addWeighted(nxt, alpha, dst, 1.0 - alpha, 0.0, dst=dst)
        return
    if style == "dip":
        # Fade out to the field colour, then in to the next scene. Unlike a
        # crossfade it never shows both scenes at once, which is what makes it
        # read cleanly between two layouts that share no structure -- a
        # windowed composite and a full-bleed camera have nothing to ghost
        # INTO each other. Dips to the brand field rather than pure black so
        # the darkest moment is still the plate's own colour.
        field = np.array(FIELD[::-1], dtype=np.float32)
        k = abs(alpha - 0.5) * 2.0            # 1 at the ends, 0 in the middle
        src = dst if alpha < 0.5 else nxt
        np.copyto(dst, (src * k + field * (1.0 - k)).astype(np.uint8))
        return
    if style == "wipe":
        soft = max(1.0, WIPE_SOFT * W)
        # The edge has to travel a full frame width PLUS its own softness at
        # each end, or the wipe starts already part-done and finishes early.
        edge = alpha * (W + 2 * soft) - soft
        xs = np.arange(W, dtype=np.float32)
        m = np.clip((edge - xs) / soft + 0.5, 0.0, 1.0)[None, :, None]
        np.copyto(dst, (nxt * m + dst * (1.0 - m)).astype(np.uint8))
        return
    if alpha >= 0.5:                      # "cut": no blend, just switch over
        np.copyto(dst, nxt)


def build_timeline(scenes, cards, duration):
    """One ordered list of (start, end, kind) covering [0, duration).

    Cards are not in scenes.json -- they come from cards.json and override
    whatever scene they land inside -- so the two have to be merged before
    anything can ask "what is the next scene, and when does it start". Doing
    that once, up front, is also what makes a dissolve possible at a chunk
    boundary: every worker derives the same timeline from the same two files.
    """
    spans = []
    cut = sorted({0.0, duration}
                 | {s["start"] for s in scenes} | {s["end"] for s in scenes}
                 | {c["start"] for c in cards} | {c["end"] for c in cards})
    cut = [t for t in cut if 0.0 <= t <= duration]
    for a, b in zip(cut, cut[1:]):
        if b - a < 1e-6:
            continue
        mid = (a + b) / 2
        kind = "card" if card_at(cards, mid) else scene_at(scenes, mid)
        if spans and spans[-1][2] == kind:
            spans[-1] = (spans[-1][0], b, kind)
        else:
            spans.append((a, b, kind))
    return spans


def transition_at(timeline, t, n=DISSOLVE_FRAMES, fps=FPS):
    """(kind_from, kind_to, alpha) at time t, or (kind, None, 0.0).

    The dissolve is centred on the cut, so it starts half a window early and
    ends half a window late. alpha runs 0 -> 1 across it.
    """
    half = (n / fps) / 2.0
    for i in range(1, len(timeline)):
        boundary = timeline[i][0]
        if abs(t - boundary) <= half:
            a = (t - (boundary - half)) / (2 * half)
            return timeline[i - 1][2], timeline[i][2], min(1.0, max(0.0, a))
    for a, b, kind in timeline:
        if a <= t < b:
            return kind, None, 0.0
    return (timeline[-1][2] if timeline else "scene_a"), None, 0.0


def card_at(cards, t):
    for c in cards:
        if c["start"] <= t < c["end"]:
            return c
    return None


def compose_card(canvas, card):
    """The card, whole. No plate, no rail, no footer -- it carries its own."""
    canvas[:, :] = card["img"]


def compose_kind(canvas, kind, plates, cards, slide, cam, x, t,
                 crop_h=PIP_CROP_H, crop_y=PIP_CROP_Y):
    """Draw whichever scene `kind` names into `canvas`."""
    if kind == "card":
        c = card_at(cards, t)
        if c is not None:
            compose_card(canvas, c)
            return
        kind = "scene_a"
    if kind == "scene_b":
        compose_scene_b(canvas, plates, slide, cam, x,
                        footer=SCENE_B_FOOTER)
    else:
        compose_scene_a(canvas, plates, slide, cam, x,
                        crop_h=crop_h, crop_y=crop_y)


# How often a Scene B frame is measured for watermark legibility, and how faint
# is too faint. The threshold is a floor, not a target: 0.06 is the mark moving
# the pixels under it by about 15/255, which is around where a 60%-alpha white
# mark stops being visible at all. Lecture 12 measures 0.003 -- the top right of
# this camera framing is the projection screen, and white on white is nothing.
WATERMARK_SAMPLE_EVERY = 25             # frames, i.e. once a second
WATERMARK_FLOOR = 0.06

# Scene B carries the adaptive footer rather than the corner unitmark. The
# unitmark path is kept because it is what every video published before this
# change looks like, and because the legibility measurement above is the
# evidence for having moved it -- deleting the code would delete the ability
# to reproduce the finding. Set False (or --scene-b-watermark) to go back.
SCENE_B_FOOTER = True


def report_watermark(samples, where=""):
    """Say whether Scene B's watermark could actually be seen.

    A render that finishes and a watermark that survives are different
    questions, and only one of them has ever been checked. This answers the
    other one from frames the loop already had in hand.
    """
    if not samples:
        return
    faint = [v for v in samples if v < WATERMARK_FLOOR]
    mean = sum(samples) / len(samples)
    msg = (f"[layout] scene_b watermark legibility {mean:.3f} mean over "
           f"{len(samples)} sample(s){where}")
    if faint:
        msg += (f"; {100 * len(faint) / len(samples):.0f}% below {WATERMARK_FLOOR:g} "
                f"-- the mark is effectively INVISIBLE there. The top right of "
                f"this framing is the projection screen. Ask for a darker or "
                f"repositioned watermark rather than accepting it.")
    print(msg, flush=True)


def build_plates(meta, plate_dir=None, verify=True):
    """Both scenes' overlays, with this lecture's type already drawn in.

    Built once per process -- including once per worker in the parallel render,
    which is why this takes the metadata rather than a finished Plate: numpy
    arrays that large are not worth pickling across a process boundary when
    rebuilding them costs a PNG decode.
    """
    a = os.path.join(plate_dir, "scene-a-overlay.png") if plate_dir \
        else brand.SCENE_A_PLATE
    b = os.path.join(plate_dir, "scene-b-overlay.png") if plate_dir \
        else brand.SCENE_B_PLATE
    return {"scene_a": Plate(a, meta, verify=verify),
            # Scene B has no windows to check and no footer to lose.
            "scene_b": Plate(b, verify=False, windows=None),
            # The Scene A footer band, for Scene B to lay over the room. Taken
            # from the SAME art, so the two scenes cannot drift apart.
            "footer": brand.footer_strip(a)}


# ---------------------------------------------------------------------------
# pass 1: detect
# ---------------------------------------------------------------------------
def _track_usable(doc, cam, t0, t1):
    """Whether a cached track covers this range AND is worth reusing.

    Two checks beyond the range, both learned the hard way.

    A track with NO detections in it is not a track. A detection pass over a
    camera file that does not exist produces exactly that -- every frame None --
    and the old code cached it, so the next run logged "track from
    instructor_track.json" and centred the rail for the whole lecture. A cache
    hit that reads as good news is the worst way to lose the framing.

    And a track is only valid for the FILE it was measured on. The cache used to
    record t0/t1/fps and nothing else, so a track taken from one camera would be
    silently reused for another.
    """
    if doc.get("fps") != FPS:
        return False
    if not (doc["t0"] <= t0 + 1e-6 and doc["t1"] >= t1 - 1e-6):
        return False
    src = doc.get("source")
    if src and src != os.path.basename(cam):
        print(f"[layout] cached track was measured on {src}, rendering from "
              f"{os.path.basename(cam)} -- re-detecting")
        return False
    if not any(doc["boxes"]):
        print("[layout] cached track has no detections in it at all; that is a "
              "failed pass, not a still lecture -- re-detecting")
        return False
    return True


def _write_track(cache_path, cam, t0, t1, boxes):
    """Cache a track, unless it found nothing -- see _track_usable."""
    found = sum(b is not None for b in boxes)
    if not found:
        print(f"[layout] not caching {os.path.basename(cache_path)}: 0 of "
              f"{len(boxes)} frames tracked. Re-running is cheaper than "
              f"rendering 90 minutes off a dead track.")
        return
    with open(cache_path, "w") as f:
        json.dump({"t0": t0, "t1": t1, "fps": FPS,
                   "source": os.path.basename(cam),
                   "boxes": [list(b) if b else None for b in boxes]}, f)
    print(f"[layout] wrote {cache_path}")


def detect_track(cam, t0, dur, cache_path=None, retrack=False):
    """Instructor box per frame over [t0, t0+dur), as a list with None for gaps.

    A whole extra decode of the camera, which is why the result is cached: the
    pan planner needs the entire track before it can place a single frame, so
    this cannot be folded into the render loop. track_instructor.py caches its
    own detections for the same reason -- detection is the expensive half and
    the derived crop is nearly free, so re-deciding the framing should not mean
    re-detecting.
    """
    t1 = t0 + dur
    if cache_path and os.path.exists(cache_path) and not retrack:
        with open(cache_path) as f:
            doc = json.load(f)
        if _track_usable(doc, cam, t0, t1):
            off = int(round((t0 - doc["t0"]) * FPS))
            n = int(round(dur * FPS))
            boxes = [tuple(b) if b else None
                     for b in doc["boxes"][off:off + n]]
            if len(boxes) >= n - 1:
                print(f"[layout] track from {os.path.basename(cache_path)} "
                      f"({doc['t0']:.0f}-{doc['t1']:.0f}s)")
                return boxes
        print(f"[layout] cached track covers {doc['t0']:.0f}-{doc['t1']:.0f}s, "
              f"need {t0:.0f}-{t1:.0f}s -- re-detecting")

    print(f"[layout] detecting the instructor over {t0:.0f}-{t1:.0f}s")
    proc = _reader(cam, t0, dur, 1280, 720)
    tr = Tracker()
    boxes = []
    while True:
        f = _read_frame(proc, 1280, 720)
        if f is None:
            break
        boxes.append(tr.detect(f))
        if len(boxes) % (FPS * 120) == 0:
            print(f"[layout]   {len(boxes) / FPS:.0f}s tracked", flush=True)
    proc.stdout.close()
    proc.wait()
    found = sum(b is not None for b in boxes)
    print(f"[layout] tracked {len(boxes)} frames, found him in {found} "
          f"({100 * found / max(1, len(boxes)):.0f}%)")

    if cache_path:
        _write_track(cache_path, cam, t0, t1, boxes)
    return boxes


def _reader(path, t0, dur, w, h):
    return subprocess.Popen(
        ["ffmpeg", "-v", "error", "-ss", str(t0), "-t", str(dur), "-i", path,
         "-vf", f"fps={FPS},scale={w}:{h}", "-f", "rawvideo",
         "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        bufsize=w * h * 3 * 4)


def _read_frame(proc, w, h):
    n = w * h * 3
    buf = proc.stdout.read(n)
    if len(buf) < n:
        return None
    return np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)


def slide_source(paths):
    """Which screen file the slide region is drawn from.

    screen_with_cards.mp4 when cards.py has run, because a question card
    REPLACES whole screen frames -- reading screen_sync instead silently drops
    every card, and the cards are the privacy substitute for a student's muted
    question. That is the opposite choice from scenes.py, which must detect
    freezes on screen_sync precisely BECAUSE a card is a static full-frame
    image and would otherwise read as a frozen slide.
    """
    if os.path.exists(paths.screen_with_cards):
        return paths.screen_with_cards
    return paths.screen_sync


def scene_at(scenes, t):
    """Canonical scene id at time t. Uncovered time is Scene A -- the cut list
    is meant to tile the whole lecture, and defaulting a hole to the full-bleed
    room would put the camera up with no plate and no explanation."""
    for s in scenes:
        if s["start"] <= t < s["end"]:
            return scene_name(s["scene"])
    return "scene_a"


def _cropdetect_at(path, t, window, limit):
    out = subprocess.run(
        ["ffmpeg", "-v", "info", "-ss", str(t), "-t", str(window),
         "-i", path, "-vf", f"cropdetect={limit}:2:0", "-f", "null", "-"],
        capture_output=True, text=True).stderr
    found = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", out)
    return tuple(int(v) for v in found[-1]) if found else None


def detect_content_box(path, duration, samples=4, limit=24, window=4.0,
                       must_hold_at=()):
    """The capture's real content rectangle, as (w, h, x, y), or None if full.

    Panopto hands back whatever the room's capture appliance produced, and not
    every room produces 16:9. 17-635 lecture 13 is a 4:3 deck pillarboxed into
    a 1920x1080 frame: 1440x1080 of slide with a 240px black bar down each
    side. Scaled into the slide window unchanged, those bars come along --
    the slide lands about 1035px wide inside a 1380px window with black
    pillars either side of it, INSIDE the plate's rounded cutout. It reads as
    a broken render, and a quarter of the window is wasted.

    Sampled at several points rather than one, and by consensus, because a
    single dark slide reads as letterboxed on its own.
    """
    if not duration or duration <= 0:
        return None
    votes = {}
    for k in range(samples):
        t = duration * (k + 0.5) / samples
        box = _cropdetect_at(path, t, window, limit)
        if box:
            votes[box] = votes.get(box, 0) + 1
    if not votes:
        return None
    box, n = max(votes.items(), key=lambda kv: kv[1])
    w, h, x, y = box
    src = probe_size(path)
    if src and (w, h) == src and (x, y) == (0, 0):
        return None
    if n <= samples // 2:
        print(f"[layout] cropdetect disagreed across samples ({votes}); "
              f"leaving the capture uncropped")
        return None
    # The box has to hold over the WHOLE track, not just the slides. Question
    # cards are authored as full-frame 1920x1080 art and inserted into the same
    # stream, so a crop derived from a pillarboxed 4:3 deck would shave 240px
    # off each side of every card -- straight through the CMU lockup on one
    # side and the tartan wedge on the other. Evenly spaced samples never catch
    # this: on 17-635 lecture 13 the cards are 9 seconds out of 4,925.
    for t in must_hold_at:
        other = _cropdetect_at(path, t, min(window, 2.0), limit)
        if other and other != box:
            print(f"[layout] content box {w}x{h}+{x}+{y} does not hold at "
                  f"{t:.0f}s (there it is {other[0]}x{other[1]}+{other[2]}+"
                  f"{other[3]}) -- a card or another full-frame insert. Not "
                  f"cropping; a single crop cannot suit both.")
            return None
    if src:
        print(f"[layout] screen capture is {src[0]}x{src[1]} with "
              f"{w}x{h}+{x}+{y} of content -- cropping the "
              f"{'pillar' if x else 'letter'}box off before the slide window")
    return box


def probe_size(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", path],
        capture_output=True, text=True).stdout.strip().split("\n")[0]
    try:
        w, h = out.split("x")[:2]
        return int(w), int(h)
    except ValueError:
        return None


def _slide_filter(box=None):
    """The scale/crop/pad chain that puts a slide inside the slide window.

    The plates README says "crop, do not stretch", and
    force_original_aspect_ratio=increase plus crop is exactly that -- for a
    source whose aspect matches the window. 1380x776 is 1.7784:1 against a
    16:9 capture's 1.7778:1, a 0.03% difference, so those go in as specified.

    A 4:3 capture is a different question, and cropping is the wrong answer to
    it. Filling a 16:9 window from 4:3 content means cutting 26% off the top
    and bottom, which on a lecture slide is the title and the footer. Slides
    are the one thing in this layout that must not lose content, so anything
    that is not already the window's shape is fitted INSIDE it and padded to
    the brand field colour. Wasting a little window on a genuinely 4:3 deck is
    the correct trade; silently cropping the title is not.
    """
    x, y, w, h = brand.SLIDE_WINDOW
    pre = f"crop={box[0]}:{box[1]}:{box[2]}:{box[3]}," if box else ""
    src_ar = (box[0] / box[1]) if box else None
    win_ar = w / h
    if src_ar is None or abs(src_ar - win_ar) / win_ar < 0.02:
        return (f"{pre}scale={w}:{h}:force_original_aspect_ratio=increase,"
                f"crop={w}:{h}")
    field = "0x%02X%02X%02X" % FIELD
    return (f"{pre}scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:{field}")


def _slide_reader(path, t0, dur, box=None):
    """The slide window's frames, fitted into it without distortion."""
    x, y, w, h = brand.SLIDE_WINDOW
    return subprocess.Popen(
        ["ffmpeg", "-v", "error", "-ss", str(t0), "-t", str(dur), "-i", path,
         "-vf", f"fps={FPS},{_slide_filter(box)}",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        bufsize=w * h * 3 * 4)


def render(paths, scenes, t0, dur, out_path, meta, crf=18, preset="veryfast",
           crop_h=PIP_CROP_H, crop_y=PIP_CROP_Y, pan_sigma=PAN_SIGMA,
           retrack=False, plate_dir=None, slide_box=None, encoder="auto",
           style="dissolve", tframes=DISSOLVE_FRAMES):
    cam = paths.resolve_camera_for_assembly()
    plates = build_plates(meta, plate_dir)
    sx, sy, sw, sh = brand.SLIDE_WINDOW
    cw_out, ch_out = brand.SPEAKER_WINDOW[2], brand.SPEAKER_WINDOW[3]

    cards = load_cards(paths)
    boxes = detect_track(cam, t0, dur, paths.instructor_track, retrack)
    cw, _ = crop_size(1280, 720, cw_out, ch_out, crop_h)
    pan, _ = plan_pan(boxes, 1280, cw, sigma=pan_sigma)

    slide_in = _slide_reader(slide_source(paths), t0, dur, slide_box)
    cam_in = _reader(cam, t0, dur, 1280, 720)

    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-",
         "-ss", str(t0), "-t", str(dur), "-i", cam,
         "-map", "0:v", "-map", "1:a?",
         *video_encoder(encoder, 0, crf, preset),
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
         "-movflags", "+faststart", "-shortest", out_path],
        stdin=subprocess.PIPE)

    timeline = build_timeline(scenes, cards, t0 + dur)
    canvas = np.empty((H, W, 3), dtype=np.uint8)
    other = np.empty((H, W, 3), dtype=np.uint8)
    canvas[:, :] = FIELD[::-1]          # BGR
    n, cuts, prev, wm, blended = 0, 0, None, [], 0
    while True:
        slide = _read_frame(slide_in, sw, sh)
        frame = _read_frame(cam_in, 1280, 720)
        if slide is None or frame is None:
            break
        t = t0 + n / FPS
        x = pan[min(n, len(pan) - 1)]
        sc, to, alpha = transition_at(timeline, t, tframes)
        if sc != prev:
            if prev is not None:
                cuts += 1
            print(f"[layout]   t={t:8.1f}s  -> {sc}")
            prev = sc
        compose_kind(canvas, sc, plates, cards, slide, frame, x, t,
                     crop_h, crop_y)
        if to is not None and to != sc:
            compose_kind(other, to, plates, cards, slide, frame, x, t,
                         crop_h, crop_y)
            blend_transition(canvas, other, alpha, style)
            blended += 1
        elif (sc == "scene_b" and not SCENE_B_FOOTER
              and n % WATERMARK_SAMPLE_EVERY == 0):
            wm.append(plates["scene_b"].legibility(canvas))
        enc.stdin.write(canvas.tobytes())
        n += 1
        if n % (FPS * 30) == 0:
            print(f"[layout]   {n / FPS:.0f}s / {dur:.0f}s", flush=True)

    enc.stdin.close()
    enc.wait()
    for p in (slide_in, cam_in):
        p.stdout.close()
        p.wait()
    report_watermark(wm)
    print(f"[layout] {n} frames ({n / FPS:.1f}s), {cuts} cuts, "
          f"{blended} dissolved frame(s) -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# parallel whole-lecture render
# ---------------------------------------------------------------------------
# Chunks are rendered VIDEO ONLY and the audio is muxed once at the end, from
# the camera, in a single pass. Per-chunk audio would put an AAC encoder
# boundary at every join, and those are audible as clicks on a lecture where
# the joins land mid-sentence.
def _detect_range(args):
    """One worker's slice of the detection pass."""
    idx, cam, t0, dur, warm = args
    pre = min(warm, t0)
    proc = _reader(cam, t0 - pre, dur + pre, 1280, 720)
    tr = Tracker()
    skip = int(round(pre * FPS))
    out, i = [], 0
    while True:
        f = _read_frame(proc, 1280, 720)
        if f is None:
            break
        b = tr.detect(f)
        if i >= skip:
            out.append(b)
        i += 1
    proc.stdout.close()
    proc.wait()
    n = int(round(dur * FPS))
    out = (out + [None] * n)[:n]
    print(f"[layout] track chunk {idx} ({t0:.0f}-{t0 + dur:.0f}s) "
          f"{sum(b is not None for b in out) * 100 // max(1, n)}% found",
          flush=True)
    return idx, out


def _render_range(args):
    (idx, lecture_dir, meta, plate_dir, crop_h, crop_y, scenes, pan, t0, dur,
     out_path, crf, preset, threads, slide_box, encoder, style, tframes) = args
    p = LecturePaths(lecture_dir)
    cam = p.resolve_camera_for_assembly()
    # Verified once in the parent; re-verifying in every worker would print the
    # same contract check N times for no new information.
    plates = build_plates(meta, plate_dir, verify=False)
    cards = load_cards(p)
    sx, sy, sw, sh = brand.SLIDE_WINDOW
    pan = np.asarray(pan)
    n_frames = int(round(dur * FPS))

    slide_in = _slide_reader(slide_source(p), t0, dur, slide_box)
    cam_in = _reader(cam, t0, dur, 1280, 720)
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-an",
         *video_encoder(encoder, threads, crf, preset),
         "-pix_fmt", "yuv420p", out_path],
        stdin=subprocess.PIPE)

    timeline = build_timeline(scenes, cards, t0 + dur if t0 + dur > 0 else dur)
    canvas = np.empty((H, W, 3), dtype=np.uint8)
    other = np.empty((H, W, 3), dtype=np.uint8)
    canvas[:, :] = FIELD[::-1]
    last, wm = None, []
    for i in range(n_frames):
        slide = _read_frame(slide_in, sw, sh)
        frame = _read_frame(cam_in, 1280, 720)
        if slide is None or frame is None:
            if last is None:
                break
            # Hold the last frame rather than ending short. Every chunk must be
            # exactly n_frames long or the concatenated video drifts against the
            # audio muxed in afterwards.
            slide, frame = last
        else:
            last = (slide, frame)
        t = t0 + i / FPS
        x = pan[min(i, len(pan) - 1)]
        sc, to, alpha = transition_at(timeline, t, tframes)
        compose_kind(canvas, sc, plates, cards, slide, frame, x, t,
                     crop_h, crop_y)
        if to is not None and to != sc:
            compose_kind(other, to, plates, cards, slide, frame, x, t,
                         crop_h, crop_y)
            blend_transition(canvas, other, alpha, style)
        elif (sc == "scene_b" and not SCENE_B_FOOTER
              and i % WATERMARK_SAMPLE_EVERY == 0):
            wm.append(plates["scene_b"].legibility(canvas))
        enc.stdin.write(canvas.tobytes())

    enc.stdin.close()
    enc.wait()
    for r in (slide_in, cam_in):
        r.stdout.close()
        r.wait()
    report_watermark(wm, f" in chunk {idx}")
    print(f"[layout] chunk {idx} rendered {t0:.0f}-{t0 + dur:.0f}s", flush=True)
    return idx, out_path


def render_parallel(paths, scenes, t0, dur, out_path, meta, jobs, crf=18,
                    preset="veryfast", crop_h=PIP_CROP_H, crop_y=PIP_CROP_Y,
                    pan_sigma=PAN_SIGMA, retrack=False, work=None, threads=2,
                    plate_dir=None, slide_box=None, encoder="auto",
                    style="dissolve", tframes=DISSOLVE_FRAMES):
    cam = paths.resolve_camera_for_assembly()
    work = work or os.path.join(paths.dir, ".layout-work")
    os.makedirs(work, exist_ok=True)

    # Build the plates once here, before any encoding starts, purely so the
    # contract check and the type-fitting warnings land at the TOP of the log
    # rather than interleaved with 12 workers' progress -- or, worse, after an
    # hour of encode.
    build_plates(meta, plate_dir)

    edges = [t0 + dur * k / jobs for k in range(jobs + 1)]
    spans = [(edges[k], edges[k + 1] - edges[k]) for k in range(jobs)]

    # --- 1. detection, in parallel. Each chunk decodes a second of pre-roll it
    # then discards, because the motion ring needs filling before it can report
    # anything and a cold start would blind the first frames of every chunk.
    boxes = None
    if not retrack and os.path.exists(paths.instructor_track):
        with open(paths.instructor_track) as f:
            doc = json.load(f)
        if _track_usable(doc, cam, t0, t0 + dur):
            off = int(round((t0 - doc["t0"]) * FPS))
            boxes = [tuple(b) if b else None
                     for b in doc["boxes"][off:off + int(round(dur * FPS))]]
            print(f"[layout] track from {os.path.basename(paths.instructor_track)}")
    if boxes is None:
        print(f"[layout] detecting over {jobs} chunks")
        with ProcessPoolExecutor(max_workers=jobs,
                                 initializer=_init_worker,
                                 initargs=(brand.LAYOUT,
                                           SCENE_B_FOOTER)) as ex:
            parts = list(ex.map(_detect_range,
                                [(k, cam, s, d, 1.0)
                                 for k, (s, d) in enumerate(spans)]))
        boxes = [b for _, part in sorted(parts) for b in part]
        _write_track(paths.instructor_track, cam, t0, t0 + dur, boxes)

    # --- 2. plan the pan ONCE, over the whole track. Planning per chunk would
    # put a smoothing discontinuity at every join.
    cw, _ = crop_size(1280, 720, brand.SPEAKER_WINDOW[2],
                      brand.SPEAKER_WINDOW[3], crop_h)
    pan, _ = plan_pan(boxes, 1280, cw, sigma=pan_sigma)

    # --- 3. render
    tasks = []
    for k, (s, d) in enumerate(spans):
        i0 = int(round((s - t0) * FPS))
        i1 = i0 + int(round(d * FPS))
        tasks.append((k, paths.dir, meta, plate_dir, crop_h, crop_y, scenes,
                      [float(v) for v in pan[i0:i1 + 2]], s, d,
                      os.path.join(work, f"part{k:03d}.mp4"), crf, preset,
                      threads, slide_box, encoder, style, tframes))
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker,
                             initargs=(brand.LAYOUT,
                                       SCENE_B_FOOTER)) as ex:
        done = sorted(ex.map(_render_range, tasks))
    parts = [p for _, p in done]

    # --- 4. join, then lay the audio over the whole thing in one pass
    # A chunk that decoded nothing writes a valid-but-streamless mp4, and
    # concat's complaint about it ("Output file does not contain any stream")
    # names the concat, not the chunk that was empty or why. Check here, where
    # the cause is still in scope.
    empty = [pth for pth in parts
             if not os.path.exists(pth) or os.path.getsize(pth) < 1024]
    if empty:
        raise SystemExit(
            f"{len(empty)} of {len(parts)} rendered chunks are empty, e.g. "
            f"{os.path.basename(empty[0])}. The readers produced no frames -- "
            f"check that {os.path.basename(cam)} and "
            f"{os.path.basename(slide_source(paths))} both decode over "
            f"{t0:.0f}-{t0 + dur:.0f}s.")

    lst = os.path.join(work, "parts.txt")
    with open(lst, "w") as f:
        for pth in parts:
            f.write(f"file '{os.path.abspath(pth)}'\n")
    silent = os.path.join(work, "video.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
                    "-i", lst, "-c", "copy", silent], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", silent,
                    "-ss", str(t0), "-t", str(dur), "-i", cam,
                    "-map", "0:v", "-map", "1:a?", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "128k", "-shortest",
                    "-movflags", "+faststart", out_path], check=True)
    print(f"[layout] joined {len(parts)} chunks -> {out_path}")
    return out_path



# Which layout a deck wants, from its own shape.
#
# "Create and test separate layouts for 4:3 and 16:9 slides" only helps if
# something picks. Left to a flag, the 4:3 layout would be used on the one
# lecture somebody remembered it for, and every other 4:3 deck in the corpus
# would quietly get the 16:9 one -- which is not broken, just wasteful, and so
# would never be noticed. The content box is already measured for the crop, so
# the choice costs nothing.
AUTO_LAYOUTS = [("wide43", 4 / 3), ("wide", 16 / 9)]


def pick_layout(box, default=brand.DEFAULT_LAYOUT):
    """Layout name for a detected content box, nearest aspect wins."""
    if not box or not box[1]:
        print(f"[layout] no content box measured; using {default}")
        return default
    ar = box[0] / box[1]
    name, want = min(AUTO_LAYOUTS, key=lambda kv: abs(ar - kv[1]))
    print(f"[layout] deck measures {box[0]}x{box[1]} ({ar:.3f}:1) -> "
          f"layout {name} ({want:.3f}:1)")
    return name




def default_jobs(cores=None):
    """How many chunk workers to run. A THIRD of the cores, not all of them.

    This defaulted to cores-2, and cores-2 is measurably the wrong answer.
    Rendering the same 60s of lecture 12 on a 10-core laptop:

        jobs=2  34.4s      jobs=5  33.6s      jobs=8   48.5s
        jobs=3  32.4s      jobs=6  33.9s      jobs=10  57.6s

    Flat from 2 to 6 and then sharply worse -- the old default of 8 was 50%
    slower than 3. The reason is the one psc_scan.sbatch already carries for
    the scanner: every worker spawns its own ffmpeg readers and encoder, each
    internally threaded, so N workers ask for far more than N cores' worth of
    threads and spend the difference context-switching.

    That the curve is FLAT from 2 to 6, rather than improving, says the limit
    is not core count at all -- swapping h264_videotoolbox for libx264 changes
    nothing at jobs=3 (32.2s vs 33.1s), so it is not the encoder either. It is
    the per-frame decode and composite, which is already parallel inside
    ffmpeg. Adding processes on top of that only competes with it.
    """
    cores = cores or os.cpu_count() or 4
    return max(1, cores // 3)



def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_lecture_args(ap)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=None,
                    help="Seconds to render. Default: to the end.")
    ap.add_argument("--out", default=None,
                    help="Default: <key>-layout.mp4 in the lecture dir")
    ap.add_argument("--term", default=None,
                    help="Overrides the term derived from the Panopto start "
                         "time, e.g. 'Spring 2026'")
    ap.add_argument("--course-title", default=None,
                    help="Overrides assets/brand/courses.json for this render")
    ap.add_argument("--layout", default="auto",
                    choices=sorted(brand.LAYOUTS) + ["auto"],
                    help="Window geometry. 'auto' (default) measures the "
                         "deck and picks 'wide' for 16:9 or 'wide43' for 4:3; "
                         "'handoff' is Phillip's numbers verbatim. Each has "
                         "its own plate -- switching this switches both, and "
                         "they must match or the art lands over the wrong "
                         "rectangles.")
    ap.add_argument("--plate-dir", default=None,
                    help="Directory holding scene-a-overlay.png and "
                         "scene-b-overlay.png. Default: assets/brand/plates. "
                         "Point this at a redesign to try it -- the two live "
                         "windows must stay where the contract says.")
    ap.add_argument("--pip-crop-h", type=float, default=PIP_CROP_H,
                    help=f"Speaker crop height as a fraction of the source "
                         f"frame. Default {PIP_CROP_H:g} (the whole frame); "
                         f"lower zooms in on him and crops the room.")
    ap.add_argument("--pip-crop-y", type=float, default=PIP_CROP_Y,
                    help="Top edge of that crop, same units")
    ap.add_argument("--pan-sigma", type=float, default=PAN_SIGMA,
                    help="Pan smoothing width in frames (25 = 1s). Higher is "
                         "calmer.")
    ap.add_argument("--jobs", type=int, default=default_jobs(),
                    help=f"Render in this many parallel processes (default "
                         f"{default_jobs()} here). Chunks are video-only and "
                         f"the audio is muxed once at the end. More is not "
                         f"faster -- see default_jobs().")
    ap.add_argument("--x264-threads", type=int, default=2)
    ap.add_argument("--retrack", action="store_true",
                    help="Re-run detection even if a cached track covers this "
                         "range")
    ap.add_argument("--slide-crop", default=None,
                    help="Content box in the screen capture as W:H:X:Y. "
                         "Default: detected with cropdetect. 'none' disables "
                         "the detection and uses the frame as captured.")
    ap.add_argument("--transition", default="dissolve",
                    choices=["cut", "dissolve", "dip", "wipe"],
                    help="How one scene becomes the next. 'dissolve' is the "
                         "brand spec's 4-6 frame crossfade; 'wipe' is a "
                         "soft-edged travelling edge; 'dip' fades out to "
                         "the field colour and back in, never showing both "
                         "scenes at once; 'cut' is no transition at all.")
    ap.add_argument("--transition-frames", type=int, default=None,
                    help=f"Length of the transition in frames. Default "
                         f"{DISSOLVE_FRAMES} ({DISSOLVE_FRAMES / FPS:.2f}s) "
                         f"for dissolve and dip; a wipe defaults to "
                         f"{WIPE_FRAMES} because one travelling edge needs "
                         f"longer than a crossfade to read as one movement.")
    ap.add_argument("--encoder", default="auto",
                    help="Video encoder for the chunks. Default auto: "
                         "smoke-tests h264_nvenc, then h264_videotoolbox, then "
                         "libx264. On Apple Silicon the hardware path both "
                         "encodes faster and frees the CPU for compositing.")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--preset", default="veryfast")
    ap.add_argument("--scene-b-watermark", action="store_true",
                    help="Put the 60%% corner unitmark on the full-camera "
                         "scene instead of the adaptive footer. What videos "
                         "published before 2026-08-26 look like; the footer "
                         "is the default because the corner mark measured "
                         "invisible against the projection screen.")
    ap.add_argument("--allow-unanonymized", action="store_true")
    args = ap.parse_args()
    # A wipe and a dissolve want different lengths, so the flag defaults to
    # None and the style picks. An explicit --transition-frames still wins.
    if args.transition_frames is None:
        args.transition_frames = (WIPE_FRAMES if args.transition == "wipe"
                                  else DISSOLVE_FRAMES)
    # 'auto' cannot be resolved yet -- it needs the detected slide box, which
    # needs the lecture dir -- so the layout is set below, once. set_layout
    # rebinds brand.SLIDE_WINDOW, brand.SPEAKER_WINDOW and brand.SCENE_A_PLATE
    # together, so the plate and the rectangles it is cut for can never
    # disagree.
    if args.layout != "auto":
        brand.set_layout(args.layout)
    global SCENE_B_FOOTER
    SCENE_B_FOOTER = not args.scene_b_watermark

    p = LecturePaths(args.lecture_dir)
    cam = p.resolve_camera_for_assembly()
    if not os.path.exists(cam):
        raise SystemExit(
            f"no camera to render from: {cam} does not exist.\n"
            f"resolve_camera_for_assembly() picks {os.path.basename(p.camera_anon)} "
            f"when it exists and falls back to {os.path.basename(p.camera_muted)}, "
            f"and neither is here. Run the audio and face_anon stages, or copy "
            f"their output into the lecture directory.\n"
            f"(--allow-unanonymized only waives the ANONYMIZATION check; it "
            f"cannot conjure a file. Without this check the readers produce "
            f"zero frames and the failure surfaces much later, as a concat "
            f"error on empty chunks.)")
    if cam == p.camera_muted and not args.allow_unanonymized:
        raise SystemExit(
            f"refusing to render: {p.camera_anon} does not exist, so faces are "
            f"NOT anonymized. Scene B shows the camera FULL FRAME, which makes "
            f"every student in it larger.\n"
            f"Run face_anon first, or pass --allow-unanonymized.")

    if not os.path.exists(p.scenes):
        raise SystemExit(f"no {p.scenes}; run "
                         f"'python -m src.video.scenes --lecture-dir {p.dir}'")
    with open(p.scenes) as f:
        doc = json.load(f)
    scenes = doc["scenes"]

    total = get_duration(p.screen_sync)
    dur = args.duration if args.duration else total - args.start
    dur = min(dur, total - args.start)

    meta = brand.lecture_meta(p.metadata, term=args.term,
                              title=args.course_title)
    print(f"[layout] {meta['course_code']} / {meta['lecture']} / {meta['term']}")

    # Once, here, rather than per chunk: cropdetect is a decode, and every
    # worker deriving the same box independently would be twelve of them.
    if args.slide_crop == "none":
        slide_box = None
    elif args.slide_crop:
        slide_box = tuple(int(v) for v in args.slide_crop.split(":"))
    else:
        # No must_hold_at any more: question cards no longer go through the
        # slide window at all -- load_cards draws them full frame -- so a crop
        # derived from the deck cannot clip them. That check existed only
        # because a single crop had to suit both, and now it does not.
        slide_box = detect_content_box(slide_source(p), total)

    if args.layout == "auto":
        brand.set_layout(pick_layout(slide_box))

    out = args.out or p.layout
    inside = [s for s in scenes
              if s["end"] > args.start and s["start"] < args.start + dur]
    print(f"[layout] {args.start:.0f}s +{dur:.0f}s covers {len(inside)} scene(s): "
          + ", ".join(scene_name(s["scene"]) for s in inside))

    if args.jobs > 1:
        render_parallel(p, scenes, args.start, dur, out, meta, args.jobs,
                        crf=args.crf, preset=args.preset,
                        crop_h=args.pip_crop_h, crop_y=args.pip_crop_y,
                        pan_sigma=args.pan_sigma, retrack=args.retrack,
                        threads=args.x264_threads, plate_dir=args.plate_dir,
                        slide_box=slide_box, encoder=args.encoder,
                        style=args.transition, tframes=args.transition_frames)
    else:
        render(p, scenes, args.start, dur, out, meta, crf=args.crf,
               preset=args.preset, crop_h=args.pip_crop_h,
               crop_y=args.pip_crop_y, pan_sigma=args.pan_sigma,
               retrack=args.retrack, plate_dir=args.plate_dir,
               slide_box=slide_box, encoder=args.encoder,
               style=args.transition, tframes=args.transition_frames)


if __name__ == "__main__":
    main()
