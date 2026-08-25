"""Visual measurement: is there a usable slide track, and a usable camera.

Both streams are read off the keyframe grid (see media.iter_frames) rather than
a full decode, which on Panopto output is a uniform sample roughly every 2.4
seconds for a twentieth of the CPU. Every number below is a rate or a fraction
over minutes, so that grid is far finer than anything being measured.

The slide-change count is deliberately NOT ffmpeg's freezedetect, even though
src/video/scenes.py uses freezedetect for the same underlying question. scenes
needs exact interval boundaries to cut on and can afford a full decode of one
lecture; the scanner needs a count over a semester and cannot. Comparing
successive keyframes gives the count directly, and the threshold below is
calibrated so that compression noise and a moving mouse pointer do not register
as a new slide.

The black measurement matters more than it looks. sync.py trims the screen's
black lead only insofar as the camera/screen duration difference covers it; a
lecture where the black outlasts that difference publishes minutes of black,
and verify.py passes it, because black encodes perfectly well. `black_lead_s`
and the resulting `sync_risk` are how the scanner catches that before anyone
spends an encode on it.
"""

import numpy as np

from src.scan.media import iter_frames

# Sampling geometry. Coarse for the screen -- a slide change is a global event
# -- and finer for the camera, because sharpness measured at 160x90 is mostly
# measuring the scaler.
SCREEN_W, SCREEN_H = 160, 90
CAMERA_W, CAMERA_H = 480, 270

# A frame is black if it is dark on average AND flat. The second half matters:
# a dark slide with white text is not a black frame, and only the variance
# tells them apart.
BLACK_MEAN_MAX = 16.0
BLACK_STD_MAX = 6.0

# The leading black run is measured as "still 90% black up to here", so that a
# single lit frame partway through a ten-minute dark lead does not end it.
BLACK_LEAD_PURITY = 0.90

# ...but the lead must also be LEADING, and purity alone cannot enforce that:
# a cumulative mean dips below 90% when the lecture starts and climbs back
# above it if the screen goes dark again later, which put the lead of a
# synthetic 4.5-minute black tail at the whole file. The lead therefore ends,
# permanently, at the first stretch of light this long. Lecture 12's dark lead
# is interrupted by exactly one keyframe of light at 131.8s (measured on
# data/15210-lecture12/screen.mp4), so the window has to be several keyframes
# wide: 12s is five of them on Panopto's 2.4s GOP grid, and no lecture opens
# with less than twelve seconds of picture and then goes dark for minutes.
SUSTAINED_LIT_S = 12.0

# A pixel counts as part of the slide's content box if it is lit in at least
# this fraction of frames. Excludes permanent pillarbox bars while tolerating
# the dark lead-in and the occasional dark slide.
LIT_FRACTION = 0.15

# How far above the video's black point a pixel has to sit to count as lit.
# BLACK_MEAN_MAX is a FRAME mean and reading it as a per-pixel threshold --
# which the content box used to do -- says "part of the slide" means "bright",
# so a dark-theme deck contributes only its text: a full-frame 16:9 deck on a
# value-8 background measured 2.50 where the truth is 1.78, i.e. read as more
# broken than a genuine 4:3 pillarbox.
#
# The threshold is relative to the black point rather than absolute because
# ffmpeg's `format=gray` does NOT range-convert (verified: a limited-range
# source with luma 16 decodes to 16, not 0), so black is 0 on one lecture and
# 16 on the next and a fixed number cannot serve both. Those two are the only
# black points a video file has, so the level is snapped to whichever it is
# near rather than taken literally -- a dark grey at 8 is not a black point in
# either convention, it is a design choice, and it belongs to the slide.
#
# Temporal variation was the other candidate and does NOT work here: on
# 17-635 lecture 13 the pillarbox bars are not static at all -- the capture
# goes full-width for about 9% of its frames -- so bar pixels change 45 times
# over the lecture and any "it moved, so it is content" rule swallows the
# bars and loses the very case this measurement exists for.
BLACK_POINTS = (0.0, 16.0)
BLACK_POINT_TOL = 4.0
PIXEL_LIT_DELTA = 6.0

# Mean absolute inter-frame difference, 0-255, above which the slide changed.
# Calibrated on the two reference lectures: their inter-keyframe difference
# averages 0.89 with a long tail to 129 at real transitions. 2.5 sits well
# clear of pointer movement and encoder noise while still catching a build
# step that only adds a bullet.
SLIDE_CHANGE_DIFF = 2.5

# Margin, in seconds, by which the camera/screen duration difference must
# exceed the screen's black lead for sync_risk to read a clean 1.0. Lecture 12
# clears it by 89.7s, which CLAUDE.md already flags as uncomfortably close.
SYNC_COMFORT_S = 120.0


def _laplacian_var(frame):
    """Variance of the 4-neighbour Laplacian -- the standard focus measure.

    Done in numpy rather than cv2.Laplacian so that a scan of a few hundred
    lectures does not import OpenCV in every worker for one convolution.
    Always measured at CAMERA_W x CAMERA_H so the number is comparable across
    lectures of different source resolutions.
    """
    f = frame.astype(np.float32)
    lap = (4.0 * f[1:-1, 1:-1] - f[:-2, 1:-1] - f[2:, 1:-1]
           - f[1:-1, :-2] - f[1:-1, 2:])
    return float(lap.var())


def _black_point(pixel_min):
    """The level THIS file codes black at, from its darkest pixels.

    A video has exactly two black points: 0 if it is tagged full range, 16 if
    limited, and ffmpeg's gray output preserves whichever it is rather than
    normalising (measured). So the darkest thing in the file is snapped to the
    nearer of the two, and anything that sits between them -- a deck whose
    background is a dark grey rather than black -- is content, not black, and
    correctly leaves the black point at 0.

    Taken as a low percentile of the per-pixel minimum rather than the outright
    minimum, so one undershooting pixel of compression noise cannot decide it.
    """
    ref = float(np.percentile(pixel_min, 1))
    return 16.0 if abs(ref - 16.0) <= BLACK_POINT_TOL else 0.0


def _leading_black(black, step):
    """How many frames of black the file OPENS with. Never more.

    This is the quantity sync.py has to remove, so both directions of error
    are expensive, and both have been made:

    * Stopping at the first non-black frame underestimates it. Lecture 12's
      screen is dark for 629s but flickers once at 131.8s, and that rule put
      the lead at 132s -- five minutes of black published under a check that
      said the trim covered it.
    * Trusting the 90% purity rule alone overestimates it, without bound. The
      cumulative mean of the black mask is not monotonic: it falls when the
      lecture starts and rises again wherever the screen goes dark later, so
      the last index satisfying it can land anywhere in the file. On a
      synthetic screen that is black for its last 4.5 minutes the "lead" came
      out as the entire file, which would have reported a sync deficit of
      about a thousand seconds against a real one of none.

    The fix is to bound the search first and apply purity second: the lead
    cannot extend past the first sustained stretch of light, where sustained
    is SUSTAINED_LIT_S -- long enough that lecture 12's one-keyframe flicker
    does not end it, short enough that no real lecture opening is missed. The
    window is also capped at a quarter of the file so a short clip (a test
    fixture, a five-minute recitation) still terminates somewhere sensible.
    """
    n = int(black.size)
    if n == 0 or not black[0]:
        return 0
    lit = ~black
    span = max(2, int(round(SUSTAINED_LIT_S / step))) if step > 0 else 2
    span = max(2, min(span, max(2, n // 4)))
    end = n
    if span <= n:
        runs = np.convolve(lit.astype(np.int32),
                           np.ones(span, np.int32), mode="valid")
        starts = np.flatnonzero(runs == span)
        if starts.size:
            end = int(starts[0])
    head = np.flatnonzero(black[:end])
    if not head.size:
        return 0
    lead = int(head[-1]) + 1

    # Purity, now applied only inside the leading region. A lead that is
    # mostly light -- a screen that strobes on and off for ten minutes -- is
    # not a black lead, and the trim must not be told that it is.
    cum = np.cumsum(black[:lead]) / np.arange(1, lead + 1)
    ok = np.flatnonzero((cum >= BLACK_LEAD_PURITY) & black[:lead])
    return int(ok[-1]) + 1 if ok.size else 0


def measure_screen(path, duration_s):
    """Slide-track metrics: how often it changes, how dark, what shape."""
    m = {}
    means, stds, diffs = [], [], []
    prev = None
    # How OFTEN each pixel is lit, not how bright it ever got. A permanent
    # pillarbox bar is black in every frame; a single full-width flash would
    # defeat a max-over-time union, and did -- 17-635 lecture 13's 4:3 deck
    # measured as a clean 16:9 until this became a frequency count.
    #
    # Counted against both candidate black points, because which one applies
    # is only known once the whole file has been seen and this is a single
    # streaming pass. Two 160x90 counters is a rounding error next to holding
    # the frames.
    lit_counts = {bp: None for bp in BLACK_POINTS}
    pixel_min = None
    for _, f in iter_frames(path, SCREEN_W, SCREEN_H):
        fl = f.astype(np.float32)
        means.append(float(fl.mean()))
        stds.append(float(fl.std()))
        if prev is not None:
            diffs.append(float(np.mean(np.abs(fl - prev))))
        pixel_min = fl if pixel_min is None else np.minimum(pixel_min, fl)
        for bp in BLACK_POINTS:
            lit = fl > (bp + PIXEL_LIT_DELTA)
            lit_counts[bp] = (lit.astype(np.int32) if lit_counts[bp] is None
                              else lit_counts[bp] + lit)
        prev = fl
    n = len(means)
    if n < 2:
        return m

    means = np.asarray(means)
    stds = np.asarray(stds)
    diffs = np.asarray(diffs)
    step = duration_s / n if duration_s > 0 else 2.4
    m["screen_frames_sampled"] = n
    m["screen_sample_step_s"] = float(step)

    black = (means <= BLACK_MEAN_MAX) & (stds <= BLACK_STD_MAX)
    m["screen_black_pct"] = float(black.mean() * 100.0)
    # The leading black -- the run sync.py is trying to remove. See
    # _leading_black for why it is neither "up to the first lit frame" nor
    # "as far as the purity rule reaches".
    lead = _leading_black(black, step)
    m["black_lead_s"] = float(lead * step)

    changes = diffs > SLIDE_CHANGE_DIFF
    hours = (duration_s or n * step) / 3600.0
    m["slide_change_count"] = int(changes.sum())
    m["slide_change_per_hour"] = float(changes.sum() / hours) if hours > 0 else 0.0

    # Longest stretch with no change, ignoring any leading black -- twenty
    # minutes of black before the lecture is a sync problem, already counted,
    # not a slide that nobody advanced.
    idx = np.flatnonzero(changes)
    idx = idx[idx >= lead]
    if lead >= n - 1:
        # Nothing survives the lead, so there is no slide track to measure and
        # the subtraction below would report 0.0 -- a perfect score for a
        # screen that never showed anything. Report the whole sampled span:
        # a wholly black screen is the worst dead slide there is, not the
        # absence of one.
        m["longest_static_slide_s"] = float(n * step)
    elif idx.size:
        gaps = np.diff(np.concatenate(([lead], idx, [n - 1])))
        m["longest_static_slide_s"] = float(gaps.max() * step)
    else:
        m["longest_static_slide_s"] = float((n - lead) * step)

    # Content box, from the pixels that carry slide rather than surround. A
    # 4:3 deck pillarboxed into a 16:9 frame leaves permanent black bars, and
    # layout._slide_filter letterboxes it rather than cropping -- correctly,
    # but the slide then uses about three quarters of the window, so it is
    # worth scoring. 17-635 lecture 13 is exactly this case
    # (crop=1440:1080:240:0).
    #
    # "Carries slide" is "sits above the video's own black point", not "is
    # bright": the distinction is what keeps a dark-theme deck from measuring
    # as a sliver of text. See BLACK_POINTS.
    if pixel_min is not None:
        lit_count = lit_counts[_black_point(pixel_min)]
        often = lit_count >= (n * LIT_FRACTION)
        lit_cols = np.flatnonzero(often.any(axis=0))
        lit_rows = np.flatnonzero(often.any(axis=1))
        if lit_cols.size and lit_rows.size:
            w_px = lit_cols[-1] - lit_cols[0] + 1
            h_px = lit_rows[-1] - lit_rows[0] + 1
            # Back out of the sampling geometry into the source's pixel aspect.
            aspect = (w_px / SCREEN_W) / max(h_px / SCREEN_H, 1e-6) * (16.0 / 9.0)
            m["screen_content_aspect"] = float(aspect)
            target = 16.0 / 9.0
            m["screen_aspect"] = float(
                max(0.0, 1.0 - abs(aspect - target) / target))
    return m


def measure_camera(path, duration_s):
    """Camera metrics: is it sharp, is it exposed, and does it move.

    One decode pass. The Laplacian runs on every sampled frame rather than a
    subset -- at 480x270 it is about a millisecond a frame, so skipping frames
    would save a couple of seconds against a decode that costs eight.
    """
    m = {}
    sharp, crushed, blown, contrast, motion = [], [], [], [], []
    prev = None
    for _, f in iter_frames(path, CAMERA_W, CAMERA_H):
        sharp.append(_laplacian_var(f))
        crushed.append(float(np.mean(f < 8)))
        blown.append(float(np.mean(f > 247)))
        p5, p95 = np.percentile(f, [5, 95])
        contrast.append(float(p95 - p5))
        if prev is not None:
            motion.append(float(np.mean(np.abs(f.astype(np.float32)
                                               - prev.astype(np.float32)))))
        prev = f

    if not sharp:
        return m
    m["camera_frames_sampled"] = len(sharp)
    m["camera_sharpness"] = float(np.median(sharp))
    m["camera_motion"] = float(np.median(motion)) if motion else 0.0

    # Exposure: half clipping, half contrast. A silhouetted lecturer in front
    # of a lit projector screen fails both halves at once, which is the point.
    clip = float(np.mean(crushed) + np.mean(blown))
    clip_score = max(0.0, 1.0 - clip / 0.15)
    contrast_score = min(1.0, float(np.median(contrast)) / 128.0)
    m["camera_exposure"] = float(0.5 * clip_score + 0.5 * contrast_score)
    m["camera_crushed_pct"] = float(np.mean(crushed) * 100.0)
    m["camera_blown_pct"] = float(np.mean(blown) * 100.0)
    return m


def sync_risk(camera_duration, screen_duration, black_lead_s):
    """How much room sync.py has to remove the screen's black lead.

    The duration difference is what the alignment removes; the black lead is
    what has to fit inside it. Positive margin means the published video starts
    on the lecture. Negative means it starts on black, and nothing downstream
    notices -- verify.py checks duration, size and bytes-per-second, all of
    which a black segment satisfies.

    Returns (score 0..1, margin in seconds).
    """
    if not camera_duration or not screen_duration:
        return None, None
    delta = screen_duration - camera_duration
    margin = delta - (black_lead_s or 0.0)
    return float(max(0.0, min(1.0, margin / SYNC_COMFORT_S))), float(margin)
