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

# A pixel counts as part of the slide's content box if it is lit in at least
# this fraction of frames. Excludes permanent pillarbox bars while tolerating
# the dark lead-in and the occasional dark slide.
LIT_FRACTION = 0.15

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


def measure_screen(path, duration_s):
    """Slide-track metrics: how often it changes, how dark, what shape."""
    m = {}
    means, stds, diffs = [], [], []
    prev = None
    # How OFTEN each pixel is lit, not how bright it ever got. A permanent
    # pillarbox bar is black in every frame; a single full-width flash would
    # defeat a max-over-time union, and did -- 17-635 lecture 13's 4:3 deck
    # measured as a clean 16:9 until this became a frequency count.
    lit_count = None
    for _, f in iter_frames(path, SCREEN_W, SCREEN_H):
        fl = f.astype(np.float32)
        means.append(float(fl.mean()))
        stds.append(float(fl.std()))
        if prev is not None:
            diffs.append(float(np.mean(np.abs(fl - prev))))
        lit = (fl > BLACK_MEAN_MAX)
        lit_count = lit.astype(np.int32) if lit_count is None else lit_count + lit
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
    # The leading black -- the run sync.py is trying to remove -- is not
    # simply "frames until the first non-black one". Lecture 12's screen is
    # dark for 629s but flickers once at 132s, and taking the first non-black
    # frame put the lead at 132s: a five-minute underestimate of exactly the
    # quantity that decides whether the published video opens on black.
    # Instead: the furthest point up to which the frames are still 90% black.
    lead = 0
    if black.any() and black[0]:
        cum = np.cumsum(black) / np.arange(1, n + 1)
        # Anchored on an actually-black frame. Purity alone is not enough: it
        # tolerates 10% non-black by construction, so it keeps walking past
        # the end of the dark lead into the lecture and reported 697s where
        # the whole file only holds 628s of black.
        mostly = np.flatnonzero((cum >= BLACK_LEAD_PURITY) & black)
        lead = int(mostly[-1]) + 1 if mostly.size else 0
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
    if idx.size:
        gaps = np.diff(np.concatenate(([lead], idx, [n - 1])))
        m["longest_static_slide_s"] = float(gaps.max() * step)
    else:
        m["longest_static_slide_s"] = float((n - lead) * step)

    # Content box, from the pixels that are ever lit. A 4:3 deck pillarboxed
    # into a 16:9 frame leaves permanent black bars, and layout._slide_filter
    # letterboxes it rather than cropping -- correctly, but the slide then uses
    # about three quarters of the window, so it is worth scoring. 17-635
    # lecture 13 is exactly this case (crop=1440:1080:240:0).
    if lit_count is not None:
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
