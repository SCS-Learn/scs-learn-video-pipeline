"""Decide, per interval, whether the frame is slide-with-instructor or
instructor-with-slide.

    python -m src.video.scenes --lecture-dir data/15210-lecture12 --dry-run
    python -m src.video.scenes --lecture-dir data/15210-lecture12

Writes `scenes.json`: a tiling of [0, duration) into

    pip    the slide fills the 16:9 region, the instructor runs down the rail
    full   the instructor fills the frame, uncropped, and the slide is dropped

Those are the SCS brand package's Scene A and Scene B respectively. The names
here predate the package and every scenes.json already on disk uses them, so
they stay; src/assembly/brand.py holds the one table that maps them, and the
renderer accepts either spelling. "Cut to B when the room is the content and
back to A when the slide is" is exactly what the rules below implement.

The published video used one layout for ninety minutes. When the instructor
steps away to work something on the board, or spends four minutes talking over a
slide that has not changed, that layout stares at a still image while the person
actually teaching is a stripe down the side.

The signal
----------
`freezedetect` over screen_sync.mp4. A slide that has not changed in a while is
exactly what a frozen video is, and ffmpeg will report those intervals for the
cost of one decode pass.

Two things about that are load-bearing:

* It runs on **screen_sync.mp4, never screen_with_cards.mp4**. A question card
  is a static full-frame image, so after cards every card span looks like a
  frozen slide -- and cutting away to the instructor there would hide the card
  that exists for privacy reasons in the first place.
* `freezedetect` logs at **info** level. Run it at `-v error` and a completely
  frozen file reports no freezes at all. CLAUDE.md records the same trap for
  `blackdetect`; it is the same mistake with a different filter.

The rules are asymmetric on purpose: slow to leave the slide (T_ENTER of
stillness before cutting away), instant to return (back on the frame the moment
the slide changes). A viewer wants to see a new slide the instant it is up.

Everything this decides is written to scenes.json with the parameters that
produced it, so a rendered video can be traced back to its cut list, and so the
list can be reviewed -- or hand-edited -- before committing to a 90-minute
encode.
"""

import json
import os
import re
import subprocess

from src.paths import LecturePaths, lecture_parser
from src.sync import get_duration

# How long the slide must sit still before the frame leaves it.
T_ENTER = 25.0
# Shorter than this and a `full` run is not worth the cut; it reverts to pip.
T_MIN_FULL = 12.0
# A new slide is held at least this long before any further cut away from it.
T_MIN_PIP = 6.0
# Fraction of a candidate interval the instructor must actually be speaking.
SPEECH_FRAC = 0.60
# Card spans are widened by this much at each end before they veto a cut.
CARD_MARGIN = 1.5
# Never cut away inside the opening. This is the floor on cutting TO the
# instructor mid-lecture; the opening shot itself is governed by OPEN_MAX
# below, which is a different rule pointing the other way.
LEAD_IN = 30.0

# --- the four rules Phillip asked for on 2026-08-20 -----------------------
# "Open lectures with the instructor view and switch to slides only when their
# content becomes relevant." The lecture used to open on the slide because
# LEAD_IN forbade any cut inside the first 30s, which is the opposite: it
# opened on a title card while a person talked over it. Relevance has a
# signal already in hand -- the first time the slide CHANGES is the first time
# it carries something the viewer has not seen. Capped, because a deck whose
# first change is twenty minutes in should not hold the room that long.
OPEN_MAX = 120.0
# ...and floored, so a deck that advances immediately does not produce a
# two-second opening shot that reads as a mistake.
OPEN_MIN = 12.0

# "Student-question scenes transition directly to the instructor view." The
# card is the privacy substitute for the muted question, so the frame stays on
# it while it is up; the ANSWER belongs on the person giving it. This is how
# long we stay on him afterwards before the ordinary rules resume.
ANSWER_HOLD = 20.0

# "Return to the instructor once slide content is finished and he begins
# summarising."
#
# The obvious signal -- the last slide change -- does not work, and it is
# worth recording why rather than leaving the rule to quietly never fire.
# Measured across the three lectures taken end to end, the deck advances to
# within seconds of the finish every time:
#
#     15-210 lecture 12     last change 4770.1s of 4772.7   tail  2.6s
#     17-635 lecture 13     last change 4924.9s of 4924.9   tail  0.0s
#     17-635 recitation 4   last change 4335.7s of 4341.7   tail  6.0s
#
# Nobody stops advancing slides and then summarises; they summarise over the
# last slide and then stop. So "slide content is finished" is not detectable
# from the screen, and a rule keyed to it would be dead code that reads as
# implemented.
#
# What Phillip actually asked for is an editorial convention -- end the video
# on the person -- so it is implemented as one: the last CLOSE_TAIL seconds go
# to him. Gated on speech, because cutting to a wide shot of a room that has
# started packing up is worse than staying on the slide.
CLOSE_ON_INSTRUCTOR = True
CLOSE_TAIL = 25.0

# "Prevent end-of-slideshow and no-signal footage from appearing before the
# lecture closes on the instructor." A projector switched off, a deck exited
# to the desktop, or a capture that lost signal all end up as a dark or frozen
# tail on the screen stream. Whatever is there, it is not content, and the
# frame should be on the instructor over it.
TAIL_BLACK_PIC_TH = 0.98
TAIL_BLACK_MIN = 2.0
# How long before a slide reference the frame returns to the slide, so the
# viewer is already looking at it when he says "this one".
REF_LEAD = 1.5

# Language that means "the thing I am talking about is ON SCREEN". A full-frame
# shot of the instructor is right while he is reasoning aloud and wrong the
# instant he points at something -- the viewer is then looking at a man gesturing
# at a slide they cannot see.
#
# Deliberately built from phrases that carry a visual referent rather than from
# bare deictics. "this" and "here" alone are far too common in speech about
# algorithms ("this takes log n") to mean anything; it is "this LINE", "look at
# this", "up here" that reliably point at the screen.
REFERENCE_PATTERNS = [
    r"\blook(ing)? at\b", r"\bif you look\b", r"\byou can see\b",
    r"\bas you (can )?see\b", r"\byou'?ll see\b", r"\bwe see\b",
    r"\bon the (screen|slide|board|left|right|top|bottom)\b",
    r"\bthis (slide|line|code|picture|diagram|figure|equation|definition|"
    r"example|function|tree|node|case|expression|term|part|bit)\b",
    r"\bthese (two|three|four|lines|nodes|cases|examples)\b",
    r"\b(up|down|over|right|back) here\b", r"\bhere we (have|see|go)\b",
    r"\bnotice (that|the|how)\b", r"\bhighlighted\b", r"\bshown here\b",
    r"\bpoint(ing)? (at|to)\b", r"\bthe (top|bottom|left|right) (one|side|half)\b",
    r"\bwritten (here|there|down)\b", r"\bread (this|that|it) off\b",
]

FREEZE_NOISE = "-60dB"


def detect_freezes(video, min_seconds=8.0, noise=FREEZE_NOISE, verbose=True):
    """Intervals over which the screen does not change, via ffmpeg.

    min_seconds is deliberately well below T_ENTER: the filter only reports runs
    at least this long, and having the shorter ones in hand is what makes the
    summary honest about how much of the lecture is near the threshold.
    """
    cmd = ["ffmpeg", "-v", "info", "-i", video, "-an",
           "-vf", f"freezedetect=n={noise}:d={min_seconds}", "-f", "null", "-"]
    out = subprocess.run(cmd, capture_output=True, text=True).stderr

    starts = [float(m) for m in re.findall(r"freeze_start:\s*([\d.]+)", out)]
    ends = [float(m) for m in re.findall(r"freeze_end:\s*([\d.]+)", out)]
    if not starts:
        print("[scenes] freezedetect reported nothing. If the screen really "
              "does change constantly that is the answer; if not, the noise "
              f"tolerance ({noise}) may be too strict for this capture.")
        return []
    # A run still open at EOF has a start and no end.
    if len(ends) < len(starts):
        ends.append(get_duration(video))
    spans = [(s, e) for s, e in zip(starts, ends) if e > s]
    if verbose:
        total = sum(e - s for s, e in spans)
        print(f"[scenes] {len(spans)} still stretches >= {min_seconds:g}s, "
              f"{total / 60:.1f} min of screen time")
    return spans


def trailing_dead_screen(video, duration, freezes, scan_seconds=600.0,
                        pic_th=TAIL_BLACK_PIC_TH, min_run=TAIL_BLACK_MIN,
                        verbose=True):
    """When the screen stops carrying anything, through to the end.

    Two ways a lecture's screen dies before the lecture does, and they need
    different filters. A projector switched off or a capture that loses signal
    goes BLACK -- blackdetect finds that. A deck exited to the last slide, or
    to the desktop, does not go black at all; it goes STILL, and the freeze
    intervals already in hand find that. Either way the tail is not content.

    Returns the start of the dead tail, or None. Only a run that reaches the
    end counts: a black slide in the middle of a deck is a design choice, and
    a still stretch in the middle is what the ordinary rules already handle.

    Like every other black measurement in this repo, ffmpeg must run at
    `-v info` -- blackdetect logs its findings at info level, so `-v error`
    reports a completely black file as having no black at all.
    """
    start = max(0.0, duration - scan_seconds)
    cmd = ["ffmpeg", "-v", "info", "-ss", f"{start:.3f}", "-i", video, "-an",
           "-vf", f"blackdetect=d={min_run}:pic_th={pic_th}", "-f", "null", "-"]
    out = subprocess.run(cmd, capture_output=True, text=True).stderr
    runs = [(start + float(a), start + float(b)) for a, b in
            re.findall(r"black_start:\s*([\d.]+)\s+black_end:\s*([\d.]+)", out)]
    # blackdetect does not report a run that is still open at EOF, so a tail
    # that is black all the way out shows up as a start with no end.
    opens = [start + float(m) for m in
             re.findall(r"black_start:\s*([\d.]+)(?!\s+black_end)", out)]

    dead = None
    for a, b in runs:
        if b >= duration - 1.0:
            dead = a
            break
    if dead is None and opens:
        last = max(opens)
        if not runs or last > max(b for _, b in runs) - 1.0:
            dead = last

    # The still-tail case. A freeze that runs to the end and is long enough to
    # matter is a dead screen even though every pixel is lit.
    for a, b in freezes:
        if b >= duration - 1.0 and duration - a >= CLOSE_TAIL:
            dead = a if dead is None else min(dead, a)
            break

    if verbose and dead is not None:
        print(f"[scenes] screen goes dead at {dead:.1f}s and never recovers "
              f"({duration - dead:.1f}s to the end) -- holding the instructor "
              f"over it rather than publishing it")
    return dead


def speech_spans(paths):
    """When the instructor -- not a student -- is talking.

    Student questions are excluded because their audio is muted by design, so an
    interval covered only by them is silence, and cutting to a full-frame shot
    of someone standing quietly is the worst version of this feature.
    """
    p = paths.resolve_transcript_classified()
    if not os.path.exists(p):
        print("[scenes] no classified transcript; skipping the speech gate")
        return None
    with open(p) as f:
        segs = json.load(f)
    return [(s["start"], s["end"]) for s in segs
            if not s.get("is_student_question")]


def reference_times(paths, patterns=None):
    """When the instructor says something that only makes sense on screen.

    Returns the START of every transcript segment containing such a phrase, so
    the frame can be back on the slide before he gets to it.
    """
    p = paths.resolve_transcript_classified()
    if not os.path.exists(p):
        return []
    with open(p) as f:
        segs = json.load(f)
    rx = re.compile("|".join(patterns or REFERENCE_PATTERNS), re.I)
    hits = [s["start"] for s in segs
            if not s.get("is_student_question") and rx.search(s.get("text", ""))]
    print(f"[scenes] {len(hits)} slide reference(s) in {len(segs)} segments")
    return sorted(hits)


def card_spans(paths):
    """Where a question card replaces the screen.

    cards.py writes cards.json once it has run; before that the transcript's own
    flags are the best available answer. Falling back matters because the veto
    these provide is a privacy guarantee -- never cut away from a card -- and it
    should not depend on stage ordering.
    """
    if os.path.exists(paths.cards_manifest):
        with open(paths.cards_manifest) as f:
            return [(c["start"], c["end"]) for c in json.load(f)]
    p = paths.resolve_transcript_classified()
    if not os.path.exists(p):
        return []
    with open(p) as f:
        segs = json.load(f)
    spans = [(s["start"], s["end"]) for s in segs if s.get("is_student_question")]
    if spans:
        print(f"[scenes] no {os.path.basename(paths.cards_manifest)}; using "
              f"{len(spans)} flagged question(s) from the transcript instead")
    return spans


# ---------------------------------------------------------------------------
def _covered(span, spans):
    """Seconds of `span` covered by any of `spans`."""
    s, e = span
    return sum(max(0.0, min(e, b) - max(s, a)) for a, b in spans)


def _subtract(span, blocks):
    """`span` minus every block, as a list of surviving pieces."""
    pieces = [span]
    for a, b in blocks:
        nxt = []
        for s, e in pieces:
            if b <= s or a >= e:
                nxt.append((s, e))
                continue
            if s < a:
                nxt.append((s, a))
            if b < e:
                nxt.append((b, e))
        pieces = nxt
    return pieces


def _union(spans):
    """Merge overlapping/abutting spans into a sorted, disjoint list."""
    out = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def build_scenes(duration, freezes, speech, cards, refs=(), t_enter=T_ENTER,
                 t_min_full=T_MIN_FULL, t_min_pip=T_MIN_PIP,
                 speech_frac=SPEECH_FRAC, card_margin=CARD_MARGIN,
                 lead_in=LEAD_IN, ref_lead=REF_LEAD, open_max=OPEN_MAX,
                 open_min=OPEN_MIN, answer_hold=ANSWER_HOLD,
                 close_on_instructor=CLOSE_ON_INSTRUCTOR,
                 close_tail=CLOSE_TAIL, dead_screen=None):
    """Freeze intervals -> a gapless list of scenes.

    Every filter here fails toward `pip`. Showing the slide is the layout that
    is never wrong, only sometimes dull; showing the instructor when he is
    silent, or over a question card, is actively wrong.

    Four runs are FORCED onto the instructor rather than earned by the rules
    above -- the opening, the beat after each question card, the close, and any
    dead screen at the end. They are forced because each answers a question the
    freeze/speech signals cannot: those signals say when the SLIDE is dull, and
    these four are about when the slide is wrong to show at all. They are
    applied last, and they win, because a rule that can be outvoted by the
    ordinary machinery is not a rule.
    """
    blocked = [(max(0.0, s - card_margin), e + card_margin) for s, e in cards]

    cand = []
    for s, e in freezes:
        start = max(s + t_enter, lead_in)
        if e - start >= t_min_full:
            cand.extend(_subtract((start, e), blocked))

    # A stretch on the instructor ends the moment he points at the slide. The
    # freeze that opened it is still going -- the slide has not changed -- so
    # nothing else would bring the frame back, and he would be gesturing at
    # something the viewer cannot see. Ending early is always safe: pip is the
    # layout that is never wrong.
    refs = sorted(refs)
    full = []
    for s, e in cand:
        cut = next((r - ref_lead for r in refs if s < r - ref_lead < e), None)
        if cut is not None:
            e = cut
        if e - s < t_min_full:
            continue
        if speech is not None and _covered((s, e), speech) / (e - s) < speech_frac:
            continue
        full.append((s, e))
    full.sort()

    # Hold every new slide on screen for a beat before cutting away again. The
    # gap between two `full` runs is a slide CHANGE -- flashing it for two
    # seconds is worse than not cutting at all -- so a short gap is widened by
    # eating into the next full run, never by merging over it.
    held, prev_end = [], 0.0
    for s, e in full:
        s = max(s, prev_end + t_min_pip)
        if e - s >= t_min_full:
            held.append((s, e))
            prev_end = e
    full = held

    # --- the forced runs ---------------------------------------------------
    forced = []

    # 1. Open on the instructor, until the slide first has something new on it.
    #    The first freeze END is the first slide change; with no freezes at all
    #    the deck is moving constantly and the opening cap is the only bound.
    if open_max > 0:
        first_change = min((e for _, e in freezes), default=None)
        stop = open_max if first_change is None else min(first_change, open_max)
        stop = max(stop, open_min)
        if stop >= open_min:
            forced.append((0.0, min(stop, duration)))

    # 2. The beat after a question card belongs to whoever answers it. The
    #    card itself stays on screen -- it is the privacy substitute for the
    #    muted question and dropping it would defeat the point -- so this
    #    starts where the card ends.
    if answer_hold > 0:
        for _, e in cards:
            forced.append((e, min(e + answer_hold, duration)))

    # 3. Close on the instructor. A fixed tail rather than a detected one --
    #    see CLOSE_TAIL. The speech gate is the same one the ordinary rules
    #    use, so a lecture that ends in silence or applause keeps the slide up
    #    rather than cutting to a room of people standing.
    if close_on_instructor and close_tail > 0:
        s0 = max(0.0, duration - close_tail)
        if speech is None or _covered((s0, duration), speech) / max(
                duration - s0, 1e-6) >= speech_frac:
            forced.append((s0, duration))
        else:
            print(f"[scenes] not closing on the instructor: he is speaking "
                  f"for less than {speech_frac:.0%} of the last "
                  f"{close_tail:g}s")

    # 4. Never publish a dead screen. This one deliberately ignores close_min
    #    and the speech gate: a black or abandoned screen is wrong to show for
    #    any length of time, whether or not anyone is talking over it.
    if dead_screen is not None:
        forced.append((max(0.0, dead_screen), duration))

    full = _union(full + [(s, e) for s, e in forced if e > s])

    scenes, t = [], 0.0
    for s, e in full:
        if s > t:
            scenes.append({"start": round(t, 3), "end": round(s, 3),
                           "scene": "pip"})
        scenes.append({"start": round(s, 3), "end": round(min(e, duration), 3),
                       "scene": "full"})
        t = min(e, duration)
    if t < duration:
        scenes.append({"start": round(t, 3), "end": round(duration, 3),
                       "scene": "pip"})
    return [s for s in scenes if s["end"] > s["start"]]


# How far a cut may move to land on a phrase boundary. Wider than this and the
# cut stops being the one the rules chose; narrower and most cuts have no
# boundary to reach.
SNAP_WINDOW = 1.5


def snap_to_phrases(scenes, segments, window=SNAP_WINDOW):
    """Move each cut to the nearest gap between spoken segments.

    "Default to a hard cut on a phrase boundary" -- the cut list is built from
    freeze intervals and speech coverage, which know nothing about where
    sentences end, so left alone a cut lands mid-word about as often as not.
    A gap between two transcript segments IS a phrase boundary, and the
    transcript is already in hand.

    Only the interior cuts move; 0 and the final end are fixed. Each cut is
    moved at most `window` seconds and never past its neighbours, so a snapped
    list still tiles the lecture with the same scenes in the same order.
    """
    if len(scenes) < 2 or not segments:
        return scenes
    gaps = []
    for a, b in zip(segments, segments[1:]):
        if b["start"] > a["end"]:
            gaps.append((a["end"] + b["start"]) / 2)
    if not gaps:
        return scenes
    gaps.sort()

    import bisect
    moved = 0
    out = [dict(sc) for sc in scenes]
    for i in range(1, len(out)):
        t = out[i]["start"]
        j = bisect.bisect_left(gaps, t)
        cands = [g for g in gaps[max(0, j - 1):j + 1] if abs(g - t) <= window]
        if not cands:
            continue
        g = min(cands, key=lambda g: abs(g - t))
        # Never cross a neighbouring boundary, and never invert a scene.
        lo = out[i - 1]["start"] + 0.5
        hi = out[i]["end"] - 0.5
        g = min(max(g, lo), hi)
        if abs(g - t) > 1e-3:
            moved += 1
            out[i]["start"] = round(g, 3)
            out[i - 1]["end"] = round(g, 3)
    print(f"[scenes] snapped {moved} of {len(out) - 1} cut(s) to a phrase "
          f"boundary (within {window:g}s)")
    return out


def summarise(scenes, duration):
    full = [s for s in scenes if s["scene"] == "full"]
    secs = sum(s["end"] - s["start"] for s in full)
    print(f"[scenes] {len(scenes)} scenes, {len(scenes) - 1} cuts, "
          f"{len(full)} on the instructor")
    print(f"[scenes] {secs / 60:.1f} min full ({100 * secs / duration:.1f}% of "
          f"{duration / 60:.1f} min)")
    if full:
        longest = max(full, key=lambda s: s["end"] - s["start"])
        print(f"[scenes] longest full run {longest['end'] - longest['start']:.0f}s "
              f"at {longest['start']:.0f}s")


def main():
    parser = lecture_parser("Decide pip vs full-frame per interval.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the cut list; do not write scenes.json")
    parser.add_argument("--t-enter", type=float, default=T_ENTER)
    parser.add_argument("--t-min-full", type=float, default=T_MIN_FULL)
    parser.add_argument("--t-min-pip", type=float, default=T_MIN_PIP)
    parser.add_argument("--speech-frac", type=float, default=SPEECH_FRAC)
    parser.add_argument("--freeze-noise", default=FREEZE_NOISE,
                        help="freezedetect noise tolerance; loosen if a noisy "
                             "capture never registers as still")
    parser.add_argument("--no-speech-gate", action="store_true")
    parser.add_argument("--no-reference-gate", action="store_true",
                        help="Do not cut back to the slide when he refers to "
                             "something on it")
    parser.add_argument("--ref-lead", type=float, default=REF_LEAD)
    parser.add_argument("--open-max", type=float, default=OPEN_MAX,
                        help="Longest opening shot on the instructor before "
                             "the frame goes to the slide regardless. 0 opens "
                             "on the slide, as it did before.")
    parser.add_argument("--open-min", type=float, default=OPEN_MIN)
    parser.add_argument("--answer-hold", type=float, default=ANSWER_HOLD,
                        help="Seconds on the instructor after a question "
                             "card. 0 returns to the slide instead.")
    parser.add_argument("--no-close-on-instructor", action="store_true",
                        help="End on the slide rather than cutting back to "
                             "him once the deck stops advancing")
    parser.add_argument("--close-tail", type=float, default=CLOSE_TAIL,
                        help="Seconds of closing shot on the instructor")
    parser.add_argument("--keep-dead-screen", action="store_true",
                        help="Publish a black or abandoned screen at the end "
                             "instead of holding the instructor over it")
    args = parser.parse_args()
    p = LecturePaths(args.lecture_dir)

    if not os.path.exists(p.screen_sync):
        raise SystemExit(f"no {p.screen_sync}; run sync first")

    duration = get_duration(p.screen_sync)
    freezes = detect_freezes(p.screen_sync, noise=args.freeze_noise)
    speech = None if args.no_speech_gate else speech_spans(p)
    cards = card_spans(p)
    refs = [] if args.no_reference_gate else reference_times(p)
    dead = None if args.keep_dead_screen else trailing_dead_screen(
        p.screen_sync, duration, freezes)

    scenes = build_scenes(duration, freezes, speech, cards, refs,
                          t_enter=args.t_enter, t_min_full=args.t_min_full,
                          t_min_pip=args.t_min_pip,
                          speech_frac=args.speech_frac,
                          ref_lead=args.ref_lead,
                          open_max=args.open_max, open_min=args.open_min,
                          answer_hold=args.answer_hold,
                          close_on_instructor=not args.no_close_on_instructor,
                          close_tail=args.close_tail, dead_screen=dead)
    # The transcript is the only thing that knows where phrases end.
    tpath = p.resolve_transcript_classified()
    if os.path.exists(tpath):
        with open(tpath) as f:
            scenes = snap_to_phrases(scenes, json.load(f))
    else:
        print("[scenes] no transcript; cuts left where the rules put them "
              "(they will not land on phrase boundaries)")
    summarise(scenes, duration)

    doc = {"duration": round(duration, 3),
           "params": {"t_enter": args.t_enter, "t_min_full": args.t_min_full,
                      "t_min_pip": args.t_min_pip,
                      "speech_frac": None if args.no_speech_gate
                      else args.speech_frac,
                      "ref_lead": None if args.no_reference_gate else args.ref_lead,
                      "freeze_noise": args.freeze_noise,
                      "card_margin": CARD_MARGIN, "lead_in": LEAD_IN,
                      "open_max": args.open_max, "open_min": args.open_min,
                      "answer_hold": args.answer_hold,
                      "close_on_instructor": not args.no_close_on_instructor,
                      "close_tail": args.close_tail,
                      "dead_screen": dead},
           "scenes": scenes}

    if args.dry_run:
        for s in scenes:
            print(f"  {s['start']:8.1f} -> {s['end']:8.1f}  "
                  f"{s['end'] - s['start']:6.1f}s  {s['scene']}")
        return
    with open(p.scenes, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"[scenes] wrote {p.scenes}")


if __name__ == "__main__":
    main()
