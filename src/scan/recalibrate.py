"""Re-fit the rubric's bands to the corpus actually in front of it.

The thresholds in rubric.py were set against two lectures, and rubric.py says
so. Two is not a calibration set, and the consequence showed up the moment a
real semester went through: 42 graded lectures scored between 76 and 97, and
28 of them graded A. A grader that gives everything an A has measured nothing,
however carefully each individual metric was measured.

This module fixes that by re-fitting each metric's band to the observed
distribution, so the scale spans the lectures that exist rather than the two it
was guessed from.

    python -m src.scan --courses-dir data/spring2026 --recalibrate
    python -m src.scan --courses-dir data/spring2026 --recalibrate --apply

The first prints a proposal. The second writes `rubric_overrides.json` next to
rubric.py, which rubric.load_overrides() picks up on the next run.

WHAT THIS TRADES AWAY, because it matters and is easy to forget:

Recalibrated scores are RELATIVE. "97" stops meaning "good in absolute terms"
and starts meaning "near the top of this cohort". A semester of uniformly bad
recordings will still produce an A, because something has to be at the top.
The absolute bands in rubric.py are a judgement about what a watchable lecture
is; these are a judgement about what a typical lecture *here* is. Keep both --
that is why this writes an override file rather than editing the table, and
why every proposal below reports the original alongside the fitted value.

The safe reading: use absolute scores to decide whether to publish at all, and
recalibrated scores to decide what to publish FIRST.

THE THREE WAYS A NAIVE FIT GOES WRONG, all of them seen on the real 42-lecture
run, and what is done about each:

1. **Fitting noise.** `clipped_pct` is 0.0 for nearly every lecture in the
   corpus (p10 = 0, p90 = 2e-05) and the first version of this module duly
   proposed `("ramp", 0.0, 0.0)`. score_metric() guards `good == bad` by
   returning 1.0, so that proposal silently converted a 1.5-weight metric into
   a constant that still carried its weight into the total: an invisible
   thumb on the scale. `screen_height` is 1080 on every lecture; `sync_risk`
   is exactly 1.0 on 90% of them. A metric with no meaningful spread is not
   re-fitted at all -- see MIN_RELATIVE_SPREAD.

2. **Punishing a metric for being fine.** `clipped_pct` at full marks
   everywhere is the CORRECT answer: nobody's audio is clipping. That is good
   news about the corpus, not a broken threshold, and manufacturing a penalty
   for it would mark down lectures with nothing wrong. "Saturated" therefore
   splits in two, and only one half is re-fitted -- see _spread_status().

3. **Condemning a lecture for being unlike its peers.** Fitted purely to raw
   percentiles, `duration_min` came out as band(52.4, 72.2, 78.7, 81.3),
   which scores a perfectly good 65-minute lecture at 0.6 for no reason
   except that this corpus happens to be 72-to-80-minute lectures. Every fit
   is therefore held to ABSOLUTE_FLOOR: a value the absolute rubric calls
   perfect keeps at least half marks under the fitted scale. Cohort fitting
   may re-order lectures; it may not fail one.

And because the whole point of the exercise is discrimination, validate()
re-scores the cohort under the proposal and prints the before/after spread.
A proposal that does not widen the spread has not earned its keep, and
render() says so in as many words.
"""

import json
import math
import os

import numpy as np

from src.scan import rubric
# score's dimension arithmetic is reused rather than reimplemented in
# validate(): a second copy of the weighting would drift from the grader's
# and the validation would then be measuring the wrong thing.
from src.scan import score as scoring

# A metric needs at least this many measurements before its distribution is
# worth fitting to. Below it the percentiles are noise wearing a number.
MIN_SAMPLES = 12

# A metric is "saturated" when nearly every lecture scores full marks on it: it
# is carrying weight in the total while telling nobody apart. The same in
# reverse is "floored".
SATURATED_AT = 0.95
FLOORED_AT = 0.05
SATURATED_SHARE = 0.80

# How much raw spread a metric must show before its distribution is fitted at
# all, as a fraction of the width of its OWN original scale.
#
# The yardstick has to be the rubric's own band, not the metric's magnitude.
# Relative to its own values, clipped_pct's p10=0 -> p90=2e-05 is an infinite
# spread; relative to the ramp it is scored on (0.5 percentage points wide) it
# is four thousandths of one percent, which is what it actually is. The band
# is the only place anybody has written down what size of change in this
# measurement is supposed to matter, so it is the only honest denominator.
#
# 5% is chosen because at that width the entire p10..p90 range of the cohort
# moves the sub-score by at most 0.05 -- less than the rounding on a printed
# score, and well inside what re-measuring the same lecture would wobble by.
# Fitting to that is fitting to noise.
#
# The failure mode of this choice is a metric whose original band was drawn
# far too wide: its real spread then looks small against it and no fit is
# proposed. That errs towards leaving the absolute rubric alone, which is the
# safe direction, and analyse() still reports the observed percentiles so a
# human can see it and edit rubric.py by hand.
MIN_RELATIVE_SPREAD = 0.05

# The floor promised to a lecture the absolute rubric is happy with.
#
# A fitted scale is allowed to say "this lecture is mid-table for this
# cohort". It is not allowed to say "this lecture fails" about a measurement
# rubric.py awards full marks to -- that is not calibration, it is punishing a
# 65-minute lecture for being in a corpus of 75-minute ones. Half marks is the
# strongest form of the promise that still leaves room to re-order: the fitted
# ends are pulled outwards until the absolute rubric's full-marks value scores
# at least this, which compresses the range without changing anybody's rank.
ABSOLUTE_FLOOR = 0.5

# Every gap in a fitted band -- the two shoulders and the flat top -- must be
# at least this fraction of the band's full width. A shoulder narrower than
# this is a step function wearing a ramp's clothing: a 1% change in the
# measurement swings the sub-score from 1.0 to 0.0, which amplifies noise
# rather than discriminating. A flat top that narrow means p25 and p75 have
# collapsed together, i.e. there was nothing to fit.
MIN_BAND_GAP_FRAC = 0.05

OVERRIDES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "rubric_overrides.json")

_PERCENTILES = (2, 10, 25, 50, 75, 90, 98)
_GRADE_ORDER = ["A", "B", "C", "D", "F", "?"]


def _values(results, metric_id):
    out = []
    for r in results:
        v = (r.get("metrics") or {}).get(metric_id)
        if isinstance(v, bool):
            v = 1.0 if v else 0.0
        if isinstance(v, (int, float)) and math.isfinite(v):
            out.append(float(v))
    return np.asarray(out, dtype=float)


def _subscores(results, metric_id):
    return np.asarray([s for s in
                       ((r.get("subscores") or {}).get(metric_id)
                        for r in results) if s is not None], dtype=float)


def _scale_span(scale):
    """Width of the region the rubric's own scale cares about, or None.

    For a ramp that is bad..good; for a band, hard edge to hard edge. Outside
    it the sub-score is pinned at 0 or 1, so a difference out there is not a
    difference the grader can see.
    """
    kind = scale[0]
    if kind == "ramp":
        return abs(float(scale[2]) - float(scale[1]))
    if kind == "band":
        return abs(float(scale[4]) - float(scale[1]))
    return None


def _spread_status(raw, scale):
    """(has_spread, spread, relative_spread) for one metric's observations.

    This is the test that separates the two kinds of "everything gets full
    marks", which look identical in the sub-scores and mean opposite things:

      * full marks with WIDE raw spread -- the lectures genuinely differ and
        the band is too generous to notice. Re-fit it; that is the whole job.
      * full marks with NO raw spread -- every lecture really is the same, and
        really is fine. clipped_pct is 0.0 everywhere because nobody's audio
        clips; screen_height is 1080 everywhere because Panopto records 1080p.
        Re-fitting invents a difference that does not exist and then charges
        lectures for landing on the wrong side of it.

    Only the first is a broken threshold. The second is a fact about the
    corpus, and the right response to a fact is to report it and move on.
    """
    spread = float(np.percentile(raw, 90) - np.percentile(raw, 10))
    span = _scale_span(scale)
    if not span:
        # Booleans, and anything with a degenerate scale: nothing to fit, so
        # "no spread" is the only answer that keeps propose() from trying.
        return False, spread, 0.0
    rel = abs(spread) / span
    return rel >= MIN_RELATIVE_SPREAD, spread, rel


def _fittable(results):
    """The lectures whose measurements are allowed to define the scale.

    A lecture that failed a hard gate is not a data point about what a
    typical lecture here looks like -- it is a lecture with no camera, or no
    audio, or two streams that are not the same event. Its measurements are
    still real numbers and would still move a percentile, which is how a
    failed download ends up widening the band that every good lecture is then
    graded against. Same for a lecture that produced no score at all.

    Called by analyse(), propose() and validate() rather than left to the
    caller, because __main__ hands over every scanned lecture and the caller
    should not have to know this.
    """
    out = []
    for r in results:
        if r.get("gates_failed"):
            continue
        if "score" in r and r.get("score") is None:
            continue
        out.append(r)
    return out


def analyse(results):
    """Per-metric: how it is distributed, and whether it discriminates."""
    results = _fittable(results)
    report = {}
    for mid, spec in rubric.METRICS.items():
        raw = _values(results, mid)
        subs = _subscores(results, mid)
        if raw.size < MIN_SAMPLES:
            report[mid] = {"n": int(raw.size), "status": "too few samples"}
            continue
        pct = {p: float(np.percentile(raw, p)) for p in _PERCENTILES}
        sat = float(np.mean(subs >= SATURATED_AT)) if subs.size else 0.0
        flo = float(np.mean(subs <= FLOORED_AT)) if subs.size else 0.0
        has_spread, spread, rel = _spread_status(raw, spec["scale"])
        if not has_spread:
            # Checked before saturation, because a constant metric is always
            # also saturated or floored and the constancy is the real finding.
            status = "constant"
        elif sat >= SATURATED_SHARE:
            status = "saturated"
        elif flo >= SATURATED_SHARE:
            status = "floored"
        else:
            status = "discriminating"
        report[mid] = {
            "n": int(raw.size), "status": status,
            "percentiles": pct,
            "subscore_mean": float(subs.mean()) if subs.size else None,
            "saturated_share": sat, "floored_share": flo,
            "spread": spread, "relative_spread": rel,
            "scale": list(spec["scale"]),
        }
    return report


def _fit_ramp(scale, p):
    """(new_scale, skip_reason, guarded) for one ramp metric.

    p10/p90 rather than min/max so that one broken recording cannot define the
    whole scale, and the direction of the original ramp is preserved: a
    lower-is-better metric (good < bad, e.g. clipped_pct, dead_air_pct,
    student_face_pct) gets bad=p90 and good=p10, so the quietest, cleanest,
    least-student-exposing lecture in the cohort still comes out on top.

    This is the single easiest thing in the module to get silently backwards
    -- backwards, it reads as a plausible band and quietly ranks the most
    student-exposing lecture first -- so the direction is asserted in the
    tests against a fitted lower-is-better cohort rather than eyeballed here.
    """
    bad, good = float(scale[1]), float(scale[2])
    lo_end, hi_end = p[10], p[90]
    if good > bad:
        new_bad, new_good = lo_end, hi_end     # higher is better
    else:
        new_bad, new_good = hi_end, lo_end     # lower is better
    if abs(new_good - new_bad) < 1e-9:
        return None, "cohort has no spread to fit", False

    # ABSOLUTE_FLOOR. `good` is the value rubric.py awards full marks to; if
    # the fitted ramp would score it below the floor, slide the bad end out
    # until it lands exactly on the floor. This only ever widens the ramp, so
    # the cohort's ordering is untouched -- the scores compress towards the
    # top rather than re-arranging.
    at_good = (good - new_bad) / (new_good - new_bad)
    guarded = at_good < ABSOLUTE_FLOOR
    if guarded:
        new_bad = ((good - ABSOLUTE_FLOOR * new_good)
                   / (1.0 - ABSOLUTE_FLOOR))
    if abs(new_good - new_bad) < 1e-9:
        return None, "the floor guard collapsed the ramp", guarded
    return ("ramp", new_bad, new_good), None, guarded


def _fit_band(scale, p):
    """(new_scale, skip_reason, guarded) for one band metric.

    THE DECISION: the original hard edges are kept and only lo/hi move.

    A band has two different kinds of number in it. hard_lo/hard_hi are an
    absolute judgement -- "past here the lecture is not usable at all" -- and
    that judgement does not become untrue because this particular semester
    clustered somewhere else. lo/hi are the comfort zone, and THAT is what the
    cohort has something to say about. Fitting all four to percentiles is what
    produced duration_min = band(52.4, 72.2, 78.7, 81.3): a hard failure at 81
    minutes, from a rubric whose own considered view is that anything up to 95
    is fine and only 150 is absurd. Fitting the shoulders and leaving the
    cliffs alone keeps the absolute failure conditions absolute and puts the
    discrimination where it belongs, in the middle of the distribution.

    p25/p75 for the flat top, not p10/p90: with the hard edges pinned, wider
    percentiles leave four fifths of the cohort tied at 1.0 and the metric
    barely moves. The remaining bite is then bounded by ABSOLUTE_FLOOR below.
    """
    hard_lo, lo, hi, hard_hi = [float(x) for x in scale[1:]]
    new_lo, new_hi = p[25], p[75]

    # ABSOLUTE_FLOOR on both shoulders: the values rubric.py calls the edge of
    # perfect (lo and hi) must still score at least half under the fit.
    capped_lo = min(new_lo, hard_lo + (lo - hard_lo) / ABSOLUTE_FLOOR)
    capped_hi = max(new_hi, hard_hi - (hard_hi - hi) / ABSOLUTE_FLOOR)
    guarded = (capped_lo != new_lo) or (capped_hi != new_hi)
    new_lo, new_hi = capped_lo, capped_hi

    new = ("band", hard_lo, new_lo, new_hi, hard_hi)
    if new_lo <= hard_lo or new_hi >= hard_hi:
        # The middle of the cohort is outside the rubric's own hard edges,
        # i.e. every lecture here is already scoring zero on this metric. The
        # honest reading is that the corpus is bad on this axis, not that the
        # threshold is; moving a hard edge is a judgement for rubric.py and a
        # human, not something a percentile gets to do quietly.
        return (None, "the cohort sits outside the absolute hard edges -- "
                "moving those is a rubric.py decision, not a fit", guarded)
    if not (new_lo < new_hi):
        return None, "fitted band inverts or collapses", guarded
    # Every gap non-trivial, or the band is a step function (see
    # MIN_BAND_GAP_FRAC).
    width = hard_hi - hard_lo
    gaps = (new_lo - hard_lo, new_hi - new_lo, hard_hi - new_hi)
    if min(gaps) < MIN_BAND_GAP_FRAC * width:
        return (None, "a band segment came out too narrow to be a ramp",
                guarded)
    return new, None, guarded


def propose(results, only_broken=True):
    """New band edges fitted to the cohort. Returns {metric_id: proposal}."""
    stats = analyse(results)                 # analyse() drops failed gates
    out = {}
    for mid, s in stats.items():
        status = s.get("status")
        if status in (None, "too few samples"):
            continue
        # Bugs 1 and 2: no meaningful spread means either the measurement
        # cannot discriminate or the cohort genuinely is uniform, and in
        # neither case is there a distribution to fit. Left alone, reported
        # by render().
        if status == "constant":
            continue
        if only_broken and status == "discriminating":
            continue
        spec = rubric.METRICS[mid]
        kind = spec["scale"][0]
        p = s["percentiles"]
        if kind == "ramp":
            new, why, guarded = _fit_ramp(spec["scale"], p)
        elif kind == "band":
            new, why, guarded = _fit_band(spec["scale"], p)
        else:
            continue                        # booleans have nothing to fit
        if new is None:
            s["skipped"] = why
            continue
        out[mid] = {"from": list(spec["scale"]), "to": list(new),
                    "status": status, "n": s["n"], "guarded": guarded}
    return out


# --------------------------------------------------------------------------
# Validation: did the proposal actually buy any discrimination?
# --------------------------------------------------------------------------

def _score_cohort(results):
    """Re-score every lecture under whatever scales rubric holds right now.

    Gates and coverage are taken from the stored result: neither depends on
    the scales, and re-deriving them would need the probe info, which the
    cached scan does not keep. Everything that does depend on the scales is
    recomputed through score.py's own arithmetic.
    """
    out = []
    for r in results:
        metrics = r.get("metrics") or {}
        _, dims = scoring._dimension_scores(metrics)
        total = scoring._weighted_total(dims)
        coverage = sum(d["coverage"] * d["weight"] for d in dims.values())
        gates = r.get("gates_failed") or []
        if total is None:
            grade = "?"
        elif coverage < scoring.MIN_GRADE_COVERAGE and not gates:
            grade = "?"                 # incomplete scan, as score.py has it
        else:
            grade = rubric.grade_for(total, gates)[0]
        out.append({"key": r.get("key"), "score": total, "grade": grade})
    return out


def _spread_stats(rows):
    scores = [r["score"] for r in rows if r["score"] is not None]
    hist = {}
    for r in rows:
        hist[r["grade"]] = hist.get(r["grade"], 0) + 1
    if not scores:
        return {"n": 0, "grades": hist}
    arr = np.asarray(scores, dtype=float)
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
        "stdev": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "range": float(arr.max() - arr.min()),
        "grades": hist,
        "distinct_grades": len([g for g, n in hist.items() if n and g != "?"]),
    }


def validate(results, proposals):
    """Re-score the cohort under `proposals` and compare the spread.

    Recalibration exists to make a semester of lectures tell each other apart.
    That is a claim about the resulting distribution, and it is cheap to
    check: score everybody twice and look at the spread. Without this the
    module can only assert that it moved some numbers around.

    stdev is the headline because it is what "all A" actually means -- 42
    lectures inside 21 points, most of them inside 8. range is reported beside
    it because one salvageable outlier can widen the range while the body of
    the cohort stays glued together, and the grade histogram is reported
    because that is the thing a human reads off the report: five grades in use
    is a working grader, one grade in use is not.
    """
    results = _fittable(results)             # the cohort that was fitted to
    before = _score_cohort(results)
    saved = {mid: rubric.METRICS[mid] for mid in proposals}
    try:
        for mid, pr in proposals.items():
            rubric.METRICS[mid] = dict(rubric.METRICS[mid],
                                       scale=tuple(pr["to"]))
        after = _score_cohort(results)
    finally:
        rubric.METRICS.update(saved)

    b, a = _spread_stats(before), _spread_stats(after)
    improved = (a.get("stdev", 0.0) > b.get("stdev", 0.0) + 1e-9
                and a.get("range", 0.0) >= b.get("range", 0.0) - 1e-9)
    return {"before": b, "after": a, "improved": bool(improved),
            "rows_before": before, "rows_after": after,
            "n_proposed": len(proposals)}


def _fmt_grades(hist):
    parts = [f"{g}={hist[g]}" for g in _GRADE_ORDER if hist.get(g)]
    parts += [f"{g}={n}" for g, n in sorted(hist.items())
              if g not in _GRADE_ORDER]
    return ", ".join(parts) or "none"


def _validation_lines(v, width):
    b, a = v["before"], v["after"]
    if not b.get("n"):
        return ["No scored lectures to validate against."]
    lines = ["-" * width,
             f"VALIDATION -- {b['n']} lecture(s) re-scored under the "
             f"proposal", "-" * width,
             "                 min   median      max    stdev    range",
             f"  before    {b['min']:7.1f}  {b['median']:7.1f}  "
             f"{b['max']:7.1f}  {b['stdev']:7.2f}  {b['range']:7.1f}",
             f"  after     {a['min']:7.1f}  {a['median']:7.1f}  "
             f"{a['max']:7.1f}  {a['stdev']:7.2f}  {a['range']:7.1f}",
             "",
             f"  grades before: {_fmt_grades(b['grades'])}",
             f"  grades after:  {_fmt_grades(a['grades'])}",
             ""]
    if not v["n_proposed"]:
        lines.append("Nothing was proposed, so nothing changed.")
    elif v["improved"]:
        lines += [
            f"The proposal widens the spread: stdev {b['stdev']:.2f} -> "
            f"{a['stdev']:.2f}, range {b['range']:.1f} ->",
            f"{a['range']:.1f}, {b['distinct_grades']} grade(s) in use -> "
            f"{a['distinct_grades']}. That is the point of it."]
    else:
        lines += [
            "*** WARNING: THIS PROPOSAL DOES NOT IMPROVE DISCRIMINATION ***",
            f"    stdev {b['stdev']:.2f} -> {a['stdev']:.2f}, range "
            f"{b['range']:.1f} -> {a['range']:.1f}.",
            "    Recalibration exists to tell these lectures apart, and this",
            "    set of scales does not do that better than the absolute",
            "    rubric already does. Do NOT --apply it: it would trade away",
            "    the absolute meaning of a score and buy nothing. Scan more",
            "    lectures, run a deeper tier so more metrics are measured, or",
            "    edit the bands in rubric.py by hand.",
        ]
    return lines


def _round(scale):
    return [round(x, 4) if isinstance(x, float) else x for x in scale]


def render(results, proposals, width=78):
    stats = analyse(results)
    fitted = _fittable(results)
    header = f"RUBRIC RECALIBRATION over {len(fitted)} lecture(s)"
    if len(fitted) != len(results):
        header += f" ({len(results) - len(fitted)} skipped: gate failed)"
    lines = ["=" * width, header, "=" * width, ""]

    broken = sorted(m for m, s in stats.items()
                    if s.get("status") in ("saturated", "floored"))
    constant = sorted(m for m, s in stats.items()
                      if s.get("status") == "constant")
    lines.append(f"{len(broken)} metric(s) are not telling these lectures "
                 f"apart:")
    for mid in broken:
        s = stats[mid]
        lines.append(f"  {rubric.METRICS[mid]['label']} [{mid}] -- "
                     f"{s['status']} ({s['saturated_share']:.0%} at full "
                     f"marks, n={s['n']})")
        p = s["percentiles"]
        lines.append(f"      observed p10={p[10]:.4g} p50={p[50]:.4g} "
                     f"p90={p[90]:.4g}")
        if s.get("skipped"):
            lines.append(f"      NOT re-fitted: {s['skipped']}")
    lines.append("")

    if constant:
        lines.append(f"{len(constant)} metric(s) are constant in this cohort "
                     f"and are left alone:")
        for mid in constant:
            s = stats[mid]
            p = s["percentiles"]
            lines.append(f"  {rubric.METRICS[mid]['label']} [{mid}] -- "
                         f"p10={p[10]:.4g} p50={p[50]:.4g} p90={p[90]:.4g} "
                         f"({s['relative_spread']:.1%} of its own scale)")
        lines += [
            "  Every lecture measures the same, so there is no distribution",
            "  to fit. Where they all measure WELL -- no clipping, 1080p",
            "  capture -- that is good news about the corpus, not a broken",
            "  threshold, and re-fitting it would invent a penalty for",
            "  lectures with nothing wrong with them.",
            ""]

    if not proposals:
        lines.append("No proposals: nothing had a distribution worth "
                     "fitting.")
        lines.append("")
        lines += _validation_lines(validate(results, proposals), width)
        return "\n".join(lines)

    lines.append("Proposed bands, fitted to the observed spread:")
    for mid, pr in sorted(proposals.items()):
        lines.append(f"  {rubric.METRICS[mid]['label']} [{mid}]  n={pr['n']}")
        lines.append(f"      from {_round(pr['from'])}")
        lines.append(f"      to   {_round(pr['to'])}")
        if pr.get("guarded"):
            lines.append("      (widened past the raw percentiles so the "
                         "absolute rubric's")
            lines.append(f"       full-marks value still keeps "
                         f"{ABSOLUTE_FLOOR:.0%})")
    lines += ["",
              "Band fits keep the ORIGINAL hard edges and move only the",
              "comfort zone, and every fit is held to a floor: a value the",
              "absolute rubric calls perfect still scores at least "
              f"{ABSOLUTE_FLOOR:.0%}.",
              "Cohort fitting re-orders lectures; it must never fail one for",
              "being unlike its peers.",
              ""]
    lines += _validation_lines(validate(results, proposals), width)
    lines += ["",
              "These scores become RELATIVE: near the top of THIS",
              "cohort, not good in absolute terms. A semester of uniformly",
              "poor recordings still yields an A. Use absolute scores to",
              "decide whether to publish at all, and recalibrated ones to",
              "decide what to publish first. --apply writes",
              "rubric_overrides.json; delete that file to go back to the",
              "absolute rubric."]
    return "\n".join(lines)


def apply(proposals, path=OVERRIDES_PATH, cohort_size=0):
    payload = {
        "_comment": [
            "Written by src/scan/recalibrate.py --apply. Loaded by",
            "rubric.load_overrides() at import. DELETE THIS FILE to return to",
            "the absolute bands in rubric.py.",
            "Scores computed with these overrides are relative to the cohort",
            "they were fitted on, not absolute judgements of watchability.",
        ],
        "cohort_size": cohort_size,
        "scales": {mid: pr["to"] for mid, pr in proposals.items()},
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path
