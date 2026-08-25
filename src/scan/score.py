"""Turn measurements into a grade, honestly.

The arithmetic is deliberately dull -- every judgement lives in
src/scan/rubric.py -- but three decisions here are worth stating, because they
are what stop the number from lying.

**A tier that did not run must not read as a zero.** Dimensions are averaged
over the metrics actually measured and each reports its own `coverage`. A
probe-only scan gives an audio score of "unknown", not "terrible". Without
this, a fast first pass over a semester would rank every lecture bottom and
look like a result.

**Gates are separate from the score.** A lecture with no instructor in shot can
still score 80 on audio and delivery, and that 80 is true and useless. Gates
answer "publish this at all", the score answers "how good is it", and mixing
them produces a number that means neither.

**Two scores, not one.** `score` is the lecture as recorded; `potential` is the
same lecture after the remediation the pipeline can actually apply. Sorting a
semester by score finds work to do; sorting by potential finds the lectures
that are genuinely not worth the hours. A lecture at 52/78 is quiet and hissy
and worth publishing; one at 52/54 is a monotone talk over a dead slide, and
no encoder setting fixes that.
"""

from src.scan import rubric

# Hard gate thresholds. Kept here rather than in the rubric table because they
# are pass/fail on raw measurements, not scored ramps.
MIN_DURATION_S = 8 * 60
MAX_DURATION_S = 4 * 3600
MIN_SPEECH_PCT = 2.0
MIN_ASR_CONFIDENCE = -1.1
MIN_INSTRUCTOR_IN_FRAME_PCT = 25.0
MAX_DURATION_DELTA_RATIO = 0.25

# Below this weighted coverage, a lecture gets a score but NOT a grade.
#
# Renormalising over measured dimensions is what makes a partial scan
# comparable rather than punished, but taken to its limit it produces a
# confident-looking A from three metrics: a probe-only pass over this corpus
# scored 93.8 at 6% coverage, because duration, resolution and chapter count
# all happened to be fine. A number that good, that early, is worse than no
# number -- somebody would ship on it. So a thin scan reports `incomplete` and
# says what it would take to finish.
MIN_GRADE_COVERAGE = 0.55


def _check_gates(metrics, probe_info):
    """(failed_labels, detail). A failed gate means skip, whatever the score.

    Every gate is skipped when the measurement it needs is absent: a scan that
    never ran the vision tier must not fail `instructor_visible` for want of
    looking. Unmeasured is not the same as bad, and conflating them would make
    a cheap first pass condemn the whole semester.
    """
    failed, detail = [], []

    def gate(gate_id, ok, reason):
        spec = next(g for g in rubric.GATES if g["id"] == gate_id)
        if ok is None:
            detail.append({"id": gate_id, "label": spec["label"],
                           "passed": None, "reason": "not measured"})
            return
        detail.append({"id": gate_id, "label": spec["label"],
                       "passed": bool(ok), "reason": reason})
        if not ok:
            failed.append(spec["label"])

    cam = probe_info.get("camera")
    scr = probe_info.get("screen")
    # Not "None means unmeasured" here, unlike the gates below. A camera that
    # is absent or will not probe IS the finding -- the probe tier always
    # looks, so there is no third state. Treating it as unmeasured let a
    # lecture whose download had failed come back as a provisional 100.0 with
    # no camera at all, which is precisely the kind of confident-looking
    # nonsense the gates exist to stop.
    gate("media_readable", bool(cam and cam.get("has_video")),
         "camera missing or will not decode" if not cam else "ok")
    gate("has_audio", bool(cam and cam.get("has_audio")),
         "camera missing or has no audio stream"
         if not (cam and cam.get("has_audio")) else "ok")

    dur = (cam or {}).get("duration") or metrics.get("duration_s")
    gate("duration_sane",
         None if not dur else (MIN_DURATION_S <= dur <= MAX_DURATION_S),
         f"{(dur or 0) / 60:.0f} min" if dur else "unknown")

    speech = metrics.get("speech_pct")
    gate("not_silent",
         None if speech is None else speech >= MIN_SPEECH_PCT,
         f"{speech:.1f}% of runtime carries speech" if speech is not None else "")

    conf = metrics.get("asr_confidence")
    gate("intelligible",
         None if conf is None else conf >= MIN_ASR_CONFIDENCE,
         f"mean avg_logprob {conf:.2f}" if conf is not None else "")

    inframe = metrics.get("instructor_in_frame_pct")
    gate("instructor_visible",
         None if inframe is None else inframe >= MIN_INSTRUCTOR_IN_FRAME_PCT,
         f"in {inframe:.0f}% of sampled frames" if inframe is not None else "")

    if cam and scr and cam.get("duration") and scr.get("duration"):
        delta = abs(scr["duration"] - cam["duration"])
        ratio = delta / max(cam["duration"], 1.0)
        gate("sync_recoverable", ratio <= MAX_DURATION_DELTA_RATIO,
             f"streams differ by {delta:.0f}s ({ratio:.0%} of runtime)")
    else:
        gate("sync_recoverable", None, "")
    return failed, detail


def _dimension_scores(metrics, remediate=False):
    """Per-dimension 0..1 score plus the coverage it was computed over."""
    subscores, dims = {}, {}
    for dim, meta in rubric.DIMENSIONS.items():
        total_w = got_w = acc = 0.0
        for mid in rubric.dimension_ids(dim):
            spec = rubric.METRICS[mid]
            total_w += spec["weight"]
            value = metrics.get(mid)
            if remediate and spec["fixable"]:
                value = rubric.remediated_value(mid, value)
            sub = rubric.score_metric(mid, value)
            if sub is None:
                continue
            if not remediate:
                subscores[mid] = sub
            got_w += spec["weight"]
            acc += sub * spec["weight"]
        dims[dim] = {
            "score": (acc / got_w) if got_w else None,
            "coverage": (got_w / total_w) if total_w else 0.0,
            "weight": meta["weight"],
        }
    return subscores, dims


def _weighted_total(dims):
    """Weighted mean over the dimensions that have any measurement.

    Renormalising over measured dimensions is what keeps a partial scan
    comparable to a full one instead of silently penalised for its own
    cheapness. `coverage` on the result says how much was actually seen.
    """
    num = den = 0.0
    for d in dims.values():
        if d["score"] is None:
            continue
        num += d["score"] * d["weight"]
        den += d["weight"]
    return (num / den * 100.0) if den else None


def _remediation(metrics, base_subscores, base_score):
    """What each fixable metric is worth, in points of the final score.

    The gain is MEASURED -- re-score the lecture with that one metric
    remediated and take the difference -- rather than derived from the
    weights by hand. An earlier version did the arithmetic itself, dividing
    by the dimension's full weight and multiplying by the dimension's share,
    and that only matches the real total at full coverage: `_dimension_scores`
    normalises by the weights actually MEASURED and `_weighted_total`
    renormalises again over the dimensions that have any measurement.

    On a probe+signal scan the two disagreed badly. The report printed a
    headline gap of +14.7 points and then itemised gains summing to 7.5
    directly beneath it -- on precisely the cheap first-pass scans the tier
    system exists to encourage. Re-scoring costs a few dozen dictionary
    passes and cannot drift from the grader, because it IS the grader.
    """
    out = []
    for mid, spec in rubric.METRICS.items():
        if not spec["fixable"] or mid not in rubric.REMEDIATED:
            continue
        value = metrics.get(mid)
        if value is None:
            continue
        now = base_subscores.get(mid)
        fixed = rubric.remediated_value(mid, value)
        then = rubric.score_metric(mid, fixed)
        if now is None or then is None or then <= now + 1e-6:
            continue
        _, dims_one = _dimension_scores(dict(metrics, **{mid: fixed}),
                                        remediate=False)
        gain = (_weighted_total(dims_one) or 0.0) - (base_score or 0.0)
        if gain <= 1e-6:
            continue
        out.append({
            "metric": mid,
            "label": spec["label"],
            "from": float(value),
            "gain": float(gain),
            "note": REMEDY_NOTE.get(mid, "handled by the pipeline"),
        })
    out.sort(key=lambda r: -r["gain"])
    return out


# What to actually do about each fixable metric. Concrete, because "audio is
# poor" is not an instruction and "run loudnorm" is.
REMEDY_NOTE = {
    "loudness_lufs": "ffmpeg loudnorm to -18 LUFS",
    "loudness_range_lu": "loudnorm; pulls the range into band as a side effect",
    "snr_db": "ffmpeg afftdn denoise pass (~6 dB before artefacts show)",
    "noise_floor_dbfs": "same afftdn pass",
    "level_stability_db": "per-minute gain ride, or re-record with a lapel mic",
    "dead_air_pct": "cut the gaps over 5s; the scan reports where they are",
    "longest_dead_air_s": "cut that one gap",
    "screen_black_pct": "sync.py trims the lead; check its 'trimming a further' line",
    "longest_static_slide_s": "scenes.py cuts to full-frame camera over dead slides",
    "camera_exposure": "a levels curve on the camera before layout",
    "screen_aspect": "already handled -- layout._slide_filter letterboxes 4:3",
    "admin_talk_pct": "cut the admin spans listed in the scan",
    "chapter_count": "generate chapters from the transcript",
}


def evaluate(metrics, probe_info, identity=None, tiers_run=(), warnings=(),
             errors=()):
    """Measurements in, graded result out. The shape report.py consumes."""
    identity = identity or {}
    gates_failed, gate_detail = _check_gates(metrics, probe_info)
    subscores, dims = _dimension_scores(metrics, remediate=False)
    _, dims_fixed = _dimension_scores(metrics, remediate=True)

    score = _weighted_total(dims)
    potential = _weighted_total(dims_fixed)
    if score is not None and potential is not None:
        # Remediation can only help. If the arithmetic says otherwise it is a
        # band whose remediated target sits the wrong side of an edge, and the
        # honest answer is "no improvement", not a negative one.
        potential = max(potential, score)

    coverage = sum(d["coverage"] * d["weight"] for d in dims.values())
    grade, verdict, blurb = rubric.grade_for(score or 0.0, gates_failed)
    if score is None:
        grade, verdict = "?", "unscanned"
        blurb = "Nothing measured."
    elif coverage < MIN_GRADE_COVERAGE and not gates_failed:
        grade, verdict = "?", "incomplete"
        missing = sorted({rubric.METRICS[m]["tier"] for m in rubric.METRICS
                          if m not in subscores},
                         key=rubric.TIERS.index)
        blurb = (f"Only {coverage:.0%} of the rubric was measured -- provisional "
                 f"score, not a grade. Run a deeper tier ({', '.join(missing)}).")
        warnings = list(warnings) + [blurb]
    measured = sum(1 for d in dims.values() if d["score"] is not None)
    return {
        "schema": 1,
        "key": identity.get("key"),
        "dir": identity.get("dir"),
        "course": identity.get("course"),
        "title": identity.get("title"),
        "owner": identity.get("owner"),
        "duration_s": (probe_info.get("camera") or {}).get("duration")
                      or identity.get("duration_s"),
        "scanned_at": identity.get("scanned_at"),
        "tiers_run": list(tiers_run),
        "metrics": metrics,
        "subscores": subscores,
        "dimensions": dims,
        "score": score,
        "potential": potential,
        "grade": grade,
        "verdict": verdict,
        "verdict_blurb": blurb,
        "coverage": coverage,
        "gates_failed": gates_failed,
        "gate_detail": gate_detail,
        "remediation": _remediation(metrics, subscores, score),
        "dimensions_measured": measured,
        "warnings": list(warnings),
        "errors": list(errors),
    }
