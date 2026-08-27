"""Rank a whole semester -- every course, every lecture -- from the scan cache.

    python scripts/semester_report.py
    python scripts/semester_report.py --cache reports/psc-cache --out reports/semester

`scripts/scan_all_courses.sh` fans one Slurm job out per course, and each job
writes its own `_reports/` for its own course. That is the right unit for
"which lecture in 17-635", and the wrong one for the question actually being
asked -- which COURSE is worth starting with -- because thirty separate
reports cannot be compared: each is ranked within itself, and the per-course
means in them are computed over different mixes of failed downloads.

This reads the per-lecture `scan.json` cache the jobs left behind, re-scores
every lecture through the same src.scan.score everything else uses, and
renders one report over the lot plus a course-level league table.

Why it re-scores rather than reading the jobs' reports
------------------------------------------------------
Because the tiers arrive in two passes. `scan_all_courses.sh manifests/ signal`
runs first and each job writes a report at ~44% coverage, which is BELOW the
55% floor at which the scanner is willing to grade -- so those reports are
entirely ungraded, and a course-level mean taken from them is a mean of
provisional numbers. The vision pass then deepens every lecture's scan.json
without rewriting the reports. Re-scoring from the cache is what puts the two
passes together, and it is the difference between a right answer and a wrong
one: over the first two courses scanned, signal alone ranked 15-210 above
17-635, and adding vision reversed it.

Gates, and why probe_info is reconstructed
------------------------------------------
score._check_gates wants the ffprobe result, which the cache does not store --
only the metrics derived from it. Enough is recoverable to run every gate that
matters: a lecture whose camera failed to download has no camera_height, which
is exactly the `media_readable` finding, and duration/speech/instructor gates
read metrics directly. A gate whose input is genuinely absent is skipped, not
failed, which is the scanner's rule everywhere else.
"""

import argparse
import collections
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.scan import report, rubric, score as S  # noqa: E402
from textpdf import Doc, wrap as pdf_wrap  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def probe_from_metrics(m, tiers):
    """The subset of ffprobe's answer the gates need, from cached metrics.

    `has_audio` is the one that cannot be read straight off a metric. Every
    audio measurement lives in the signal tier, so on a probe-only lecture
    their absence means "not measured yet", not "silent" -- and reporting the
    gate as failed there would condemn a lecture for a tier nobody ran.
    """
    if not m.get("camera_height"):
        return {"camera": None, "screen": None}
    has_audio = (m.get("loudness_lufs") is not None
                 or "signal" not in tiers)
    cam = {"has_video": True, "has_audio": has_audio,
           "duration": m.get("duration_s")}
    scr = ({"duration": m.get("duration_s")}
           if m.get("screen_height") else None)
    return {"camera": cam, "screen": scr}


def load(cache):
    """Every cached lecture, re-scored. Sorted best first."""
    # Recursive, not `*/*/scan.json`. Most courses sit two levels down
    # (<course>/<key>/scan.json), but not all: the corpus dir for the
    # cross-listed 10-301-601 holds a `10-301` subdirectory whose lecture keys
    # are `601_<guid>`, because discover.py read the folder name as course
    # 10-301 plus a 601 section. A fixed-depth glob silently dropped all 32 of
    # its lectures and the course simply did not appear in the ranking --
    # which looks exactly like a course nobody scanned.
    out = []
    for path in sorted(glob.glob(os.path.join(cache, "**", "scan.json"),
                                 recursive=True)):
        try:
            d = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue
        lecture_dir = os.path.dirname(path)
        meta_path = os.path.join(lecture_dir, "metadata.json")
        meta = {}
        if os.path.exists(meta_path):
            try:
                meta = json.load(open(meta_path))
            except (json.JSONDecodeError, OSError):
                pass
        m = d.get("metrics") or {}
        tiers = d.get("tiers_run") or []
        identity = {
            "key": d.get("key") or os.path.basename(lecture_dir),
            "dir": lecture_dir,
            "course": meta.get("course")
                      or os.path.basename(os.path.dirname(lecture_dir)),
            "title": meta.get("name") or "",
            "owner": meta.get("owner") or "",
            "duration_s": m.get("duration_s") or meta.get("durationSec"),
            "scanned_at": d.get("scanned_at"),
        }
        out.append(S.evaluate(m, probe_from_metrics(m, tiers),
                              identity=identity, tiers_run=tiers))
    out.sort(key=lambda r: (r.get("score") is None, -(r.get("score") or 0)))
    return out



# "The quality has to be good." Resolution is the one quality floor a
# remediation pass cannot lift: loudness, noise and letterboxing are all
# fixable, and missing pixels are not. So it is a FILTER, not another scored
# metric -- a 720p lecture is not a worse 1080p lecture, it is a different
# thing to decide about.
#
# Both streams have to clear the bar, and the camera is what actually decides
# it: across the Spring corpus 87% of screen captures are 1080 or better and
# only 24% of cameras are, so a screen-only test would pass 625 lectures and
# mean almost nothing. Requiring both leaves 120.
HD_HEIGHT = 1080


def stream_height(r):
    """The lower of the two stream heights -- the one that limits the render.

    A missing screen counts as 0 rather than being ignored: a lecture with no
    slide capture cannot be a good 1080p publication whatever its camera did.
    """
    m = r.get("metrics") or {}
    return min(m.get("camera_height") or 0, m.get("screen_height") or 0)


def is_hd(r, min_height=HD_HEIGHT):
    return stream_height(r) >= min_height


def course_table(results, deepest="vision", keep=None):
    """One row per course, ranked by mean score over its publishable lectures.

    Two numbers a per-course report cannot give you, and both matter here:

      * `n/all` -- how many of a course's lectures actually count. A course
        whose downloads half failed has a mean over the survivors, and reading
        that next to a complete course without knowing is how 15-210's seven
        missing cameras inflated its first ranking.
      * `cov` -- how much of the rubric was measured. A course still mid-scan
        is comparable on the lectures that finished and not on the ones that
        have not, and the coverage column is what says so.
    """
    total = collections.Counter(r.get("course") for r in results)
    usable = collections.defaultdict(list)
    for r in results:
        if r.get("gates_failed") or r.get("score") is None:
            continue
        if deepest and deepest not in (r.get("tiers_run") or []):
            continue
        if keep and not keep(r):
            continue
        usable[r.get("course")].append(r)

    rows = []
    for course, rs in usable.items():
        s = [r["score"] for r in rs]
        best = max(rs, key=lambda r: r["score"])
        rows.append({
            "course": course,
            "n": len(rs),
            "all": total[course],
            "mean": statistics.mean(s),
            "median": statistics.median(s),
            "best": max(s),
            "potential": statistics.mean([r["potential"] for r in rs]),
            "coverage": statistics.mean([r["coverage"] for r in rs]),
            "grades": collections.Counter(r["grade"] for r in rs),
            "best_key": best["key"],
            "best_title": best.get("title") or best["key"],
            "best_score": best["score"],
            "best_potential": best["potential"],
        })
    rows.sort(key=lambda r: -r["mean"])
    return rows


def render_courses(rows, results):
    lines = ["# Spring 2026 -- course ranking", ""]
    graded = [r for r in results if not r.get("gates_failed")
              and r.get("score") is not None]
    failed = [r for r in results if r.get("gates_failed")]
    lines += [
        f"- **{len(rows)} courses**, {sum(r['n'] for r in rows)} lectures "
        f"ranked of {len(results)} scanned",
        f"- {len(failed)} lecture(s) failed a hard gate and are excluded "
        f"(almost all of them a camera that never downloaded)",
        f"- Mean score across every ranked lecture: "
        f"{statistics.mean([r['score'] for r in graded]):.1f}" if graded else "",
        "",
        "Ranked by mean score over the lectures that pass every hard gate and "
        "have been scanned to the vision tier. `n/all` is how many of the "
        "course's lectures that is -- a low ratio means failed downloads or a "
        "scan still running, and the mean is over the survivors either way.",
        "",
        "| # | Course | n/all | Mean | Median | Best | Potential | Coverage | "
        "Best lecture |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for i, r in enumerate(rows, 1):
        title = (r["best_title"] or "")[:52]
        lines.append(
            f"| {i} | `{r['course']}` | {r['n']}/{r['all']} | "
            f"{r['mean']:.1f} | {r['median']:.1f} | {r['best']:.1f} | "
            f"{r['potential']:.1f} | {r['coverage']:.0%} | "
            f"{title} ({r['best_score']:.1f}) |")
    lines.append("")
    return "\n".join(l for l in lines if l is not None)



def render_courses_text(rows, results, incomplete=(), hd_rows=None,
                        min_height=HD_HEIGHT):
    """The same league table as plain text, for pasting into an email.

    Fixed-width columns rather than markdown: the audience for this is a mail
    client, where a pipe table renders as pipes.
    """
    graded = [r for r in results if not r.get("gates_failed")
              and r.get("score") is not None]
    failed = [r for r in results if r.get("gates_failed")]
    ranked = sum(r["n"] for r in rows)

    out = []
    out.append("SPRING 2026 LECTURE SCAN -- COURSE RANKING")
    out.append("=" * 78)
    out.append("")
    out.append(f"{len(rows)} courses, {ranked} lectures ranked of "
               f"{len(results)} scanned.")
    if graded:
        out.append(f"Mean score across every ranked lecture: "
                   f"{statistics.mean([r['score'] for r in graded]):.1f} "
                   f"out of 100.")
    out.append(f"{len(failed)} lectures are excluded for failing a hard gate "
               f"-- almost all of")
    out.append("them a camera that never downloaded from Panopto.")
    out.append("")
    out.append("Scores are absolute, on the rubric in src/scan/rubric.py: five")
    out.append("weighted dimensions -- audio 28%, delivery 22%, student exposure")
    out.append("20%, visual 17%, content and structure 13%. 'Potential' is the same")
    out.append("lecture after the remediation the pipeline can actually apply")
    out.append("(loudness normalisation, noise reduction, letterboxing, cutting dead")
    out.append("air), so the gap between the two is the part worth spending time on.")
    out.append("")
    out.append("N/ALL is how many of a course's lectures are ranked. A low ratio")
    out.append("means failed downloads or a scan still running; the mean is over the")
    out.append("survivors either way.")
    out.append("")
    out.append("-" * 78)
    out.append(f"{'#':>3}  {'COURSE':<12} {'N/ALL':>7} {'MEAN':>6} {'MEDIAN':>7} "
               f"{'BEST':>6} {'POTENTIAL':>10}")
    out.append("-" * 78)
    for i, r in enumerate(rows, 1):
        flag = " *" if r["course"] in incomplete else ""
        out.append(f"{i:>3}  {r['course']:<12} "
                   f"{str(r['n']) + '/' + str(r['all']):>7} "
                   f"{r['mean']:>6.1f} {r['median']:>7.1f} {r['best']:>6.1f} "
                   f"{r['potential']:>10.1f}{flag}")
    out.append("-" * 78)
    if incomplete:
        out.append("")
        out.append(f"* scan still running for {', '.join(sorted(incomplete))}; "
                   f"those rows cover only")
        out.append("  the lectures finished so far and will move.")
    if hd_rows is not None:
        hd_n = sum(r["n"] for r in hd_rows)
        out.append("")
        out.append("")
        out.append(f"RANKING RESTRICTED TO {min_height}p+ RECORDINGS")
        out.append("=" * 78)
        out.append("")
        out.append(f"{hd_n} of {ranked} ranked lectures have BOTH the camera "
                   f"and the slide capture at")
        out.append(f"{min_height} lines or better, leaving {len(hd_rows)} of "
                   f"{len(rows)} courses. Resolution is a filter rather")
        out.append("than a scored metric because it is the one quality floor "
                   "no remediation")
        out.append("pass can lift -- loudness, noise and letterboxing are all "
                   "fixable, missing")
        out.append("pixels are not.")
        out.append("")
        out.append("The camera is what decides it. Across this corpus 87% of "
                   "screen captures are")
        out.append("1080 or better and only 24% of cameras are, so a "
                   "screen-only test would pass")
        out.append("almost everything and mean nothing.")
        out.append("")
        out.append("-" * 78)
        out.append(f"{'#':>3}  {'COURSE':<12} {'HD/ALL':>8} {'MEAN':>6} "
                   f"{'MEDIAN':>7} {'BEST':>6} {'POTENTIAL':>10}")
        out.append("-" * 78)
        for i, r in enumerate(hd_rows, 1):
            out.append(f"{i:>3}  {r['course']:<12} "
                       f"{str(r['n']) + '/' + str(r['all']):>8} "
                       f"{r['mean']:>6.1f} {r['median']:>7.1f} "
                       f"{r['best']:>6.1f} {r['potential']:>10.1f}")
        out.append("-" * 78)
        dropped = [r["course"] for r in rows
                   if r["course"] not in {h["course"] for h in hd_rows}]
        out.append("")
        out.append(f"Dropped entirely for having no {min_height}p+ lecture "
                   f"({len(dropped)} courses):")
        line = "  "
        for c in dropped:
            if len(line) + len(c) + 2 > 76:
                out.append(line); line = "  "
            line += c + ", "
        out.append(line.rstrip(", "))

    out.append("")
    out.append("")
    out.append("BEST LECTURE IN EACH COURSE")
    out.append("=" * 78)
    out.append("")
    for i, r in enumerate(rows, 1):
        out.append(f"{i:>3}. {r['course']}  --  {r['best_score']:.1f} "
                   f"(potential {r['best_potential']:.1f})")
        out.append(f"     {r['best_title']}")
        out.append(f"     {r['best_key']}")
        out.append("")
    return "\n".join(out)



def build_pdf(rows, results, path, incomplete=(), absolute=True,
              hd_rows=None, min_height=HD_HEIGHT):
    """The course league table as a PDF, with the rubric stated up front.

    The explanation is not padding. A bare ranking invites the two readings
    that are wrong -- that 79.9 means a bad course, and that the numbers are
    comparable to anything outside this cohort -- so the page that precedes it
    says what the axes are, what a grade band means, and which lectures were
    excluded and why. All of it is read out of src/scan/rubric.py at render
    time, so the document cannot drift from the grader it describes.
    """
    graded = [r for r in results if not r.get("gates_failed")
              and r.get("score") is not None]
    failed = [r for r in results if r.get("gates_failed")]
    ranked = sum(r["n"] for r in rows)
    mean = (statistics.mean([r["score"] for r in graded]) if graded else 0.0)

    d = Doc("Spring 2026 Lecture Scan",
            "Carnegie Mellon School of Computer Science - open courseware triage")

    d.heading("What this is", gap_before=6)
    d.para(
        f"Every lecture recording from {len(rows)} Spring 2026 courses was "
        f"downloaded and measured automatically, then scored on a fixed "
        f"rubric. {ranked} of {len(results)} scanned lectures are ranked "
        f"here; the rest failed a hard gate, described below. The purpose is "
        f"triage: deciding which recordings are worth putting through "
        f"anonymization and publication, before anyone spends the hours.")
    d.para(
        "Nothing here is a human judgement of teaching. The rubric measures "
        "the RECORDING -- whether you can hear it, see the slides and see the "
        "instructor -- and a low score usually means a room, a microphone or "
        "a capture setting, not a lecturer.")

    d.heading("How a lecture is scored")
    d.para(
        "Five dimensions, weighted against each other. A lecture's score is "
        "the weighted mean over the dimensions that were actually measured, "
        "renormalised so an unmeasured one is not counted as a zero.")
    metrics_per = collections.Counter(
        m["dimension"] for m in rubric.METRICS.values())
    lines = [f"{'DIMENSION':<40} {'WEIGHT':>7} {'METRICS':>8}"]
    lines.append("-" * 57)
    for key, dim in sorted(rubric.DIMENSIONS.items(),
                           key=lambda kv: -kv[1]["weight"]):
        lines.append(f"{dim['label'][:40]:<40} {dim['weight']:>6.0%} "
                     f"{metrics_per.get(key, 0):>8}")
    lines.append("-" * 57)
    lines.append(f"{'':<40} {'':>7} {sum(metrics_per.values()):>8}")
    d.table(lines, size=9.0, head=2)
    d.bullets([f"{dim['label']} - {dim['blurb']}"
               for _, dim in sorted(rubric.DIMENSIONS.items(),
                                    key=lambda kv: -kv[1]["weight"])],
              size=9.4)

    d.heading("Score and potential")
    d.para(
        "Two numbers per lecture. SCORE is the recording as it stands. "
        "POTENTIAL is the same recording after the remediation this pipeline "
        "can actually apply -- loudness normalisation, noise reduction, "
        "letterboxing a mismatched slide capture, cutting dead air. The gap "
        "between them is the useful part: 52 rising to 78 is a quiet, hissy "
        "recording worth publishing, while 52 rising to 54 is a monotone talk "
        "over a dead slide that no encoder setting will fix.")

    d.heading("Grade bands")
    band_lines = [f"{'GRADE':<7} {'FROM':>6}  {'VERDICT':<10} MEANING"]
    band_lines.append("-" * 78)
    for cut, letter, verdict, blurb in rubric.GRADES:
        # 27 characters of prefix, and Courier at 8.6pt fits ~97 to the
        # measure, so 68 is the most the meaning column can hold. It was 44,
        # which cut "unless it is re" off mid-word.
        band_lines.append(f"{letter:<7} {cut:>6.0f}  {verdict:<10} {blurb[:68]}")
    d.table(band_lines, size=8.6, head=2)

    d.heading("Hard gates (pass/fail, separate from the score)")
    d.para(
        '"How good is it" and "publish it at all" are different questions, '
        "and averaging them answers neither. A lecture that fails any gate "
        "below is excluded from the ranking whatever it scored. A gate whose "
        "measurement is missing is skipped, not failed.")
    d.bullets([g["label"] for g in rubric.GATES], size=9.4)
    d.para(
        f"{len(failed)} of {len(results)} lectures failed a gate. Almost all "
        f"of them are a camera stream that never downloaded from Panopto, "
        f"which is a fetch problem rather than a recording problem -- those "
        f"lectures are worth re-fetching before being written off.")

    d.heading("How much was measured")
    d.para(
        "Scanning is tiered and cumulative: a fast pass over metadata and "
        "container, then audio and slide-change measurement, then face and "
        "instructor detection. The COVERAGE column is the share of the rubric "
        "actually measured for a course. Below 55% a lecture gets a "
        "provisional score and no letter grade, because a confident-looking "
        "number from three metrics is worse than no number.")
    d.para(
        "Scores in this document are ABSOLUTE -- measured against fixed "
        "thresholds, so they answer 'is this worth publishing at all'. A "
        "second version recalibrated against this cohort answers 'what "
        "first' instead, and ranks the courses differently. The two should "
        "not be mixed."
        if absolute else
        "Scores in this document are RECALIBRATED against this cohort, so "
        "they are relative to Spring 2026 and answer 'what to publish "
        "first'. They are not absolute judgements of watchability.")

    # --- the ranking ---
    d.heading("Course ranking")
    d.para(
        "Ranked by mean score over the lectures that pass every hard gate and "
        "have been scanned to the deepest tier. N/ALL is how many of a "
        "course's lectures that is: a low ratio means failed downloads or a "
        "scan still running, and the mean is over the survivors either way.")
    tl = [f"{'#':>3}  {'COURSE':<12} {'N/ALL':>7} {'MEAN':>6} {'MEDIAN':>7} "
          f"{'BEST':>6} {'POTENTL':>8} {'COV':>5}"]
    tl.append("-" * 62)
    for i, r in enumerate(rows, 1):
        flag = " *" if r["course"] in incomplete else ""
        tl.append(f"{i:>3}  {r['course']:<12} "
                  f"{str(r['n']) + '/' + str(r['all']):>7} "
                  f"{r['mean']:>6.1f} {r['median']:>7.1f} {r['best']:>6.1f} "
                  f"{r['potential']:>8.1f} {r['coverage']:>4.0%}{flag}")
    d.table(tl, size=8.6, head=2)
    d.para(f"Mean across every ranked lecture: {mean:.1f}.", size=9.4)
    if incomplete:
        d.para(f"* Scan still running for {', '.join(sorted(incomplete))}. "
               f"Those rows cover only the lectures finished so far and will "
               f"move.", size=9.4, gray=0.35)

    if hd_rows is not None:
        hd_n = sum(r["n"] for r in hd_rows)
        d.heading(f"Ranking restricted to {min_height}p+ recordings")
        d.para(
            f"{hd_n} of {ranked} ranked lectures have BOTH the camera and the "
            f"slide capture at {min_height} lines or better, which leaves "
            f"{len(hd_rows)} of {len(rows)} courses. Resolution is applied as "
            f"a filter rather than scored as another metric, because it is "
            f"the one quality floor no remediation pass can lift: loudness, "
            f"noise and letterboxing are all fixable, and missing pixels are "
            f"not.")
        d.para(
            "The camera decides it. Across this corpus 87% of screen captures "
            "are 1080 or better but only 24% of cameras are, so a screen-only "
            "test would pass almost everything and mean nothing.")
        hl = [f"{'#':>3}  {'COURSE':<12} {'HD/ALL':>8} {'MEAN':>6} "
              f"{'MEDIAN':>7} {'BEST':>6} {'POTENTL':>8}", "-" * 58]
        for i, r in enumerate(hd_rows, 1):
            hl.append(f"{i:>3}  {r['course']:<12} "
                      f"{str(r['n']) + '/' + str(r['all']):>8} "
                      f"{r['mean']:>6.1f} {r['median']:>7.1f} "
                      f"{r['best']:>6.1f} {r['potential']:>8.1f}")
        d.table(hl, size=8.6, head=2)
        dropped = [r["course"] for r in rows
                   if r["course"] not in {h["course"] for h in hd_rows}]
        d.para(f"Dropped for having no {min_height}p+ lecture at all "
               f"({len(dropped)}): {', '.join(dropped)}.", size=9.4,
               gray=0.35)

    d.heading("Best lecture in each course")
    d.para("In course-ranking order. The identifier is the Panopto session "
           "key, for anyone matching a row back to a recording.", size=9.4)
    for i, r in enumerate(rows, 1):
        d.text(f"{i}. {r['course']} - {r['best_score']:.1f} "
               f"(potential {r['best_potential']:.1f})", "bold", 9.6)
        for line in pdf_wrap(r["best_title"] or r["best_key"],
                             d.content_w - 14, "body", 9.4):
            d.text(line, "body", 9.4, gray=0.13, x=54 + 14)
        d.text(r["best_key"], "mono", 7.6, gray=0.45, x=54 + 14, gap_after=5)

    return d.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--cache", default=os.path.join(ROOT, "reports", "psc-cache"))
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "semester"))
    ap.add_argument("--tier", default="vision",
                    help="Deepest tier a lecture must have for its course row. "
                         "'' to include everything scored.")
    ap.add_argument("--limit", type=int, default=40,
                    help="Lectures in the markdown ranking")
    ap.add_argument("--min-height", type=int, default=HD_HEIGHT,
                    help="Second ranking restricted to lectures whose camera "
                         "AND screen are at least this many lines. 0 to skip "
                         "that section.")
    ap.add_argument("--incomplete", nargs="*", default=[],
                    help="Courses whose scan is still running, flagged in the "
                         "text report so a moving number is not read as final")
    ap.add_argument("--absolute", action="store_true",
                    help="Ignore rubric_overrides.json for this run")
    args = ap.parse_args()

    if not args.absolute:
        rubric.load_overrides()

    results = load(args.cache)
    if not results:
        raise SystemExit(f"no scan.json under {args.cache}")
    results = report.cohort_percentiles(results)
    rows = course_table(results, deepest=args.tier or None)
    hd_rows = (course_table(results, deepest=args.tier or None,
                            keep=lambda r: is_hd(r, args.min_height))
               if args.min_height else None)

    os.makedirs(args.out, exist_ok=True)
    courses_md = render_courses(rows, results)
    for name, text in (
            ("courses.md", courses_md),
            ("courses.txt", render_courses_text(rows, results, args.incomplete,
                                                hd_rows, args.min_height)),
            ("lectures.md", report.render_markdown(results, limit=args.limit)),
            ("lectures.csv", report.render_csv(results)),
            ("lectures.html", report.render_html(results)),
    ):
        with open(os.path.join(args.out, name), "w") as f:
            f.write(text)
    build_pdf(rows, results, os.path.join(args.out, "course-ranking.pdf"),
              incomplete=args.incomplete, absolute=args.absolute,
              hd_rows=hd_rows, min_height=args.min_height)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)

    print(courses_md)
    print(f"[semester] wrote {args.out}/ "
          f"(courses.md, courses.txt, lectures.md, lectures.csv, "
          f"lectures.html, results.json, course-ranking.pdf)")


if __name__ == "__main__":
    main()
