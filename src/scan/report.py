"""Turning a scanned semester into something a human will act on.

The scanner produces one result dict per lecture and that is already the whole
truth, but a list of forty of those is not a decision. This module renders it
four ways, and the split is by *who is reading*:

    render_markdown   the semester at a glance -- what to publish, whose
                      lectures are weak, and where an hour of remediation buys
                      the most. Pasteable into a ticket or an email.
    render_csv        one flat row per lecture with every raw measurement, for
                      the triage nobody can anticipate. Sorting and filtering
                      in a spreadsheet beats any view invented here.
    render_html       the same ranking, readable on a phone and archivable
                      next to the batch it describes.
    render_lecture    one lecture in full, which is what `--explain` prints
                      when somebody disagrees with a grade.

Usage:

    from src.scan import report

    results = report.cohort_percentiles(results)
    print(report.render_markdown(results, limit=20))
    with open("scan.csv", "w") as fh:
        fh.write(report.render_csv(results))
    with open("scan.html", "w") as fh:
        fh.write(report.render_html(results))
    print(report.render_lecture(results[0]))

Two rules run through all of it.

**Nothing here decides what is good.** Every threshold, weight, label, unit and
grade band is read out of src/scan/rubric.py at render time. A retuned rubric
changes these reports without a line changing here, and a number printed in a
report can always be traced back to the row of the table that produced it.

**A metric that was not measured is not a zero.** A scan stopped at --tier
signal has no face counts, and rendering those as 0.0 would quietly turn "we
did not look" into "there were no students in the room" -- the exact direction
of error this repo cannot afford. Absent metrics render as "not measured",
their dimensions carry an honest `coverage`, and they are left out of every
mean rather than dragging one down.
"""

import csv
import html
import io

import numpy as np

from src.scan import rubric
from src.scan.rubric import DIMENSIONS, GRADES, METRICS

# Below this many lectures the scanner reports no percentiles at all. A rank
# within three samples is not a weak signal, it is noise wearing the costume of
# information: with two lectures every metric reads 25th or 75th percentile and
# the loser looks systematically bad because somebody had to be second. The
# absolute bands in the rubric are honest at any n; the percentile only starts
# meaning something once a cohort exists, so it is withheld until one does.
MIN_COHORT = 4

# Verdicts and grade letters in rubric order, deduplicated. Read rather than
# written down so that adding a band to GRADES shows up in every summary.
VERDICTS = list(dict.fromkeys(v for _, _, v, _ in GRADES))
LETTERS = [letter for _, letter, _, _ in GRADES]

UNKNOWN = "(unknown)"


# --------------------------------------------------------------------------
# Cohort statistics
# --------------------------------------------------------------------------

def cohort_percentiles(results, min_cohort=MIN_COHORT):
    """Annotate each result with `percentiles`: metric_id -> 0..100.

    Ranked on the SUB-score, never on the raw value. Half the table is
    better-when-lower (noise floor, dead air, student face clusters) and
    several metrics are bands where both ends are bad, so a rank over raw
    numbers would put the quietest room and the loudest one at opposite ends
    of the same axis. The sub-score is already direction-normalised by
    rubric.score_metric, which makes 100 mean "best in the cohort at this"
    for every metric regardless of which way its raw scale points.

    Ties take the mid-rank, so a metric every lecture aces reads 50 rather
    than everybody simultaneously beating everybody.
    """
    results = list(results)
    if len(results) < min_cohort:
        for r in results:
            r["percentiles"] = None
        return results

    for r in results:
        r["percentiles"] = {}

    for mid in METRICS:
        have = [(r, r.get("subscores", {}).get(mid)) for r in results]
        have = [(r, s) for r, s in have if s is not None]
        # A metric measured on only a couple of lectures gets the same
        # treatment as a small cohort: no rank rather than a flattering one.
        if len(have) < min_cohort:
            continue
        vals = np.array([s for _, s in have], dtype=float)
        for r, s in have:
            below = float(np.count_nonzero(vals < s))
            equal = float(np.count_nonzero(vals == s))
            r["percentiles"][mid] = 100.0 * (below + 0.5 * equal) / vals.size

    return results


# --------------------------------------------------------------------------
# Small shared accessors. Every one of these tolerates a result dict that is
# missing keys, because a lecture that errored out mid-scan is still a row.
# --------------------------------------------------------------------------

def _score(r):
    return float(r.get("score") or 0.0)


def _potential(r):
    p = r.get("potential")
    return float(p) if p is not None else _score(r)


def _gain(r):
    # Never negative: remediation cannot make a lecture worse, and a rounding
    # artefact in that direction would sort to the top of "cheapest wins".
    return max(0.0, _potential(r) - _score(r))


def _grade(r):
    return r.get("grade") or "?"


def _verdict(r):
    return r.get("verdict") or "?"


def _key(r):
    return r.get("key") or r.get("dir") or UNKNOWN


def _dims(r):
    return r.get("dimensions") or {}


def weakest_dimension(r):
    """(dimension_id, score) for the lowest-scoring measured dimension.

    None when nothing was measured. Dimensions with zero coverage are skipped
    rather than counted as 0 -- an unrun tier is not a weakness of the
    lecture, and reporting it as one sends people to re-record a lecture whose
    problem is that nobody ran the vision pass.
    """
    scored = [(d, v) for d, v in _dims(r).items()
              if v and (v.get("coverage") or 0.0) > 0.0
              and v.get("score") is not None]
    if not scored:
        return None
    dim, got = min(scored, key=lambda kv: kv[1]["score"])
    return dim, got["score"]


def _graded(results):
    """Only the lectures that actually got a score.

    score.py leaves `score` None when it refused to grade one -- nothing
    measured, or coverage under its floor -- and every mean, best and worst on
    the page runs over these. Folding an ungraded lecture in as a zero would
    report a scan that did not finish as a lecturer who did badly.
    """
    return [r for r in results if r.get("score") is not None]


def _mean_score(results):
    return _mean([_score(r) for r in _graded(results)])


def _mean_potential(results):
    return _mean([_potential(r) for r in _graded(results)])


def _cells(r):
    """(score, potential, gain) as they should print for one lecture.

    An ungraded lecture gets dashes rather than 0.0. It still occupies a row
    -- it is work somebody has to do something about -- but printing a zero
    would rank a failed scan against lectures that were actually measured.
    """
    if r.get("score") is None:
        return "--", "--", "--"
    return _pts(_score(r)), _pts(_potential(r)), _pts(_gain(r))


def _best_worst(results):
    """(best, worst) by score among graded lectures, or (None, None)."""
    graded = _graded(results)
    if not graded:
        return None, None
    return max(graded, key=_score), min(graded, key=_score)


def _verdict_order(results):
    """The rubric's verdicts, plus any the scorer added on top of them.

    score.py emits `unscanned` and `incomplete` for lectures it refused to
    grade, and those are not in GRADES. Appending whatever actually turned up
    keeps them out of nobody's count -- a summary that silently drops the
    lectures that failed to scan is the summary that gets believed.
    """
    extra = [v for v in dict.fromkeys(_verdict(r) for r in results)
             if v not in VERDICTS]
    return VERDICTS + extra


def _dim_label(dim):
    return DIMENSIONS.get(dim, {}).get("label", dim)


def _mean(values):
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def _hms(seconds):
    """Runtime as `1 h 19 m`, or `--` when it was never established."""
    if not seconds:
        return "--"
    total = int(round(float(seconds)))
    h, rem = divmod(total, 3600)
    m = rem // 60
    if h:
        return f"{h} h {m:02d} m"
    return f"{m} m"


def _num(value, places=2):
    """A float printed the way a person would write it."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if v == int(v) and abs(v) < 1e6:
        return f"{int(v)}"
    return f"{round(v, places):g}"


def _pts(value):
    """Score points, always to one decimal.

    Deliberately not _num: a column of scores where 46.0 prints as `46` and
    45.9 prints as `45.9` stops being a column, and these are the numbers
    people compare down the page.
    """
    if value is None:
        return "--"
    return f"{float(value):.1f}"


def _fmt_raw(mid, value):
    """Raw measurement with its unit, or the words 'not measured'."""
    if value is None:
        return "not measured"
    spec = METRICS.get(mid, {})
    if spec.get("scale", (None,))[0] == "bool":
        return "yes" if value else "no"
    unit = spec.get("unit")
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{_num(value)} {unit}" if unit else _num(value)
    return str(value)


def _group(results, field):
    """Group results by a possibly-missing identity field, in first-seen
    order so that a report over a semester reads in the order it was scanned
    rather than in whatever order a dict happened to hash."""
    groups = {}
    for r in results:
        groups.setdefault(r.get(field) or UNKNOWN, []).append(r)
    return groups


def _ranked(results):
    """Best first. Failed gates sink regardless of score, because the grade
    already says skip and a high-scoring skip at the top of the table reads as
    a recommendation."""
    return sorted(
        results,
        key=lambda r: (bool(r.get("gates_failed")), -_score(r), _key(r)))


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------

def _md_cell(text):
    return str(text).replace("|", "\\|").replace("\n", " ")


def _md_bw(entry):
    """A best/worst cell. A course whose every lecture failed to scan has
    neither, and a dash says so without inventing a winner."""
    if not entry:
        return "--"
    return f"`{_md_cell(_key(entry))}` {_pts(_score(entry))}"


def render_markdown(results, limit=None):
    """The semester as a document. `limit` truncates the ranked table only.

    The per-course, per-instructor and needs-attention sections always cover
    everything scanned: truncating those would hide exactly the lecture
    somebody needs to see, which is the one at the bottom.
    """
    results = list(results)
    if not results:
        return ("# Lecture scan\n\n"
                "No lectures scanned. Point `--root` at a directory of "
                "lecture folders.\n")

    out = []
    add = out.append

    total_s = sum(float(r.get("duration_s") or 0.0) for r in results)
    order = _verdict_order(results)
    counts = {v: 0 for v in order}
    for r in results:
        counts[_verdict(r)] = counts.get(_verdict(r), 0) + 1
    scanned = sorted(r.get("scanned_at") or "" for r in results)

    add(f"# Lecture scan -- {len(results)} lectures")
    add("")
    add(f"- **Runtime scanned:** {_hms(total_s)} "
        f"across {len(results)} recordings")
    add("- **Verdicts:** " + ", ".join(
        f"{counts.get(v, 0)} {v}" for v in order))
    mean = _mean_score(results)
    mean_pot = _mean_potential(results)
    add(f"- **Mean score:** {_pts(mean)} "
        f"(potential {_pts(mean_pot)})")
    if scanned and scanned[-1]:
        add(f"- **Last scanned:** {scanned[-1]}")
    tiers = sorted({t for r in results for t in (r.get("tiers_run") or [])},
                   key=lambda t: rubric.TIERS.index(t)
                   if t in rubric.TIERS else 99)
    if tiers:
        add(f"- **Tiers run:** {', '.join(tiers)}")
    add("")

    # --- ranking ---------------------------------------------------------
    ranked = _ranked(results)
    shown = ranked[:limit] if limit else ranked
    add("## Ranked")
    add("")
    if limit and len(ranked) > len(shown):
        add(f"Top {len(shown)} of {len(ranked)}. "
            "The full set is in the CSV.")
        add("")
    add("| # | Lecture | Course | Grade | Score | Potential | Weakest |")
    add("|---:|---|---|:---:|---:|---:|---|")
    for i, r in enumerate(shown, 1):
        weak = weakest_dimension(r)
        weak_txt = (f"{_dim_label(weak[0])} ({_num(weak[1] * 100, 0)})"
                    if weak else "not measured")
        score_c, pot_c, _ = _cells(r)
        add(f"| {i} | `{_md_cell(_key(r))}` "
            f"| {_md_cell(r.get('course') or UNKNOWN)} "
            f"| {_grade(r)} | {score_c} "
            f"| {pot_c} | {_md_cell(weak_txt)} |")
    add("")

    # --- by course -------------------------------------------------------
    add("## By course")
    add("")
    add("| Course | Lectures | Mean | Best | Worst |")
    add("|---|---:|---:|---|---|")
    for course, group in sorted(
            _group(results, "course").items(),
            key=lambda kv: -(_mean_score(kv[1]) or 0.0)):
        best, worst = _best_worst(group)
        add(f"| {_md_cell(course)} | {len(group)} "
            f"| {_pts(_mean_score(group))} "
            f"| {_md_bw(best)} | {_md_bw(worst)} |")
    add("")

    # --- by instructor ---------------------------------------------------
    # Ranking lecturers is a stated goal of the scanner, and it is also the
    # section most easily misread, so it carries its own caveat rather than
    # relying on the reader to remember one.
    add("## By instructor")
    add("")
    add("Measured delivery, not teaching. A lecturer handed a bad room and a "
        "lapel mic that was not switched on will rank low here for reasons "
        "that are not theirs.")
    add("")
    add("| Instructor | Lectures | Mean | Best | Worst |")
    add("|---|---:|---:|---|---|")
    for owner, group in sorted(
            _group(results, "owner").items(),
            key=lambda kv: -(_mean_score(kv[1]) or 0.0)):
        best, worst = _best_worst(group)
        add(f"| {_md_cell(owner)} | {len(group)} "
            f"| {_pts(_mean_score(group))} "
            f"| {_md_bw(best)} | {_md_bw(worst)} |")
    add("")

    # --- needs attention -------------------------------------------------
    flagged = [r for r in results
               if r.get("gates_failed") or r.get("errors")]
    add("## Needs attention")
    add("")
    if not flagged:
        add("No failed gates and no scan errors.")
        add("")
    else:
        for r in flagged:
            add(f"### `{_key(r)}` -- {_grade(r)} / {_verdict(r)}")
            add("")
            for label in r.get("gates_failed") or []:
                why = next((g["why"] for g in rubric.GATES
                            if g["label"] == label or g["id"] == label), "")
                add(f"- **Gate failed:** {label}"
                    + (f" -- {why}" if why else ""))
            for err in r.get("errors") or []:
                add(f"- **Error:** {err}")
            for warn in r.get("warnings") or []:
                add(f"- Warning: {warn}")
            add("")

    # --- cheapest wins ---------------------------------------------------
    # The whole reason two scores exist. Sorting by score finds bad lectures;
    # sorting by the gap finds the ones an afternoon of ffmpeg would fix.
    add("## Cheapest wins")
    add("")
    add("Sorted by how much the pipeline's own remediation would recover. "
        "A large gap is a to-do; a small one on a low score means the "
        "problem is in the room, not in post.")
    add("")
    improvable = [r for r in sorted(results, key=_gain, reverse=True)
                  if _gain(r) >= 0.1]
    if not improvable:
        add("Nothing measurably improvable by remediation.")
        add("")
    for r in improvable:
        add(f"### `{_key(r)}` +{_pts(_gain(r))} points "
            f"({_pts(_score(r))} -> {_pts(_potential(r))})")
        add("")
        for item in r.get("remediation") or []:
            label = item.get("label") or METRICS.get(
                item.get("metric"), {}).get("label", item.get("metric"))
            note = item.get("note") or ""
            add(f"- **{label}** +{_pts(item.get('gain') or 0.0)}"
                + (f" -- {note}" if note else ""))
        add("")

    return "\n".join(out).rstrip() + "\n"


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------

def render_csv(results):
    """One row per lecture, one column per rubric metric, raw values.

    Column order comes from METRICS so that two scans of different semesters,
    or of different tiers, still diff and concatenate. A metric that was not
    measured is an EMPTY cell -- a spreadsheet reads a blank as missing and a
    0 as a measurement, and here the difference decides whether somebody
    re-records a lecture.
    """
    buf = io.StringIO()
    # Excel opens \r\n rows fine and so does everything else; forcing \n keeps
    # the return value diffable when it is written straight to a file.
    writer = csv.writer(buf, lineterminator="\n")

    head = ["key", "dir", "course", "title", "owner", "duration_s",
            "scanned_at", "tiers_run", "score", "potential", "grade",
            "verdict", "gates_failed", "warnings", "errors"]
    writer.writerow(head + list(METRICS))

    for r in _ranked(results):
        metrics = r.get("metrics") or {}
        row = [
            _key(r),
            r.get("dir") or "",
            r.get("course") or "",
            r.get("title") or "",
            r.get("owner") or "",
            _num(r["duration_s"], 1) if r.get("duration_s") else "",
            r.get("scanned_at") or "",
            " ".join(r.get("tiers_run") or []),
            # Blank, not 0.0, for an ungraded lecture: same reason the metric
            # columns leave a missing measurement empty.
            _pts(r.get("score")) if r.get("score") is not None else "",
            _pts(r.get("potential"))
            if r.get("potential") is not None else "",
            _grade(r),
            _verdict(r),
            "; ".join(r.get("gates_failed") or []),
            len(r.get("warnings") or []),
            "; ".join(r.get("errors") or []),
        ]
        for mid in METRICS:
            value = metrics.get(mid)
            if value is None:
                row.append("")
            elif isinstance(value, bool):
                row.append(int(value))
            elif isinstance(value, (int, float)):
                row.append(_num(value, 3))
            else:
                row.append(str(value))
        writer.writerow(row)

    return buf.getvalue()


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

def _esc(text):
    return html.escape("" if text is None else str(text))


# Palette, both themes, as CSS custom properties. Written out once here rather
# than inline on elements so that the dark overrides are three short blocks
# instead of a second stylesheet, and so a colour has exactly one definition.
_CSS = """
:root {
  --bg: #f6f7f9;
  --surface: #ffffff;
  --surface-2: #f0f2f5;
  --border: #dce0e6;
  --text: #16191d;
  --text-dim: #5c6470;
  --text-faint: #8a929e;
  --accent: #8c1d40;
  --bar-track: #e4e7ec;
  --bar-fill: #8c1d40;
  --bar-fill-low: #b4472f;
  --good: #1d6f42;
  --warn: #8a5a00;
  --bad: #a32020;
  --chip-a-bg: #d9f0e2; --chip-a-fg: #14532d;
  --chip-b-bg: #dcecfa; --chip-b-fg: #1e3f66;
  --chip-c-bg: #fbf0d2; --chip-c-fg: #6b4c00;
  --chip-d-bg: #fbe3d3; --chip-d-fg: #7a3b12;
  --chip-f-bg: #fadadd; --chip-f-fg: #7d1420;
  --shadow: 0 0.0625rem 0.125rem rgba(16, 20, 26, 0.08);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0f1216;
    --surface: #171b21;
    --surface-2: #1d222a;
    --border: #2b323c;
    --text: #e6e9ee;
    --text-dim: #9aa3b0;
    --text-faint: #6f7885;
    --accent: #e2879f;
    --bar-track: #262c35;
    --bar-fill: #d16a86;
    --bar-fill-low: #d98a5a;
    --good: #6fcf97;
    --warn: #e0b252;
    --bad: #e97b7b;
    --chip-a-bg: #16341f; --chip-a-fg: #9be3b4;
    --chip-b-bg: #182b3f; --chip-b-fg: #a8cdf0;
    --chip-c-bg: #3a2f10; --chip-c-fg: #ecd28a;
    --chip-d-bg: #3d2413; --chip-d-fg: #f0b78e;
    --chip-f-bg: #3d1519; --chip-f-fg: #f2a3a8;
    --shadow: 0 0.0625rem 0.125rem rgba(0, 0, 0, 0.5);
  }
}
:root[data-theme="dark"] {
  --bg: #0f1216;
  --surface: #171b21;
  --surface-2: #1d222a;
  --border: #2b323c;
  --text: #e6e9ee;
  --text-dim: #9aa3b0;
  --text-faint: #6f7885;
  --accent: #e2879f;
  --bar-track: #262c35;
  --bar-fill: #d16a86;
  --bar-fill-low: #d98a5a;
  --good: #6fcf97;
  --warn: #e0b252;
  --bad: #e97b7b;
  --chip-a-bg: #16341f; --chip-a-fg: #9be3b4;
  --chip-b-bg: #182b3f; --chip-b-fg: #a8cdf0;
  --chip-c-bg: #3a2f10; --chip-c-fg: #ecd28a;
  --chip-d-bg: #3d2413; --chip-d-fg: #f0b78e;
  --chip-f-bg: #3d1519; --chip-f-fg: #f2a3a8;
  --shadow: 0 0.0625rem 0.125rem rgba(0, 0, 0, 0.5);
}

* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 400 1rem/1.55 ui-sans-serif, -apple-system, "Segoe UI", Roboto,
        "Helvetica Neue", Arial, sans-serif;
  -webkit-text-size-adjust: 100%;
}
.wrap { max-width: 72rem; margin: 0 auto; padding: 2rem 1.25rem 4rem; }
h1 { font-size: 1.6rem; margin: 0 0 0.25rem; letter-spacing: -0.01em; }
h2 {
  font-size: 1.15rem; margin: 2.5rem 0 0.75rem;
  padding-bottom: 0.4rem; border-bottom: 0.0625rem solid var(--border);
}
h3 { font-size: 1rem; margin: 0 0 0.25rem; }
p { margin: 0.4rem 0; }
.sub { color: var(--text-dim); font-size: 0.9rem; }
.note { color: var(--text-faint); font-size: 0.82rem; }
code, .mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.86em;
}

.stats { display: flex; flex-wrap: wrap; gap: 0.75rem; margin: 1.25rem 0 0; }
.stat {
  flex: 1 1 8rem; background: var(--surface); border: 0.0625rem solid
  var(--border); border-radius: 0.5rem; padding: 0.7rem 0.85rem;
  box-shadow: var(--shadow);
}
.stat .k { color: var(--text-dim); font-size: 0.75rem;
           text-transform: uppercase; letter-spacing: 0.04em; }
.stat .v { font-size: 1.35rem; font-variant-numeric: tabular-nums; }

.scroll { overflow-x: auto; -webkit-overflow-scrolling: touch;
          border: 0.0625rem solid var(--border); border-radius: 0.5rem;
          background: var(--surface); box-shadow: var(--shadow); }
table { border-collapse: collapse; width: 100%; min-width: 46rem;
        font-size: 0.9rem; }
th, td { padding: 0.5rem 0.7rem; text-align: left;
         border-bottom: 0.0625rem solid var(--border); white-space: nowrap; }
thead th {
  position: sticky; top: 0; background: var(--surface-2);
  color: var(--text-dim); font-weight: 600; font-size: 0.78rem;
  text-transform: uppercase; letter-spacing: 0.04em; z-index: 1;
}
th.num, td.num { text-align: right; font-variant-numeric: tabular-nums; }
tbody tr:last-child td { border-bottom: 0; }
tbody tr:hover td { background: var(--surface-2); }
th .caret { color: var(--accent); font-size: 0.7em; }
td.wide { white-space: normal; min-width: 12rem; }

.chip {
  display: inline-block; min-width: 1.5rem; text-align: center;
  padding: 0.1rem 0.45rem; border-radius: 0.3rem; font-weight: 700;
  font-size: 0.8rem;
}
.chip-A { background: var(--chip-a-bg); color: var(--chip-a-fg); }
.chip-B { background: var(--chip-b-bg); color: var(--chip-b-fg); }
.chip-C { background: var(--chip-c-bg); color: var(--chip-c-fg); }
.chip-D { background: var(--chip-d-bg); color: var(--chip-d-fg); }
.chip-F { background: var(--chip-f-bg); color: var(--chip-f-fg); }
.chip-x { background: var(--surface-2); color: var(--text-dim);
          border: 0.0625rem solid var(--border); }

.cards { display: grid; gap: 0.9rem;
         grid-template-columns: repeat(auto-fill, minmax(20rem, 1fr)); }
.card {
  background: var(--surface); border: 0.0625rem solid var(--border);
  border-radius: 0.5rem; padding: 0.9rem 1rem; box-shadow: var(--shadow);
}
.card header { display: flex; align-items: baseline; gap: 0.5rem;
               justify-content: space-between; margin-bottom: 0.5rem; }
.dim { margin: 0.45rem 0; }
.dim .row { display: flex; justify-content: space-between;
            font-size: 0.78rem; color: var(--text-dim); }
.bar { height: 0.4rem; background: var(--bar-track); border-radius: 0.2rem;
       overflow: hidden; margin-top: 0.15rem; }
.bar span { display: block; height: 100%; background: var(--bar-fill); }
.bar.low span { background: var(--bar-fill-low); }
.dim.unmeasured .bar span { background: repeating-linear-gradient(
    45deg, var(--border) 0 0.25rem, transparent 0.25rem 0.5rem); }
.flag { color: var(--bad); font-size: 0.82rem; }
.warnline { color: var(--warn); font-size: 0.82rem; }
.fix { color: var(--good); font-size: 0.82rem; }
ul.tight { margin: 0.3rem 0 0; padding-left: 1.1rem; }
ul.tight li { font-size: 0.82rem; color: var(--text-dim); }
footer { margin-top: 3rem; color: var(--text-faint); font-size: 0.8rem; }
"""


def _chip(letter):
    # `?` is score.py's "not graded" (nothing measured, or coverage below its
    # floor). It gets the neutral chip, never the red one -- an incomplete
    # scan looking like a failing lecture is how a good lecture gets dropped.
    cls = letter if letter in LETTERS else "x"
    return f'<span class="chip chip-{_esc(cls)}">{_esc(letter)}</span>'


def _bar(score, coverage):
    """One dimension bar. An unmeasured dimension gets a hatched empty track,
    never a full-width bar at zero, so the eye does not read absent as bad."""
    if score is None or not coverage:
        return ('<div class="bar"><span style="width:100%"></span></div>')
    pct = max(0.0, min(1.0, float(score))) * 100.0
    low = " low" if float(score) < 0.6 else ""
    return (f'<div class="bar{low}">'
            f'<span style="width:{pct:.1f}%"></span></div>')


def _html_card(r):
    out = []
    add = out.append
    weak = weakest_dimension(r)
    add('<article class="card">')
    add('<header><h3 class="mono">' + _esc(_key(r)) + "</h3>"
        + _chip(_grade(r)) + "</header>")
    meta = " &middot; ".join(_esc(x) for x in [
        r.get("course") or UNKNOWN, r.get("owner") or UNKNOWN,
        _hms(r.get("duration_s"))])
    add(f'<p class="sub">{meta}</p>')
    if r.get("title"):
        add(f'<p class="sub">{_esc(r["title"])}</p>')
    add(f'<p class="sub">score <strong>{_cells(r)[0]}</strong> '
        f'&rarr; potential <strong>{_cells(r)[1]}</strong> '
        f'&middot; {_esc(_verdict(r))}</p>')

    for dim, meta_d in DIMENSIONS.items():
        got = _dims(r).get(dim) or {}
        score = got.get("score")
        coverage = got.get("coverage") or 0.0
        cls = "dim" if coverage else "dim unmeasured"
        shown = (f"{_num((score or 0.0) * 100, 0)}" if coverage
                 else "not measured")
        add(f'<div class="{cls}"><div class="row">'
            f'<span>{_esc(meta_d["label"])}</span>'
            f'<span>{shown}'
            + (f" &middot; {_num(coverage * 100, 0)}% covered"
               if coverage and coverage < 1.0 else "")
            + "</span></div>" + _bar(score, coverage) + "</div>")

    if weak:
        add(f'<p class="note">Weakest: {_esc(_dim_label(weak[0]))}</p>')
    for label in r.get("gates_failed") or []:
        add(f'<p class="flag">Gate failed: {_esc(label)}</p>')
    for err in r.get("errors") or []:
        add(f'<p class="flag">Error: {_esc(err)}</p>')
    for warn in r.get("warnings") or []:
        add(f'<p class="warnline">{_esc(warn)}</p>')
    fixes = r.get("remediation") or []
    if fixes:
        add(f'<p class="fix">+{_pts(_gain(r))} available:</p>'
            '<ul class="tight">')
        for item in fixes:
            label = item.get("label") or METRICS.get(
                item.get("metric"), {}).get("label", item.get("metric"))
            add(f"<li>{_esc(label)} &mdash; {_esc(item.get('note') or '')} "
                f"(+{_pts(item.get('gain') or 0.0)})</li>")
        add("</ul>")
    add("</article>")
    return "\n".join(out)


def render_html(results):
    """A self-contained page: no CDN, no fonts to fetch, no script.

    It is written next to the batch it describes and opened months later off a
    file:// path or out of a tarball, so anything it needed from a network
    would eventually render it blank. Theme follows the reader's OS, and
    `data-theme` on <html> overrides it for anyone printing or embedding.
    """
    results = list(results)
    ranked = _ranked(results)
    total_s = sum(float(r.get("duration_s") or 0.0) for r in results)
    order = _verdict_order(results)
    counts = {v: 0 for v in order}
    for r in results:
        counts[_verdict(r)] = counts.get(_verdict(r), 0) + 1

    out = []
    add = out.append
    add("<!doctype html>")
    add('<html lang="en">')
    add("<head>")
    add('<meta charset="utf-8">')
    add('<meta name="viewport" content="width=device-width, '
        'initial-scale=1">')
    add(f"<title>Lecture scan &middot; {len(results)} lectures</title>")
    add("<style>" + _CSS + "</style>")
    add("</head>")
    add("<body>")
    add('<div class="wrap">')
    add("<h1>Lecture scan</h1>")
    add(f'<p class="sub">{len(results)} lectures &middot; '
        f"{_esc(_hms(total_s))} of runtime scanned</p>")

    if not results:
        add('<p class="sub">Nothing scanned.</p>')
        add("</div></body></html>")
        return "\n".join(out) + "\n"

    add('<div class="stats">')
    for v in order:
        add(f'<div class="stat"><div class="k">{_esc(v)}</div>'
            f'<div class="v">{counts.get(v, 0)}</div></div>')
    add('<div class="stat"><div class="k">mean score</div>'
        f'<div class="v">{_pts(_mean_score(results))}'
        "</div></div>")
    add('<div class="stat"><div class="k">mean potential</div>'
        f'<div class="v">{_pts(_mean_potential(results))}'
        "</div></div>")
    add("</div>")

    # --- ranked table ----------------------------------------------------
    add("<h2>Ranked</h2>")
    add('<div class="scroll"><table>')
    add("<thead><tr>"
        '<th class="num">#</th><th>Lecture</th><th>Course</th>'
        "<th>Instructor</th><th>Grade</th>"
        '<th class="num">Score <span class="caret">&#9662;</span></th>'
        '<th class="num">Potential</th><th class="num">+Fix</th>'
        "<th>Weakest dimension</th><th>Verdict</th></tr></thead>")
    add("<tbody>")
    for i, r in enumerate(ranked, 1):
        weak = weakest_dimension(r)
        weak_txt = (f"{_dim_label(weak[0])} ({_num(weak[1] * 100, 0)})"
                    if weak else "not measured")
        score_c, pot_c, gain_c = _cells(r)
        add("<tr>"
            f'<td class="num">{i}</td>'
            f'<td class="mono">{_esc(_key(r))}</td>'
            f"<td>{_esc(r.get('course') or UNKNOWN)}</td>"
            f"<td>{_esc(r.get('owner') or UNKNOWN)}</td>"
            f"<td>{_chip(_grade(r))}</td>"
            f'<td class="num">{score_c}</td>'
            f'<td class="num">{pot_c}</td>'
            f'<td class="num">{gain_c}</td>'
            f'<td class="wide">{_esc(weak_txt)}</td>'
            f"<td>{_esc(_verdict(r))}</td></tr>")
    add("</tbody></table></div>")
    add('<p class="note">Sorted by score, failed gates last. '
        "Re-sort in the CSV export; this page carries no script.</p>")

    # --- by instructor ---------------------------------------------------
    add("<h2>By instructor</h2>")
    add('<div class="scroll"><table>')
    add("<thead><tr><th>Instructor</th>"
        '<th class="num">Lectures</th><th class="num">Mean</th>'
        '<th class="num">Mean potential</th><th>Best</th><th>Worst</th>'
        "</tr></thead><tbody>")
    for owner, group in sorted(
            _group(results, "owner").items(),
            key=lambda kv: -(_mean_score(kv[1]) or 0.0)):
        best, worst = _best_worst(group)
        add(f"<tr><td>{_esc(owner)}</td>"
            f'<td class="num">{len(group)}</td>'
            f'<td class="num">'
            f"{_pts(_mean_score(group))}</td>"
            f'<td class="num">'
            f"{_pts(_mean_potential(group))}</td>"
            f'<td class="mono">{_esc(_key(best) if best else "--")}</td>'
            f'<td class="mono">{_esc(_key(worst) if worst else "--")}</td>'
            "</tr>")
    add("</tbody></table></div>")
    add('<p class="note">Measured delivery, not teaching: room acoustics and '
        "microphone luck land in these numbers too.</p>")

    # --- lectures --------------------------------------------------------
    add("<h2>Lectures</h2>")
    add('<div class="cards">')
    for r in ranked:
        add(_html_card(r))
    add("</div>")

    scanned = sorted(r.get("scanned_at") or "" for r in results)
    add(f"<footer>Generated by src/scan/report.py against the rubric in "
        f"src/scan/rubric.py. Last scan {_esc(scanned[-1] or 'unknown')}."
        "</footer>")
    add("</div>")
    add("</body></html>")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# One lecture, in full
# --------------------------------------------------------------------------

def _rule(char, width):
    return char * width


def render_lecture(result, width=78):
    """Everything measured about one lecture. `--lecture-dir X --explain`.

    Prints every rubric metric that was measured with its raw value, its
    sub-score and -- when the cohort was big enough for one to mean anything
    -- its percentile within the scan. The percentile is the number to trust:
    the absolute bands are calibrated against two lectures, and the docstring
    of rubric.py says so.
    """
    if not result:
        return "No result to explain.\n"

    r = result
    out = []
    add = out.append
    pcts = r.get("percentiles")

    add(_rule("=", width))
    add(_key(r))
    if r.get("title"):
        add(r["title"])
    add(_rule("=", width))
    add(f"  course     {r.get('course') or UNKNOWN}")
    add(f"  instructor {r.get('owner') or UNKNOWN}")
    add(f"  directory  {r.get('dir') or UNKNOWN}")
    add(f"  runtime    {_hms(r.get('duration_s'))}")
    add(f"  scanned    {r.get('scanned_at') or UNKNOWN}"
        f"   tiers: {', '.join(r.get('tiers_run') or []) or 'none'}")
    add("")
    add(f"  GRADE {_grade(r)}   score {_cells(r)[0]}"
        f"   potential {_cells(r)[1]}"
        f"   verdict {_verdict(r)}")
    blurb = next((b for t, letter, _v, b in GRADES if letter == _grade(r)
                  and _score(r) >= t), None)
    if blurb:
        for line in rubric._wrap(blurb, width - 4):
            add(f"    {line}")
    add("")

    # --- gates -----------------------------------------------------------
    failed = r.get("gates_failed") or []
    add(_rule("-", width))
    add("GATES")
    add(_rule("-", width))
    if not failed:
        add("  all passed")
    for label in failed:
        gate = next((g for g in rubric.GATES
                     if g["label"] == label or g["id"] == label), None)
        add(f"  FAILED  {label}")
        if gate:
            for line in rubric._wrap(gate["why"], width - 12):
                add(f"          {line}")
    add("")

    # --- dimensions ------------------------------------------------------
    add(_rule("-", width))
    add("DIMENSIONS")
    add(_rule("-", width))
    for dim, meta in DIMENSIONS.items():
        got = _dims(r).get(dim) or {}
        coverage = got.get("coverage")
        score = got.get("score")
        weight = got.get("weight", meta["weight"])
        # 36 is the longest label in DIMENSIONS ("Student exposure & burden
        # (inverted)"), and a truncated dimension name reads as a different
        # dimension, so the bar gives up the width instead.
        label = meta["label"]
        if not coverage:
            add(f"  {label:<36} not measured    w{weight:.0%}")
            continue
        # 12 cells is enough to see the shape of a dimension at a glance and
        # narrow enough that the whole block stays inside 79 columns.
        filled = int(round(max(0.0, min(1.0, score or 0.0)) * 12))
        bar = "#" * filled + "." * (12 - filled)
        add(f"  {label:<36} {bar} {(score or 0.0) * 100:5.1f}"
            f"  w{weight:.0%} cov {coverage:.0%}")
    add("")

    # --- metrics ---------------------------------------------------------
    metrics = r.get("metrics") or {}
    subs = r.get("subscores") or {}
    add(_rule("-", width))
    # An empty percentiles dict is not the same as no percentiles: the first
    # means this lecture measured nothing rankable, the second means the scan
    # was too small to rank anything at all. Only the second is a caveat.
    add("METRICS" + ("" if pcts is not None else
                     f"   (no cohort ranks: needs {MIN_COHORT}+ lectures)"))
    add(_rule("-", width))
    for dim, meta in DIMENSIONS.items():
        ids = rubric.dimension_ids(dim)
        add(f"  [{meta['label']}]")
        for mid in ids:
            spec = METRICS[mid]
            raw = metrics.get(mid)
            sub = subs.get(mid)
            # Value column is a minimum, not a maximum: a couple of units are
            # long enough ("Laplacian variance @480x270") to push the line
            # out, and a truncated measurement is worse than a ragged column.
            line = f"    {spec['label'][:28]:<28} {_fmt_raw(mid, raw):>25}"
            if sub is not None:
                line += f" {sub * 100:5.1f}"
            else:
                line += "     -"
            pct = (pcts or {}).get(mid)
            if pct is not None:
                line += f" p{pct:>3.0f}"
            add(line)
        add("")

    # --- remediation -----------------------------------------------------
    add(_rule("-", width))
    add(f"REMEDIATION   +{_pts(_gain(r))} points available")
    add(_rule("-", width))
    fixes = r.get("remediation") or []
    if not fixes:
        add("  nothing the pipeline can fix after the fact.")
    for item in fixes:
        label = item.get("label") or METRICS.get(
            item.get("metric"), {}).get("label", item.get("metric"))
        add(f"  +{_pts(item.get('gain') or 0.0):>5}  {label}")
        note = item.get("note")
        if note:
            for line in rubric._wrap(note, width - 12):
                add(f"          {line}")
    add("")

    # --- warnings and errors ---------------------------------------------
    for title, items in (("WARNINGS", r.get("warnings") or []),
                         ("ERRORS", r.get("errors") or [])):
        if not items:
            continue
        add(_rule("-", width))
        add(title)
        add(_rule("-", width))
        for item in items:
            for i, line in enumerate(rubric._wrap(str(item), width - 4)):
                add(f"  {line}" if i == 0 else f"    {line}")
        add("")

    return "\n".join(out).rstrip() + "\n"
