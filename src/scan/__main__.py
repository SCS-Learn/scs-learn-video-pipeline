"""Scan a semester of lecture downloads and rank them for publication.

    # what the grader actually measures, and why each threshold is what it is
    python -m src.scan --explain-rubric

    # a whole semester, cheap tiers first, 6 lectures at a time
    python -m src.scan --courses-dir data/fall2026 --tier signal --jobs 6

    # go deeper on the survivors, then write the reports
    python -m src.scan --courses-dir data/fall2026 --tier speech \\
        --out reports/fall2026 --format md,csv,html

    # one lecture, in detail
    python -m src.scan --lecture-dir data/15210-lecture12 --explain

Tiers are cumulative and cached per lecture in `scan.json`, so the intended
workflow is to sweep the whole corpus at `--tier signal` (about 40s a lecture),
read the ranking, and only spend the vision and speech tiers on the lectures
still in contention. Re-running at a deeper tier re-uses everything already
measured.

Parallelism is per lecture. Note that ffmpeg is itself heavily threaded -- a
single screen decode measured 26s wall against 172s of CPU -- so the default
job count is deliberately a fraction of the core count. Setting --jobs to the
number of cores makes a scan slower, not faster.
"""

import argparse
import concurrent.futures as futures
import json
import multiprocessing
import os
import sys

from src.scan import discover, report, rubric
from src.scan.scanner import scan_one


def _default_jobs():
    cores = os.cpu_count() or 4
    return max(1, min(6, cores // 3))


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m src.scan", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--courses-dir", help="Root holding many lecture directories")
    p.add_argument("--lecture-dir", action="append", default=[],
                   help="A single lecture directory; repeatable")
    p.add_argument("--tier", default="speech", choices=rubric.TIERS,
                   help="How deep to scan (cumulative, cached). Default: speech")
    p.add_argument("--jobs", type=int, default=_default_jobs(),
                   help=f"Lectures in parallel (default {_default_jobs()}; "
                        f"ffmpeg is already threaded, so more is often slower)")
    p.add_argument("--force", action="store_true",
                   help="Re-measure everything, ignoring cached scan.json")
    p.add_argument("--vision-frames", type=int, default=200,
                   help="Frames to run face detection on per lecture")
    p.add_argument("--out", help="Directory to write reports into")
    p.add_argument("--format", default="md",
                   help="Comma-separated: md, csv, html, json")
    p.add_argument("--limit", type=int, help="Only show the top N in the table")
    p.add_argument("--explain", action="store_true",
                   help="Print the full per-metric breakdown for each lecture")
    p.add_argument("--explain-rubric", action="store_true",
                   help="Print the grading rubric and exit")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    if args.explain_rubric:
        print(rubric.explain())
        return 0

    dirs = list(args.lecture_dir)
    if args.courses_dir:
        dirs += discover.find_lectures(args.courses_dir)
    dirs = sorted({os.path.normpath(d) for d in dirs})
    if not dirs:
        p.error("nothing to scan: pass --courses-dir or --lecture-dir")

    if not args.quiet:
        print(f"[scan] {len(dirs)} lecture(s), tier={args.tier}, "
              f"jobs={args.jobs}", flush=True)

    payload = [(d, args.tier, args.force, args.vision_frames,
                not args.quiet and len(dirs) <= 4) for d in dirs]
    results = []
    if args.jobs <= 1 or len(dirs) == 1:
        for item in payload:
            results.append(scan_one(item))
            if not args.quiet:
                _progress(results[-1], len(results), len(dirs))
    else:
        # spawn, not fork: the vision tier holds a CoreML/onnxruntime session,
        # and forking a process that has already initialised one is a reliable
        # way to hang on macOS.
        ctx = multiprocessing.get_context("spawn")
        with futures.ProcessPoolExecutor(max_workers=args.jobs,
                                         mp_context=ctx) as pool:
            for r in pool.map(scan_one, payload):
                results.append(r)
                if not args.quiet:
                    _progress(r, len(results), len(dirs))

    results = [r for r in results if r]
    report.cohort_percentiles(results)
    # Same order the report uses: a failed gate sinks whatever it scored. A
    # lecture whose camera never downloaded still scores well on the two or
    # three probe metrics that survive, and sorting on score alone floats it
    # to the top of the list as if it were the pick of the semester.
    results.sort(key=lambda r: (bool(r.get("gates_failed")),
                                -(r.get("score") or -1), r.get("key") or ""))

    if args.explain:
        for r in results:
            print()
            print(report.render_lecture(r))

    text = report.render_markdown(results, limit=args.limit)
    print()
    print(text)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        wanted = {f.strip() for f in args.format.split(",") if f.strip()}
        written = []
        if "md" in wanted:
            written.append(_write(args.out, "scan-report.md", text))
        if "csv" in wanted:
            written.append(_write(args.out, "scan-report.csv",
                                  report.render_csv(results)))
        if "html" in wanted:
            written.append(_write(args.out, "scan-report.html",
                                  report.render_html(results)))
        if "json" in wanted:
            written.append(_write(args.out, "scan-results.json",
                                  json.dumps(results, indent=2, default=str)))
        print("\n".join(f"[scan] wrote {w}" for w in written))
    return 0


def _progress(r, i, n):
    grade = r.get("grade", "?")
    sc = r.get("score")
    pot = r.get("potential")
    bad = " GATE:" + ",".join(r["gates_failed"]) if r.get("gates_failed") else ""
    err = f" ERR:{len(r['errors'])}" if r.get("errors") else ""
    shown = f"{sc:5.1f}" if sc is not None else "  n/a"
    shown_p = f"{pot:5.1f}" if pot is not None else "  n/a"
    print(f"[scan] {i:>3}/{n}  {grade}  {shown} -> {shown_p}  "
          f"{r.get('key')}{bad}{err}", flush=True)


def _write(out_dir, name, text):
    path = os.path.join(out_dir, name)
    with open(path, "w") as f:
        f.write(text)
    return path


if __name__ == "__main__":
    sys.exit(main())
