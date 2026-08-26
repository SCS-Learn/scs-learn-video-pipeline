"""Harvest a manifest for EVERY course in a semester, on one login.

    python scripts/harvest_semester.py --semester "Spring 2026" \
        --course 15-210 --course 17-635 --course "10-301/601"

    python scripts/harvest_semester.py --semester "Spring 2026" \
        --courses-file courses.txt --out-dir manifests/

Why this exists: `src/ingestion.py` harvests one course per invocation, and
each invocation opens a browser and waits for CMU SSO plus a Duo push. Rating a
whole semester therefore costs one Duo push per course, and a human sitting
there for each one -- which is the single reason a semester-wide scan is not
already a one-command job.

The authenticated Panopto session is reusable across folders, so the login is
hoisted out of the loop. One browser, one Duo push, N manifests. Everything
after that -- download, scan, rank -- an agent can drive unattended:

    ./scripts/psc.sh sbatch scripts/psc_scan.sbatch \
        '--export=ALL,MANIFEST=...,CORPUS_DIR=...,TIER=signal'

A course that fails does not abort the run. Losing a fifteen-minute login
because the fourth of six course codes had a typo is exactly the kind of thing
that makes people stop using a tool, so failures are collected and reported at
the end with the search term that missed, and every course that did resolve
keeps its manifest.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.ingestion import (build_manifest, get_assets, get_cookies,
                           get_folder_id, get_lectures)

# ingestion.py grew a login timeout (and a `.ASPXAUTH`-based wait) at some
# point; older copies have neither, and get_cookies() there takes no argument.
# Importing the constant unconditionally would make this script refuse to run
# against the older module for no better reason than a missing default, so it
# is probed rather than required.
try:
    from src.ingestion import LOGIN_TIMEOUT
except ImportError:
    LOGIN_TIMEOUT = 600.0


def _login(timeout):
    """get_cookies(), whichever signature this copy of ingestion.py has."""
    try:
        return get_cookies(timeout=timeout)
    except TypeError:
        return get_cookies()


def _safe_name(course):
    """A course code as a filename. 10-301/601 must not become a directory."""
    return course.replace("/", "-").replace(" ", "")


def _read_courses(path):
    """One course code per line; blank lines and #comments ignored."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                out.append(line)
    return out


def harvest(semester, courses, out_dir, login_timeout=LOGIN_TIMEOUT):
    os.makedirs(out_dir, exist_ok=True)
    session = _login(login_timeout)                # the one interactive step

    written, failed = [], []
    for i, course in enumerate(courses, 1):
        # Two spaces after the colon: that is how Panopto's folder titles are
        # formatted, and ingestion.py builds the same string. Getting it wrong
        # returns no folder rather than an error, so it is worth stating once.
        search_term = f"{semester}:  {course}"
        print(f"\n[{i}/{len(courses)}] {course}", flush=True)
        try:
            folder_id = get_folder_id(search_term, session)
            lectures = get_lectures(folder_id, session)
            assets = get_assets(lectures, session)
            out = os.path.join(out_dir, f"manifest.{_safe_name(course)}.json")
            build_manifest(assets, course, out_path=out)
            n = len(json.load(open(out)).get("lectures", []))
            print(f"    {n} lecture(s) -> {out}", flush=True)
            written.append((course, out, n))
        except Exception as e:                                  # noqa: BLE001
            # Keep going. The login is the expensive part and it is already
            # spent; one bad course code must not cost the other five.
            print(f"    FAILED: {e}", flush=True)
            failed.append((course, search_term, str(e)))

    print("\n" + "=" * 70)
    total = sum(n for _, _, n in written)
    print(f"{len(written)}/{len(courses)} course(s), {total} lecture(s) total")
    for course, out, n in written:
        print(f"   {course:<14} {n:>3} lectures  {out}")
    if failed:
        print(f"\n{len(failed)} course(s) failed:")
        for course, term, err in failed:
            print(f"   {course:<14} searched {term!r}")
            print(f"       {err}")
        print("\nA miss is usually the course code or the semester string, not "
              "a missing folder -- Panopto matches the folder TITLE.")
    if written:
        print("\nNext, per manifest (media stays on PSC, only json comes back):")
        print("   scp <manifest> psc-dtn:/ocean/projects/cis260220p/$USER/manifests/")
        print("   ./scripts/psc.sh sbatch scripts/psc_scan.sbatch \\")
        print("     '--export=ALL,MANIFEST=...,CORPUS_DIR=...,TIER=signal'")
    return written, failed


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--semester", required=True, help='e.g. "Spring 2026"')
    p.add_argument("--course", action="append", default=[],
                   help="Course code, repeatable. e.g. --course 15-210")
    p.add_argument("--courses-file",
                   help="File with one course code per line (# comments ok)")
    p.add_argument("--out-dir", default=".",
                   help="Where to write manifest.<course>.json")
    p.add_argument("--login-timeout", type=float, default=LOGIN_TIMEOUT,
                   help="Seconds to wait for SSO + Duo")
    args = p.parse_args()

    courses = list(args.course)
    if args.courses_file:
        courses += _read_courses(args.courses_file)
    # Preserve order but drop duplicates: harvesting the same folder twice is
    # a wasted round trip and a manifest that overwrites itself.
    seen, ordered = set(), []
    for c in courses:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    if not ordered:
        p.error("no courses given: pass --course (repeatable) or --courses-file")

    print(f"Harvesting {len(ordered)} course(s) for {args.semester}: "
          f"{', '.join(ordered)}")
    print("One browser login covers all of them.")
    _, failed = harvest(args.semester, ordered, args.out_dir,
                        login_timeout=args.login_timeout)
    return 1 if failed and len(failed) == len(ordered) else 0


if __name__ == "__main__":
    sys.exit(main())
