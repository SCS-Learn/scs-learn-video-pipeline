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


# Panopto's folder endpoint takes a free-text searchTerm and returns EVERY
# folder whose title matches, paged. ingestion.py asks it for one course and
# keeps folders[0]; asking it for the semester alone enumerates the semester.
# That is the difference between "rate the courses I can name" and "rate the
# semester", so it is worth the extra call.
FOLDER_API = "https://scs.hosted.panopto.com/Panopto/Api/Folders"
MAX_PAGES = 40


def list_semester_folders(semester, session):
    """Every folder whose title matches `semester`, as (name, id, sessions).

    Paged to exhaustion rather than trusting one response: a department's
    semester runs to dozens of folders and the endpoint returns them
    a page at a time, so stopping at page 0 would silently rate a subset --
    the same class of bug as reading folders[0] and calling it the course.
    """
    csrf = session.cookies.get("csrfToken")
    if csrf is None:
        raise RuntimeError(
            "csrfToken missing -- check that login/Duo actually completed.")
    headers = {
        "x-csrf-token": csrf,
        "x-requested-with": "XMLHttpRequest",
        "accept": "application/json, text/javascript, */*; q=0.01",
    }
    out, seen = [], set()
    for page in range(MAX_PAGES):
        r = session.get(FOLDER_API, params={
            "parentId": "null", "folderSet": 1, "searchTerm": semester,
            "includeMyFolder": "false", "includePersonalFolders": "true",
            "page": page, "sort": "Depth", "names[0]": "SessionCount",
        }, headers=headers)
        try:
            folders = r.json()
        except ValueError:
            break
        if not folders:
            break
        fresh = 0
        for f in folders:
            fid = f.get("Id")
            if not fid or fid in seen:
                continue
            seen.add(fid)
            fresh += 1
            out.append((f.get("Name") or "?", fid,
                        f.get("SessionCount") or f.get("Sessions") or 0))
        if fresh == 0:
            break                    # the endpoint is repeating itself
    return out


def course_from_folder(name, semester):
    """The course code out of a folder title like 'Spring 2026:  17-635 ...'.

    Best-effort, and only used to name the manifest file: the folder id is
    what actually gets harvested, so a title this cannot parse costs a tidy
    filename, not a course.
    """
    tail = name.split(":", 1)[1] if ":" in name else name
    tail = tail.strip()
    first = tail.split()[0] if tail.split() else ""
    return first or _safe_name(name)[:40]


def _read_courses(path):
    """One course code per line; blank lines and #comments ignored."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                out.append(line)
    return out


def harvest(semester, courses, out_dir, login_timeout=LOGIN_TIMEOUT,
            session=None, folder_ids=None):
    os.makedirs(out_dir, exist_ok=True)
    if session is None:
        session = _login(login_timeout)            # the one interactive step
    folder_ids = folder_ids or {}

    written, failed = [], []
    for i, course in enumerate(courses, 1):
        # Two spaces after the colon: that is how Panopto's folder titles are
        # formatted, and ingestion.py builds the same string. Getting it wrong
        # returns no folder rather than an error, so it is worth stating once.
        search_term = f"{semester}:  {course}"
        print(f"\n[{i}/{len(courses)}] {course}", flush=True)
        try:
            # A discovered course already has its folder id, and re-searching
            # for it by a title we parsed OUT of that folder is a needless
            # round trip that can also miss.
            folder_id = folder_ids.get(course) or get_folder_id(
                search_term, session)
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


def _discover_and_harvest(args):
    """--all-courses: enumerate the semester, then harvest what it found.

    --list-only exists because this searches on folder TITLE and returns
    everything the account can see, which on a real Panopto is not only
    courses: personal folders, meeting recordings and a department's
    administrative folders match a semester string too. Printing the list and
    stopping costs one command and saves harvesting forty folders to discover
    that six of them were somebody's Zoom calls.
    """
    session = _login(args.login_timeout)
    print(f"\nsearching Panopto for folders matching {args.semester!r} ...",
          flush=True)
    folders = list_semester_folders(args.semester, session)
    keep = [(n, i, s) for n, i, s in folders
            if (s or 0) >= args.min_sessions]
    print(f"{len(folders)} folder(s) matched, {len(keep)} with at least "
          f"{args.min_sessions} recording(s):\n")
    for name, _fid, sessions in sorted(keep, key=lambda f: -(f[2] or 0)):
        print(f"   {sessions:>4} recordings   {name}")
    if len(keep) < len(folders):
        print(f"\n   ({len(folders) - len(keep)} folder(s) below "
              f"--min-sessions {args.min_sessions}, not listed)")
    if not keep:
        print("\nNothing to harvest. Check the semester string -- Panopto "
              "matches the folder TITLE, e.g. 'Spring 2026'.")
        return 1
    if args.list_only:
        print(f"\n--list-only: stopping. Re-run without it to harvest these "
              f"{len(keep)} folder(s).")
        return 0

    courses, folder_ids = [], {}
    for name, fid, _sessions in keep:
        code = course_from_folder(name, args.semester)
        # Two folders can parse to the same code (a course with a separate
        # recitation folder). Suffix rather than silently overwrite one
        # manifest with the other.
        base, n = code, 2
        while code in folder_ids:
            code = f"{base}-{n}"
            n += 1
        courses.append(code)
        folder_ids[code] = fid
    _, failed = harvest(args.semester, courses, args.out_dir,
                        session=session, folder_ids=folder_ids)
    return 1 if failed and len(failed) == len(courses) else 0


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
    p.add_argument("--all-courses", action="store_true",
                   help="Discover every folder for the semester and harvest "
                        "all of them, instead of naming courses by hand")
    p.add_argument("--list-only", action="store_true",
                   help="With --all-courses: print what WOULD be harvested "
                        "and stop. Run this first.")
    p.add_argument("--min-sessions", type=int, default=1,
                   help="With --all-courses, skip folders holding fewer "
                        "recordings than this (default 1)")
    p.add_argument("--login-timeout", type=float, default=LOGIN_TIMEOUT,
                   help="Seconds to wait for SSO + Duo")
    args = p.parse_args()

    if args.all_courses:
        return _discover_and_harvest(args)

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
