"""Fill assets/brand/courses.json from CMU's official Schedule of Classes.

    python scripts/fetch_course_titles.py --manifests manifests/
    python scripts/fetch_course_titles.py --courses 15-210 17-635 --semester S26

The Scene A rail draws a full course title under the course code, and Panopto
does not supply one -- it gives a course NUMBER and a lecture name and nothing
else. Until now that title came from a file somebody typed two entries into by
hand, which is fine for two courses and is why 32 of the 34 courses in the
Spring 2026 corpus would have rendered with no title at all.

Where the titles come from
--------------------------
The registrar publishes the whole Schedule of Classes as a plain tab-separated
dump, unauthenticated:

    https://enr-apps.as.cmu.edu/assets/SOC/sched_layout_spring.dat

One request returns every course CMU is running that semester -- 532 kB, about
five thousand courses -- as lines of the shape

    \\t15210\\tParallel and Sequential Data Structures and Algorithms

That is the official source, not a scrape of a rendered page, so it does not
break when the site is restyled. Anything the dump does not carry -- a course
not offered this term, which is exactly the archived case -- falls back to the
per-course endpoint, which serves any semester:

    https://enr-apps.as.cmu.edu/open/SOC/SOCServlet/courseDetails?COURSE=15210&SEMESTER=S26

Both are read-only, public, and need no credential, so this can run anywhere.

What it will not do
-------------------
Overwrite a title somebody has already tuned. `15-210` in courses.json carries
an explicit newline -- "Parallel and Sequential\\nData Structures and
Algorithms" -- because that is the break in Phillip's proof render, and the
registrar's single-line string would silently replace it with whatever the
renderer's own balancing produces. Existing entries are kept unless
--overwrite is passed, and the merge says what it skipped.
"""

import argparse
import glob
import json
import os
import re
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COURSES_JSON = os.path.join(ROOT, "assets", "brand", "courses.json")

SOC_DUMP = "https://enr-apps.as.cmu.edu/assets/SOC/sched_layout_{season}.dat"
SOC_DETAIL = ("https://enr-apps.as.cmu.edu/open/SOC/SOCServlet/courseDetails"
              "?COURSE={num}&SEMESTER={semester}")

# A course line in the dump: a tab, five digits, a tab, the title. Section and
# meeting-time lines under it start with tabs and no course number, so this
# picks out exactly the course headers.
#
# The title is taken as the tab-delimited FIELD, not "the rest of the line".
# Most course headers are just number and title, but some carry the units and
# meeting columns on the same line, and matching to end-of-line pulled those
# in: 10-725 came back as "Optimization for Machine Learning:\t12.0". Both
# titles that broke happen to end in a colon, so the join looked plausible.
COURSE_LINE = re.compile(r"^\t(\d{5})\t")
# The per-course page states both in one attribute:
#   data-maintitle="15210&nbsp;&nbsp;Parallel and Sequential Data Structures..."
DETAIL_TITLE = re.compile(r'data-maintitle="(\d{5})(?:&nbsp;|\s)+(.*?)"')

SEASONS = {"S": "spring", "M": "summer", "F": "fall"}
TIMEOUT = 60


def _get(url):
    with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
        return r.read().decode("utf-8", "replace")


def normalise(code):
    """'10-301-601' or '15210' -> '15-210'. None if it is not a course code.

    The manifests carry whatever Panopto's folder was called, which includes
    cross-listed codes with a section suffix ('10-301-601'), a department
    initialism ('HCII') and one truncated to nothing ('17-'). Only the first
    is recoverable, and the other two are not errors to report -- they are
    folders that are not courses.
    """
    digits = re.sub(r"\D", "", code or "")
    if len(digits) < 5:
        return None
    return f"{digits[:2]}-{digits[2:5]}"


def from_dump(season):
    """{'15-210': 'Parallel and Sequential ...'} for a whole semester."""
    text = _get(SOC_DUMP.format(season=season))
    out = {}
    for line in text.splitlines():
        m = COURSE_LINE.match(line)
        if not m:
            continue
        fields = line.split("\t")
        title = fields[2].strip() if len(fields) > 2 else ""
        if title:
            out.setdefault(normalise(m.group(1)), title)
    out.pop(None, None)
    return out


def from_detail(code, semester):
    """One course, from the endpoint that serves any semester. None if absent."""
    num = re.sub(r"\D", "", code)[:5]
    try:
        html = _get(SOC_DETAIL.format(num=num, semester=semester))
    except (urllib.error.URLError, OSError) as e:
        print(f"  {code}: detail lookup failed ({e})")
        return None
    m = DETAIL_TITLE.search(html)
    return m.group(2).strip() if m else None


def courses_from_manifests(pattern):
    """Every course code the manifests mention, normalised and deduplicated."""
    found = {}
    for path in sorted(glob.glob(os.path.join(pattern, "manifest.*.json"))):
        raw = os.path.basename(path)[len("manifest."):-len(".json")]
        code = normalise(raw)
        if code:
            found[code] = raw
        else:
            print(f"  skipping {raw!r}: not a course code")
    return found


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--manifests", default=None,
                    help="Directory of manifest.<course>.json to take the "
                         "course list from")
    ap.add_argument("--courses", nargs="*", default=[],
                    help="Course codes instead of, or as well as, --manifests")
    ap.add_argument("--semester", default="S26",
                    help="Semester code for the fallback lookup, e.g. S26, "
                         "F25. Its first letter also picks which dump to read.")
    ap.add_argument("--out", default=COURSES_JSON)
    ap.add_argument("--overwrite", action="store_true",
                    help="Replace titles already in the file. Off by default: "
                         "an existing entry may carry a hand-placed line break "
                         "the registrar's single-line string would destroy.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    wanted = {}
    if args.manifests:
        wanted.update(courses_from_manifests(args.manifests))
    for c in args.courses:
        code = normalise(c)
        if code:
            wanted[code] = c
        else:
            print(f"  skipping {c!r}: not a course code")
    if not wanted:
        raise SystemExit("no course codes to look up -- pass --manifests "
                         "and/or --courses")

    season = SEASONS.get(args.semester[:1].upper(), "spring")
    print(f"[courses] {len(wanted)} course(s); reading the {season} Schedule "
          f"of Classes dump")
    try:
        dump = from_dump(season)
    except (urllib.error.URLError, OSError) as e:
        print(f"[courses] dump unavailable ({e}); falling back per course")
        dump = {}
    else:
        print(f"[courses] dump carries {len(dump)} courses")

    doc = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            doc = json.load(f)

    added, kept, missing = [], [], []
    for code in sorted(wanted):
        existing = doc.get(code)
        if existing and not args.overwrite:
            kept.append(code)
            continue
        title = dump.get(code) or from_detail(code, args.semester)
        if not title:
            missing.append(code)
            continue
        doc[code] = {"title": title}
        added.append((code, title))

    for code, title in added:
        print(f"  + {code}  {title}")
    if kept:
        print(f"[courses] kept {len(kept)} existing entr(y/ies) "
              f"({', '.join(kept)}); --overwrite to replace")
    if missing:
        print(f"[courses] NO TITLE for {len(missing)}: {', '.join(missing)}. "
              f"The rail draws the code and lecture line without one, so this "
              f"degrades rather than fails.")

    if args.dry_run:
        print("[courses] --dry-run: not writing")
        return 0
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"[courses] wrote {args.out} ({len(added)} added, {len(kept)} kept, "
          f"{len(missing)} missing)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
