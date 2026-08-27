#!/bin/bash
# Take a directory of harvested manifests all the way to a ranked report.
#
#     ./scripts/rank_semester.sh manifests-f26 f26 "Fall 2026"
#
# Everything between a harvest and a PDF, unattended: download and signal-scan
# every course, wait, deepen to the vision tier in the other conda env, wait,
# pull the results home, and render the reports.
#
# It exists because doing this by hand for Spring 2026 meant sitting on
# `squeue` for four hours across two passes, and the only judgement involved
# was "has it drained yet". The harvest before it still needs a human -- one
# CMU SSO login and one Duo push -- and that is the only part that does.
#
# Arguments:
#   $1  local manifest directory                      (required)
#   $2  short semester tag, e.g. f26                  (required)
#   $3  human semester name for the report title      (optional)
#
# The tag is what keeps semesters apart on /ocean. Course codes REPEAT --
# 15-210 runs in both Spring and Fall -- so without a per-semester corpus root
# a Fall job would find Spring's camera.mp4 already downloaded, skip the
# fetch, and scan last term's lecture under this term's name. Nothing errors;
# the ranking is just silently wrong.
set -uo pipefail

MANIFEST_DIR="${1:?usage: $0 <manifest-dir> <tag> [semester-name]}"
TAG="${2:?usage: $0 <manifest-dir> <tag> [semester-name]}"
SEMESTER="${3:-$TAG}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

PSC_USER="$(grep -E '^PSC_USER=' .env 2>/dev/null | cut -d= -f2- | tr -d '"')"
[ -n "$PSC_USER" ] || { echo "PSC_USER not set in .env" >&2; exit 1; }
OCEAN="/ocean/projects/cis260220p/$PSC_USER"

export CORPUS_ROOT="$OCEAN/corpus-$TAG"
export MANIFEST_DIR_REMOTE="$OCEAN/manifests-$TAG"
CACHE="reports/psc-cache-$TAG"
OUT="reports/semester-$TAG"

# How long to wait between squeue polls, and the ceiling. The signal pass over
# 34 courses took 20 minutes and the vision pass took over four hours, so the
# cap is generous; the jobs carry their own --time and will die before this.
POLL="${POLL:-120}"
MAX_WAIT="${MAX_WAIT:-28800}"

drain() {
    local waited=0
    while :; do
        n=$(./scripts/psc.sh run 'squeue -u $USER -h 2>/dev/null | wc -l' \
              2>/dev/null | tr -d ' \r')
        [ "${n:-1}" = "0" ] && { echo "  drained after $((waited / 60))m"; return 0; }
        if [ "$waited" -ge "$MAX_WAIT" ]; then
            echo "  STILL $n job(s) running after $((waited / 60))m -- giving up on" >&2
            echo "  waiting. They are not cancelled; re-run this script or just" >&2
            echo "  scripts/semester_report.py once they finish." >&2
            return 1
        fi
        printf "  %s  %s job(s) running\n" "$(date +%H:%M:%S)" "$n"
        sleep "$POLL"
        waited=$((waited + POLL))
    done
}

echo "=============================================================="
echo "$SEMESTER  --  manifests=$MANIFEST_DIR  tag=$TAG"
echo "corpus  : $CORPUS_ROOT"
echo "reports : $OUT"
echo "=============================================================="

# Refuse to start on top of somebody else's jobs. Every wait below is "is the
# queue empty", so an unrelated job already running would make the first drain
# return only when THAT finished, and the pass after it would start against a
# half-scanned corpus.
running=$(./scripts/psc.sh run 'squeue -u $USER -h 2>/dev/null | wc -l' \
            2>/dev/null | tr -d ' \r')
if [ "${running:-0}" != "0" ]; then
    echo "ERROR: $running job(s) already queued or running. This script waits" >&2
    echo "on an EMPTY queue, so it cannot tell them from its own. Wait for" >&2
    echo "them, or cancel them, then re-run." >&2
    exit 1
fi

echo
echo ">>> [1/4] signal tier (scs-learn): download + audio/slide measurement"
./scripts/scan_all_courses.sh "$MANIFEST_DIR" signal scs-learn || exit 1
drain || exit 1

echo
echo ">>> [2/4] vision tier (scs-video): instructor and student faces"
# The second pass re-uses the first's media and cached tiers, so it only adds
# what is new. It needs the OTHER env: insightface lives in scs-video, and
# installing it into scs-learn would drag in the CPU onnxruntime that shadows
# onnxruntime-gpu for face_anon.
./scripts/scan_all_courses.sh "$MANIFEST_DIR" vision scs-video || exit 1
drain || exit 1

echo
echo ">>> [3/4] pulling scan.json and metadata.json home (never the media)"
mkdir -p "$CACHE"
rsync -az -e "ssh -o BatchMode=yes" \
    --include='*/' --include='scan.json' --include='metadata.json' \
    --exclude='*' "psc-dtn:$CORPUS_ROOT/" "$CACHE/" || exit 1
echo "  $(find "$CACHE" -name scan.json | wc -l | tr -d ' ') lecture(s) cached"

echo
echo ">>> [4/4] reports"
PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY=python3
"$PY" scripts/semester_report.py --cache "$CACHE" --out "$OUT" --absolute || exit 1
"$PY" scripts/semester_report.py --cache "$CACHE" --out "$OUT-relative" || exit 1

echo
echo "=============================================================="
echo "done. $OUT/course-ranking.pdf  (absolute scores)"
echo "      $OUT-relative/course-ranking.pdf  (recalibrated to this cohort)"
echo "=============================================================="
