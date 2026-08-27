#!/bin/bash
# Scan a whole semester by fanning one Slurm job out per course.
#
#     ./scripts/scan_all_courses.sh manifests/            # signal tier
#     ./scripts/scan_all_courses.sh manifests/ vision scs-video
#
# The two-course sweep ran as two sequential jobs and took ~20 minutes. Thirty
# courses done that way is most of a day, and almost all of it is waiting: the
# work is per-course and shares nothing between courses, so it has no business
# being serial. One job per manifest turns the semester into whatever the
# SLOWEST SINGLE COURSE costs -- about fifteen minutes -- as long as the
# partition has the nodes, and RM-shared usually does because every job here
# asks for a fraction of one node.
#
# Each job still does its own download, so nothing is fetched twice and a
# course that fails takes only itself down. Re-running is safe and cheap:
# media already on /ocean is skipped, and cached tiers are not recomputed.
#
# Arguments:
#   $1  directory of manifest.<course>.json          (required)
#   $2  tier: probe|signal|vision|speech             (default signal)
#   $3  conda env: scs-learn|scs-video               (default scs-learn)
#   $4  extra --export terms, comma separated        (optional)
#
# The vision tier needs insightface, which lives in scs-video, so a full pass
# is two fan-outs and the second re-uses the first's cache:
#
#     ./scripts/scan_all_courses.sh manifests/ signal scs-learn
#     ./scripts/scan_all_courses.sh manifests/ vision scs-video
set -uo pipefail

MANIFEST_DIR="${1:?usage: $0 <manifest-dir> [tier] [conda-env] [extra-export]}"
TIER="${2:-signal}"
ENV="${3:-scs-learn}"
EXTRA="${4:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

# Read the PSC username the same way psc.sh does, so the two cannot disagree
# about whose /ocean this is.
PSC_USER="$(grep -E '^PSC_USER=' .env 2>/dev/null | cut -d= -f2- | tr -d '"' )"
[ -n "$PSC_USER" ] || { echo "PSC_USER not set in .env" >&2; exit 1; }
OCEAN="/ocean/projects/cis260220p/$PSC_USER"

# Where this semester's media and manifests live. Overridable because course
# CODES REPEAT ACROSS SEMESTERS -- 15-210 runs in both Spring and Fall -- so a
# single fixed corpus/15-210 would have a Fall job find Spring's camera.mp4
# already on disk, skip the download, and scan last term's lecture under this
# term's name. Nothing would error; the ranking would just be wrong.
#
#     CORPUS_ROOT=$OCEAN/corpus-f26 MANIFEST_DIR_REMOTE=$OCEAN/manifests-f26 \
#         ./scripts/scan_all_courses.sh manifests-f26/ signal scs-learn
CORPUS_ROOT="${CORPUS_ROOT:-$OCEAN/corpus}"
MANIFEST_DIR_REMOTE="${MANIFEST_DIR_REMOTE:-$OCEAN/manifests}"

# Per-job width. Sized so ~30 courses can be RUNNING rather than queued;
# PER_JOB_WORKERS then divides those cores between scan workers, which is the
# same oversubscription lesson psc_scan.sbatch already carries.
PER_JOB_CORES="${PER_JOB_CORES:-16}"
PER_JOB_WORKERS="${PER_JOB_WORKERS:-4}"

shopt -s nullglob
MANIFESTS=("$MANIFEST_DIR"/manifest.*.json)
[ ${#MANIFESTS[@]} -gt 0 ] || { echo "no manifests in $MANIFEST_DIR" >&2; exit 1; }

echo "=============================================================="
echo "fanning out $((${#MANIFESTS[@]})) course(s)   tier=$TIER  env=$ENV"
echo "corpus: $CORPUS_ROOT"
echo "=============================================================="

# One transfer for the lot. Thirty scp calls over the DTN is thirty round
# trips for a few megabytes.
./scripts/psc.sh run "mkdir -p '$MANIFEST_DIR_REMOTE' '$CORPUS_ROOT'" >/dev/null || exit 1
echo "copying manifests to the DTN ..."
rsync -az -e "ssh -o BatchMode=yes" "${MANIFESTS[@]}" \
    "psc-dtn:$MANIFEST_DIR_REMOTE/" || exit 1

SUBMITTED=0
SKIPPED=0
JOBIDS=()
for m in "${MANIFESTS[@]}"; do
    base="$(basename "$m")"
    course="${base#manifest.}"
    course="${course%.json}"

    # An empty manifest is a folder that matched the semester but holds no
    # lectures -- a department shell, somebody's personal folder. Submitting a
    # job to download nothing wastes a queue slot and produces a report about
    # zero lectures, so skip it here where it is one grep.
    n=$(python3 -c "import json,sys;print(len(json.load(open(sys.argv[1])).get('lectures',[])))" "$m" 2>/dev/null || echo 0)
    if [ "${n:-0}" -lt 1 ]; then
        echo "  skip  $course (no lectures)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    export_terms="ALL,CONDA_ENV=$ENV,MANIFEST=$MANIFEST_DIR_REMOTE/$base"
    export_terms="$export_terms,CORPUS_DIR=$CORPUS_ROOT/$course"
    export_terms="$export_terms,TIER=$TIER,JOBS=$PER_JOB_WORKERS,DL_JOBS=6"
    export_terms="$export_terms,VISION_FRAMES=80"
    [ -n "$EXTRA" ] && export_terms="$export_terms,$EXTRA"

    # Deliberately a SMALLER slice than the single-course job asks for. Thirty
    # jobs at 32 cores is a thousand cores, which RM-shared will queue rather
    # than run, and a queued job is infinitely slower than a narrow one. Half
    # the width each lets them all run at once, and the semester finishes in
    # the time one course takes rather than in submission order.
    out=$(./scripts/psc.sh sbatch scripts/psc_scan.sbatch \
            "--job-name=scan-$course --cpus-per-task=$PER_JOB_CORES --export=$export_terms" 2>&1 | tail -1)
    jid="$(echo "$out" | grep -oE '[0-9]{6,}' | tail -1)"
    if [ -n "$jid" ]; then
        echo "  $jid  $course  ($n lectures)"
        JOBIDS+=("$jid")
        SUBMITTED=$((SUBMITTED + 1))
    else
        echo "  FAILED to submit $course: $out"
    fi
done

echo "=============================================================="
echo "$SUBMITTED submitted, $SKIPPED skipped (empty)"
[ ${#JOBIDS[@]} -gt 0 ] && echo "jobs: ${JOBIDS[*]}"
echo
echo "watch:   ./scripts/psc.sh run 'squeue -u \$USER'"
echo "results: rsync -az -e 'ssh -o BatchMode=yes' --include='*/' \\"
echo "           --include='scan.json' --include='metadata.json' --exclude='*' \\"
echo "           psc-dtn:$CORPUS_ROOT/ reports/psc-cache/"
