#!/bin/bash
# Download a semester of Panopto lectures and grade them, on Bridges-2.
#
# This is the corpus-triage job: given a harvested manifest of ~45 lectures
# that are NOT on disk anywhere yet, fetch both streams and run src/scan over
# them so we know which recordings are worth putting through the anonymization
# pipeline at all. It exists to move the download off a home hotspot -- 45
# lectures is tens of gigabytes, and Bridges-2 is on a research backbone.
#
# CPU ONLY, AND DELIBERATELY SO. The probe and signal tiers are ffprobe,
# ffmpeg audio/keyframe passes and numpy; nothing here touches CUDA. The
# Regular allocation has 5,000 SU against ~495 GPU-hours for the entire grant,
# so a GPU node spent on curl and ffmpeg would burn a visible fraction of the
# scarce budget to do work RM-shared does for free. If you later want
# --tier vision over the survivors, note that it still runs on CPU here (it is
# 200 frames a lecture, not 120,000) -- do not "upgrade" this to the GPU
# partition. The GPU stages are transcription, face_anon and track_instructor,
# and they have their own job scripts.
#
# The media stays on /ocean. Nothing in this job copies video back to a laptop;
# only scan.json and the reports are small enough to be worth moving, and
# scripts/psc_scan_README.md covers pulling those.
#
# WHY THE DOWNLOAD RUNS ON A COMPUTE NODE AND NOT THE DTN.
# The login-node ban is on *initiating transfers from a login node*, and this
# is not a login node. Bridges-2 compute nodes have outbound HTTPS -- the
# transcription stage already relies on it to pull the gated pyannote weights
# from HuggingFace mid-job. The manifest's CloudFront URLs need no
# authentication once harvested (see src/panopto_download.py's docstring), so
# there is no credential to smuggle onto the node and nothing to authenticate.
# The DTN is for moving files *between here and your laptop*; this is an
# internet fetch that happens to land on /ocean, and it belongs where the
# scan that consumes it runs.
#
# CORES: 32. The scanner's parallelism is per lecture, and ffmpeg is itself
# heavily threaded -- src/scan/__main__.py measures a single screen decode at
# 26s wall against 172s of CPU, i.e. ~6.6x self-parallelism. So the useful
# shape is a handful of lectures in flight, each with several cores to itself:
# 32 cores / JOBS=6 is ~5 cores per lecture, which is about where a decode
# stops scaling. Asking for a whole 128-core RM node would leave three
# quarters of it idle and bill 128 core-hours an hour for it. RM-shared bills
# only the cores requested.
#
# TIME: 6h. Budget arithmetic for a 45-lecture manifest: the download is the
# long pole and is network-bound (tens of GB, unpredictable), the signal tier
# is ~40s a lecture and so ~5 minutes wall at JOBS=6, and even the vision tier
# on CPU is ~30 minutes for the set. 6h is generous cover for a slow CloudFront
# day while still being a cap -- a runaway job on 32 cores is 32 SU an hour out
# of 5,000. Raise it deliberately if you are fetching a much larger manifest;
# do not remove it.
#
# Submit -- note the SINGLE QUOTES around --export. psc.sh passes the argument
# through to the remote shell unexpanded, so $USER resolves to your PSC
# username there. Double quotes (or none) would expand it on the laptop and
# point the job at /ocean/projects/cis260220p/<your-mac-username>/.
#
#     ./scripts/psc.sh sbatch scripts/psc_scan.sbatch \
#       '--export=ALL,MANIFEST=/ocean/projects/cis260220p/$USER/manifests/manifest.15210.json,CORPUS_DIR=/ocean/projects/cis260220p/$USER/corpus/15-210,TIER=signal,LIMIT=3'
#
# Start with a small LIMIT. A first run that fetches three lectures proves the
# manifest, the network and the ffmpeg before it commits six hours to forty-five.
#
#SBATCH --job-name=lecture-scan
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --output=logs/scan-%j.out
#SBATCH --error=logs/scan-%j.out

set -euo pipefail

# --------------------------------------------------------------------------
# Inputs, all overridable via --export=ALL,KEY=VALUE
# --------------------------------------------------------------------------
#   MANIFEST    Path to a harvested manifest json (the {"lectures": [...]}
#               shape src/ingestion.py writes). REQUIRED, absolute.
#               NOTE: `psc.sh sync` excludes manifest*.json on purpose, so this
#               file will NOT arrive with the code -- scp it over the DTN
#               yourself. See scripts/psc_scan_README.md.
#   CORPUS_DIR  Where lectures are downloaded to and scanned. Must be under
#               /ocean/projects/cis260220p/$USER. REQUIRED, absolute.
#   TIER        probe | signal | vision | speech | judge. Default: signal.
#               Tiers are cumulative and cached in each lecture's scan.json,
#               so re-running deeper re-uses everything already measured.
#   JOBS        Lectures scanned in parallel. Default 6 (see CORES above).
#   DL_JOBS     Concurrent downloads. Default 4 -- CloudFront and the /ocean
#               write are the limits here, not the CPU.
#   LIMIT       Max lectures to fetch from the manifest. Default: all of them.
#               Set this small on a first run.
#   REPORT_DIR  Where the reports land. Default $CORPUS_DIR/_reports.
#   REPO        Repo checkout. Defaults to wherever sbatch was invoked.
#
# REPO is derived from SLURM_SUBMIT_DIR rather than hardcoded, exactly as in
# psc_face_anon.sbatch: a hardcoded /ocean default once disagreed with the
# repo's real location and the job died on `cd` fifteen seconds in.
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
MANIFEST="${MANIFEST:-}"
CORPUS_DIR="${CORPUS_DIR:-}"
TIER="${TIER:-signal}"
JOBS="${JOBS:-6}"
DL_JOBS="${DL_JOBS:-4}"

# Divide the cores between the scan workers rather than letting each one grab
# all of them. onnxruntime and BLAS both size their thread pools from the
# visible core count, so JOBS=8 insightface sessions on a 32-core allocation
# ask for 8x32 threads and spend the difference context-switching. ffmpeg has
# the same habit but is bounded by -threads internally.
#
# Left unset this is not a small effect: it is the difference between the
# vision tier taking minutes per lecture and taking tens of minutes.
_CORES="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-8}}"
_PER_JOB=$(( _CORES / JOBS ))
[ "$_PER_JOB" -lt 1 ] && _PER_JOB=1
export OMP_NUM_THREADS="$_PER_JOB"
export OPENBLAS_NUM_THREADS="$_PER_JOB"
export MKL_NUM_THREADS="$_PER_JOB"
export NUMEXPR_NUM_THREADS="$_PER_JOB"

# If a GPU was allocated -- submit with
#   --partition=GPU-shared --gpus=l40s-48:1
# -- make onnxruntime able to find its CUDA libraries. This is lifted from
# psc_face_anon.sbatch and it is not optional: onnxruntime-gpu ships its CUDA
# dependencies as nvidia-* pip packages under site-packages/nvidia/*/lib, a
# path the dynamic loader never searches, so without it the provider fails to
# load, onnxruntime falls back to CPU *silently*, and the job runs ~25x slower
# while reporting itself healthy.
#
# NEVER ask for a v100 here. The vision tier is face_anon's detector, and
# onnxruntime-gpu's cuDNN has no sm_70 kernels -- it dies in about three
# seconds with "err 209 no kernel image is available". l40s-48 or h100-80.
if [ -n "${SLURM_JOB_GPUS:-${SLURM_GPUS_ON_NODE:-}}" ]; then
    NVIDIA_LIBS=$(python -c "import glob,os,site;print(':'.join(sorted({p for d in site.getsitepackages() for p in glob.glob(os.path.join(d,'nvidia','*','lib'))})))" 2>/dev/null || true)
    if [ -n "${NVIDIA_LIBS:-}" ]; then
        export LD_LIBRARY_PATH="$NVIDIA_LIBS:${LD_LIBRARY_PATH:-}"
        echo "LD_LIBRARY_PATH += env-local NVIDIA libs"
    else
        module load cuda/12.6.1 2>/dev/null && echo "loaded cuda/12.6.1 module"
    fi
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
fi
LIMIT="${LIMIT:-0}"

cd "$REPO"
mkdir -p logs

if [ -z "$MANIFEST" ] || [ -z "$CORPUS_DIR" ]; then
    echo "ERROR: MANIFEST and CORPUS_DIR are both required." >&2
    echo "  sbatch --export=ALL,MANIFEST=<abs path>,CORPUS_DIR=<abs path> scripts/psc_scan.sbatch" >&2
    exit 1
fi

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: no manifest at $MANIFEST" >&2
    echo "psc.sh sync excludes manifest*.json, so it does NOT travel with the" >&2
    echo "code. Copy it over the DTN first, e.g.:" >&2
    echo "  scp manifest.15210.json psc-dtn:/ocean/projects/cis260220p/\$USER/manifests/" >&2
    exit 1
fi

# /jet/home is 25 GB and typically ~90% full; a 45-lecture corpus is tens of
# gigabytes. Refuse rather than fill the home filesystem and take out every
# other job at the same time. `Disk quota exceeded` mid-download would leave
# half-written mp4s that the scanner would then happily grade.
case "$CORPUS_DIR" in
    /ocean/*) ;;
    *) echo "ERROR: CORPUS_DIR must live under /ocean (got $CORPUS_DIR)." >&2
       echo "/jet/home is 25 GB and nearly full; lecture media does not go there." >&2
       exit 1;;
esac
REPORT_DIR="${REPORT_DIR:-$CORPUS_DIR/_reports}"
mkdir -p "$CORPUS_DIR" "$REPORT_DIR"

module load anaconda3 2>/dev/null || true
# Which env, and why it is a knob rather than a constant.
#
# The scanner's tiers straddle this repo's deliberate env split: probe, signal
# and speech need numpy and ffmpeg (scs-learn), while the vision tier borrows
# face_anon's insightface detector (scs-video). Do NOT resolve that by
# installing insightface into scs-learn -- it drags in the CPU onnxruntime,
# which shadows onnxruntime-gpu and silently drops face_anon to ~25x slower.
#
# Run two passes instead. Measurements are cached per tier in each lecture's
# scan.json, so the second pass re-uses the first's work and only adds what is
# missing:
#
#   CONDA_ENV=scs-learn TIER=signal   # download + audio/video signal
#   CONDA_ENV=scs-video TIER=vision   # faces only; signal comes from cache
#
# That ordering matters for grading, not just tidiness: signal alone covers
# only ~44% of the rubric, which is under the 55% floor score.py requires
# before it will award a grade at all, so a signal-only sweep comes back
# entirely ungraded. Adding the vision tier clears it.
CONDA_ENV="${CONDA_ENV:-scs-learn}"
source activate "$CONDA_ENV"     # the audio env by default: ffmpeg, numpy.
                                 # The vision tier borrows face_anon's detector
                                 # and will simply record a per-lecture error if
                                 # insightface is absent here, which is the
                                 # right outcome -- scs-video's onnxruntime-gpu
                                 # cannot share an env with this one.

# The conda ffmpeg, never the module. /opt/packages/ffmpeg/4.3.1/ffmpeg is a
# shell wrapper around `singularity exec -B /ocean -B /bil ...` and the /bil
# bind mount fails on compute nodes with "while getting stat for /bil:
# permission denied", so the container never starts and anything piping into
# ffmpeg dies. `source activate` already puts the env's native ffmpeg 7.1.1
# (gpl, libx264/libx265) first on PATH -- loading the module afterwards is
# precisely what shadows it. Every measurement the signal tier makes is an
# ffmpeg pass, so a broken ffmpeg here is a silently useless six-hour job.
if [ -x "${CONDA_PREFIX:-}/bin/ffmpeg" ]; then
    export PATH="$CONDA_PREFIX/bin:$PATH"
    FFMPEG_SOURCE="$CONDA_PREFIX/bin/ffmpeg (native conda build)"
else
    module load ffmpeg 2>/dev/null || true
    FFMPEG_SOURCE="$(command -v ffmpeg 2>/dev/null || echo none) (PSC module -- singularity wrapper, may fail to start on this node)"
fi

# Presence on PATH is not enough: the module wrapper resolves fine and then
# fails at exec time. Actually run it.
if ! ffmpeg -version >/dev/null 2>&1; then
    echo "ERROR: ffmpeg is not usable on this node." >&2
    echo "  resolved to: ${FFMPEG_SOURCE}" >&2
    echo "  ffmpeg -version said:" >&2
    ffmpeg -version 2>&1 | sed 's/^/    /' >&2 || true
    echo "  If that is a singularity 'stat for /bil: permission denied', you are" >&2
    echo "  on the PSC ffmpeg module. Use the scs-learn conda env's own ffmpeg" >&2
    echo "  at \$CONDA_PREFIX/bin/ffmpeg and do NOT 'module load ffmpeg'." >&2
    exit 1
fi
ffprobe -version >/dev/null 2>&1 || { echo "ERROR: ffprobe unusable (the scanner probes before it measures)" >&2; exit 1; }

# Every cache under /ocean. The defaults all point at $HOME, i.e. /jet/home's
# 25 GB: HuggingFace model weights, insightface's buffalo_l pack (~300 MB, the
# vision tier pulls it on first use), matplotlib's font cache. Any one of them
# is enough to hit `Disk quota exceeded` on a full home directory.
OCEAN_CACHE="/ocean/projects/cis260220p/$USER/.cache"
mkdir -p "$OCEAN_CACHE"
export HF_HOME="$OCEAN_CACHE/huggingface"
export XDG_CACHE_HOME="$OCEAN_CACHE/xdg"
export TORCH_HOME="$OCEAN_CACHE/torch"
export INSIGHTFACE_HOME="$OCEAN_CACHE/insightface"
export MPLCONFIGDIR="$OCEAN_CACHE/matplotlib"
export TMPDIR="$OCEAN_CACHE/tmp"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME" "$INSIGHTFACE_HOME" \
         "$MPLCONFIGDIR" "$TMPDIR"
export PYTHONUNBUFFERED=1

echo "=============================================================="
echo "job        : ${SLURM_JOB_ID:-none} on ${SLURM_JOB_PARTITION:-?}"
echo "node       : $(hostname)"
echo "cores      : ${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-?}}"
echo "repo       : $REPO"
echo "python     : $(command -v python)  ($(python -V 2>&1))"
echo "ffmpeg     : $FFMPEG_SOURCE"
echo "ffmpeg ver : $(ffmpeg -version 2>/dev/null | head -1)"
echo "manifest   : $MANIFEST ($(python -c "import json,sys;print(len(json.load(open(sys.argv[1]))['lectures']))" "$MANIFEST" 2>/dev/null || echo '?') lectures)"
echo "corpus     : $CORPUS_DIR"
echo "reports    : $REPORT_DIR"
echo "tier       : $TIER   jobs=$JOBS  dl_jobs=$DL_JOBS  limit=$([ "${LIMIT:-0}" -gt 0 ] 2>/dev/null && echo "$LIMIT" || echo all)"
echo "caches     : $OCEAN_CACHE  (nothing under /jet/home)"
df -h "$CORPUS_DIR" | tail -1 | sed 's/^/disk       : /'
echo "=============================================================="

# --------------------------------------------------------------------------
# Step 1 -- fetch. Resumable and skip-existing, so re-running this job after a
# timeout costs only the lectures that are actually missing. Re-downloading 40
# GB because a scan was re-run is the failure this whole script is shaped
# around avoiding.
# --------------------------------------------------------------------------
LIMIT_ARG=()
[ "${LIMIT:-0}" -gt 0 ] 2>/dev/null && LIMIT_ARG=(--limit "$LIMIT")

echo
echo ">>> [1/2] download  $(date '+%F %T')"
t0=$SECONDS
python scripts/scan_fetch_manifest.py \
    --manifest "$MANIFEST" \
    --out-dir "$CORPUS_DIR" \
    --jobs "$DL_JOBS" \
    --skip-existing \
    ${LIMIT_ARG[@]+"${LIMIT_ARG[@]}"}      # empty-array-safe under `set -u`
echo "<<< [1/2] download finished in $((SECONDS - t0))s"

# --------------------------------------------------------------------------
# Step 2 -- scan. No `srun`: the scanner runs one process that manages its own
# spawn-based pool, and srun under this allocation would launch a second copy
# per task, each re-scanning the same corpus and racing on the same scan.json.
# --------------------------------------------------------------------------
echo
echo ">>> [2/2] scan tier=$TIER  $(date '+%F %T')"
t0=$SECONDS
# VISION_FRAMES trades vision-tier accuracy for wall clock. The metrics it
# feeds are fractions over a whole lecture -- what share of frames hold the
# instructor, what share hold a student -- so they converge long before 200
# samples. At 80 that is still one frame every ~67s across 90 minutes, and on
# a CPU-only node (onnxruntime falling back off the GPU) it is the difference
# between a semester taking half an hour and taking two.
# FORCE_TIER re-measures one tier while keeping the rest of the cache. For
# when a tier's code changed: re-running the whole sweep would spend hours
# recomputing numbers that were never wrong, and --force does exactly that.
FORCE_ARGS=()
if [ -n "${FORCE_TIER:-}" ]; then
    for t in ${FORCE_TIER//,/ }; do
        FORCE_ARGS+=(--force-tier "$t")
    done
    echo "re-measuring tier(s): $FORCE_TIER (rest of the cache kept)"
fi

python -m src.scan \
    --courses-dir "$CORPUS_DIR" \
    --tier "$TIER" \
    --jobs "$JOBS" \
    --vision-frames "${VISION_FRAMES:-200}" \
    "${FORCE_ARGS[@]}" \
    --out "$REPORT_DIR" \
    --format md,csv,html,json
echo "<<< [2/2] scan finished in $((SECONDS - t0))s"

echo
echo "reports    : $REPORT_DIR"
ls -la "$REPORT_DIR" | sed 's/^/  /'
echo "per-lecture cache: $CORPUS_DIR/*/scan.json"
echo "done. The media stays on /ocean -- pull back only the json/md/csv/html."
