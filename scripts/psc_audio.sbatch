#!/bin/bash
# Mute every non-instructor voice in the camera's audio track, on Bridges-2.
#
# CPU only -- this is ffmpeg applying a volume envelope over the spans
# transcription classified as student speech. It is on RM-shared rather than
# folded into either GPU job because it needs no GPU, and the Regular
# allocation has 5,000 SU against ~495 GPU-hours. Five minutes of an 8-GPU node
# to run ffmpeg is 0.7 GPU-hours of a scarce grant spent on nothing.
#
# It exists as a job at all so the three PSC stages can chain without a human
# in the loop:
#
#     A=$(sbatch --parsable scripts/psc_transcription.sbatch)
#     B=$(sbatch --parsable --dependency=afterok:$A scripts/psc_audio.sbatch)
#          sbatch            --dependency=afterok:$B scripts/psc_face_anon.sbatch
#
# The dependency is afterok, not afterany, on purpose: face_anon consuming a
# camera_muted.mp4 that a failed audio stage left half-written would produce an
# anonymized video with a student still audible, which is the one failure this
# pipeline exists to prevent.
#
#SBATCH --job-name=audio-mute
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/audio-%j.out
#SBATCH --error=logs/audio-%j.out

set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
LECTURE_DIR="${LECTURE_DIR:-data/15210-lecture12}"

cd "$REPO"
mkdir -p logs

module load anaconda3 2>/dev/null || true
source activate scs-learn

# The conda ffmpeg, not the module. See psc_face_anon.sbatch: the module is a
# singularity wrapper whose /bil bind mount fails on some nodes, and this stage
# is nothing but ffmpeg.
if [ -x "${CONDA_PREFIX:-}/bin/ffmpeg" ]; then
    export PATH="$CONDA_PREFIX/bin:$PATH"
    echo "ffmpeg  : $CONDA_PREFIX/bin/ffmpeg (native conda build)"
else
    module load ffmpeg 2>/dev/null || true
    echo "ffmpeg  : $(command -v ffmpeg) (PSC module -- singularity wrapper)"
fi
command -v ffmpeg >/dev/null || { echo "ERROR: no ffmpeg on PATH" >&2; exit 1; }

echo "=============================================================="
echo "node    : $(hostname)"
echo "lecture : $LECTURE_DIR"
echo "=============================================================="

# Stop here rather than muting nothing. audio.py reads the classified
# transcript to decide what to silence; without it there is no list of student
# spans, and a "successful" run would hand face_anon a camera with every
# question still audible.
if [ ! -f "$LECTURE_DIR/transcript_classified.json" ]; then
    echo "ERROR: no $LECTURE_DIR/transcript_classified.json -- transcription" >&2
    echo "has not run, so nothing knows which speech is a student's." >&2
    exit 1
fi

srun python -m src.audio.audio --lecture-dir "$LECTURE_DIR"

echo "done: $LECTURE_DIR/camera_muted.mp4"
