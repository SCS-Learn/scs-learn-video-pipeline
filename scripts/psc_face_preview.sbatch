#!/bin/bash
# Who does face_anon think the instructor is? Writes face_clusters.png.
#
# ONE GPU for a few minutes, against the full run's eight for tens of minutes.
# It samples ~400 frames, clusters the faces it finds, and draws a contact
# sheet -- one row per cluster, largest first. The largest cluster is the one
# the full run will treat as the instructor and leave unblurred; everybody else
# gets pixelated.
#
# Worth its own job because getting it wrong is expensive in both directions.
# Backwards means blurring the instructor and exposing students. Over-splitting
# means the instructor's own profile shots score below threshold and he is
# pixelated in a share of frames -- measured at 28.8% on 15-210 lecture 12
# before the prototype merge was added. Neither shows up as a failed job; both
# show up in the finished video.
#
# l40s, NOT v100: same insightface + onnxruntime-gpu as the full run, so the
# same sm_70 crash applies. See psc_face_anon.sbatch.
#
# Submit:
#     LECTURE_DIR=/ocean/.../17635-lecture13 STREAM_INPUT=camera.mp4 \
#         ./scripts/psc.sh sbatch scripts/psc_face_preview.sbatch
#
# STREAM_INPUT can be the raw camera: who the instructor is does not depend on
# whether the audio has been muted yet, so this can run in parallel with
# transcription rather than waiting behind it.
#
#SBATCH --job-name=face-preview
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=l40s-48:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/face-preview-%j.out
#SBATCH --error=logs/face-preview-%j.out

set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
LECTURE_DIR="${LECTURE_DIR:-data/15210-lecture12}"
STREAM_INPUT="${STREAM_INPUT:-camera.mp4}"

cd "$REPO"
mkdir -p logs

module load anaconda3 2>/dev/null || true
source activate scs-video

if [ -x "${CONDA_PREFIX:-}/bin/ffmpeg" ]; then
    export PATH="$CONDA_PREFIX/bin:$PATH"
fi

# onnxruntime-gpu's CUDA libs ship as nvidia-* pip packages under
# site-packages/nvidia/*/lib, which the loader never searches. Without this it
# falls back to CPU silently. Same block as psc_face_anon.sbatch.
NVIDIA_LIBS=$(python -c "import glob,os,site;print(':'.join(sorted({p for d in site.getsitepackages() for p in glob.glob(os.path.join(d,'nvidia','*','lib'))})))" 2>/dev/null || true)
if [ -n "${NVIDIA_LIBS:-}" ]; then
    export LD_LIBRARY_PATH="$NVIDIA_LIBS:${LD_LIBRARY_PATH:-}"
else
    module load cuda/12.6.1 2>/dev/null || true
fi

echo "=============================================================="
echo "node    : $(hostname)"
echo "lecture : $LECTURE_DIR"
echo "input   : $STREAM_INPUT"
nvidia-smi --query-gpu=index,name --format=csv,noheader || true
python -c "import onnxruntime as o; print('providers:', o.get_available_providers())"
echo "=============================================================="

srun python -m src.video.face_anon \
  --lecture-dir "$LECTURE_DIR" \
  --input "$STREAM_INPUT" \
  --preview

echo "done: $LECTURE_DIR/face_clusters.png"
echo "Largest cluster = the instructor. Check it before the full run."
