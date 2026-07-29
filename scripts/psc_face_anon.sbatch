#!/bin/bash
# Facial anonymization of the lecture camera stream, on Bridges-2.
#
# This is one of only two stages that genuinely needs a GPU (the other is
# transcription): the cost is almost entirely InsightFace ONNX inference
# (measured 98.9% of runtime), and that runs on CUDA. It does NOT need NVENC --
# the h264 encode is a rounding error here and goes through libx264 on the
# node's CPUs. Every other stage is ffmpeg/CPU and belongs on RM-shared, except
# cards, which is local-only.
#
# A full GPU node has 8 GPUs, so --chunks 8 pins one worker per GPU via
# CUDA_VISIBLE_DEVICES and gives ~8x. Combined with dropping the three unused
# buffalo_l models (3.95x) and --detect-every 4, that is the difference between
# "does not finish" and a few minutes of inference.
#
# GPU TYPE IS NOT OPTIONAL HERE. The GPU partition is heterogeneous (v100-16,
# v100-32, l40s-48, h100-80, all :8 per node) and this env CANNOT RUN ON V100.
# A previous attempt on v017 died after 3 seconds:
#
#   err 209 no kernel image is available for execution on the device
#   CUDNN_FE failure 11: CUDNN_BACKEND_API_FAILED ; GPU=0
#   Non-zero status code returned while running Conv node. Name:'Conv_0'
#
# V100 is compute capability sm_70, and the cuDNN shipped with
# onnxruntime-gpu 1.28.0 has no kernels for it. L40S is sm_89 and H100 is sm_90,
# both supported -- and both are faster for inference anyway. That crash is the
# real reason this stage "could not run on PSC"; it was never about speed.
#
# If you must use a V100, downgrade instead: an older onnxruntime-gpu (CUDA 11
# era) still ships sm_70 kernels.
#
# Check the cost model on the real node before committing to a long run:
#     python -m src.video.face_anon --lecture-dir data/15210-lecture12 --benchmark
#
# Verify who the instructor is before a full run (writes face_clusters.png):
#     python -m src.video.face_anon --lecture-dir data/15210-lecture12 --preview
#
# Submit:  sbatch scripts/psc_face_anon.sbatch
#
#SBATCH --job-name=face-anon
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --gpus=l40s-48:8      # NOT v100 - see note above (sm_70 unsupported)
#SBATCH --time=01:00:00
#SBATCH --output=logs/face-anon-%j.out
#SBATCH --error=logs/face-anon-%j.out

set -euo pipefail

PROJECT="${PROJECT:-/ocean/projects/cis260220p/$USER}"
REPO="${REPO:-$PROJECT/scs-learn-video-pipeline}"
LECTURE_DIR="${LECTURE_DIR:-data/15210-lecture12}"
STREAM_INPUT="${STREAM_INPUT:-camera_muted.mp4}"   # run after audio.py so the
                                                  # muted audio is carried over
CHUNKS="${CHUNKS:-8}"
DETECT_EVERY="${DETECT_EVERY:-4}"

cd "$REPO"
mkdir -p logs

module load anaconda3 2>/dev/null || true
source activate scs-video                # separate env from scs-learn: it has
                                         # insightface + onnxruntime-gpu, which
                                         # conflicts with the audio env's
                                         # plain onnxruntime

module load ffmpeg 2>/dev/null || true
command -v ffmpeg >/dev/null || { echo "ERROR: no ffmpeg on PATH" >&2; exit 1; }

# onnxruntime-gpu takes its CUDA dependencies as nvidia-* pip packages, which
# install under site-packages/nvidia/*/lib -- a path the dynamic loader never
# searches. Without this, libonnxruntime_providers_cuda.so fails to load
# ("libcublasLt.so.12: cannot open shared object file"), onnxruntime falls back
# to CPU *silently*, and the job runs ~25x slower while looking healthy.
NVIDIA_LIBS=$(python -c "import glob,os,site;print(':'.join(sorted({p for d in site.getsitepackages() for p in glob.glob(os.path.join(d,'nvidia','*','lib'))})))" 2>/dev/null || true)
if [ -n "${NVIDIA_LIBS:-}" ]; then
    export LD_LIBRARY_PATH="$NVIDIA_LIBS:${LD_LIBRARY_PATH:-}"
    echo "LD_LIBRARY_PATH += env-local NVIDIA libs"
else
    module load cuda/12.6.1 2>/dev/null && echo "loaded cuda/12.6.1 module"
fi

echo "=============================================================="
echo "node    : $(hostname)"
echo "cores   : ${SLURM_CPUS_ON_NODE:-?}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
python -c "import onnxruntime as o; print('onnxruntime providers:', o.get_available_providers())"
echo "=============================================================="

# Sanity-check that CUDA is actually being used. Without onnxruntime-gpu this
# silently falls back to CPU, which is ~25x slower and would blow the walltime.
python - <<'PY'
# Presence of the CUDA provider is NOT enough: on a V100 the provider loads
# fine and then fails on the first convolution. Actually run one.
import sys
import numpy as np
import onnxruntime as ort
if "CUDAExecutionProvider" not in ort.get_available_providers():
    sys.exit("ERROR: onnxruntime has no CUDAExecutionProvider.\n"
             "  pip uninstall -y onnxruntime && pip install onnxruntime-gpu")
try:
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"],
                       providers=["CUDAExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    app.get(np.zeros((720, 1280, 3), dtype=np.uint8))
except Exception as e:
    sys.exit("ERROR: CUDA inference failed on this GPU.\n"
             "  %s\n"
             "  'no kernel image is available' means this GPU's compute\n"
             "  capability is unsupported by the installed cuDNN/onnxruntime.\n"
             "  V100 is sm_70 and does NOT work with onnxruntime-gpu 1.28.\n"
             "  Resubmit with --gpus=l40s-48:8 or --gpus=h100-80:8." % e)
print("CUDA inference works on this GPU")
PY

if [ ! -f "$LECTURE_DIR/$STREAM_INPUT" ]; then
  echo "NOTE: $LECTURE_DIR/$STREAM_INPUT missing, falling back to camera.mp4" >&2
  STREAM_INPUT="camera.mp4"
fi

srun python -m src.video.face_anon \
  --lecture-dir "$LECTURE_DIR" \
  --input "$STREAM_INPUT" \
  --chunks "$CHUNKS" \
  --detect-every "$DETECT_EVERY" \
  --method pixelate

echo "done: $LECTURE_DIR/$(basename "${STREAM_INPUT%.*}")_anon.mp4"
