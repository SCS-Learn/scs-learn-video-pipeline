#!/bin/bash
# WhisperX ASR + pyannote diarization + question classification, on Bridges-2.
#
# The second of the GPU stages. Unlike face_anon this one IS fine on a V100:
# it is PyTorch end to end, and PyTorch still ships sm_70 kernels. face_anon
# dies on the same node because its onnxruntime-gpu 1.28 cuDNN does not --
# same hardware, opposite outcome, which is why the two have separate scripts
# with different --gpus lines rather than one shared job.
#
# ONE GPU, on GPU-shared, not a full node. Whisper decodes a single audio
# stream sequentially; there is nothing here to spread over eight GPUs the way
# face_anon spreads eighty thousand independent frames. A full node would bill
# 8 GPU-hours per hour to leave seven idle, out of a grant of ~495.
#
# Submit:
#     LECTURE_DIR=/ocean/projects/cis260220p/$USER/lectures/17635-lecture13 \
#         ./scripts/psc.sh sbatch scripts/psc_transcription.sbatch
#
#SBATCH --job-name=transcribe
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=v100-32:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/transcribe-%j.out
#SBATCH --error=logs/transcribe-%j.out

set -euo pipefail

# Same reasoning as psc_face_anon.sbatch: derive the repo from where sbatch was
# invoked rather than hardcoding a path that can drift out of sync with where
# the repo actually is.
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
LECTURE_DIR="${LECTURE_DIR:-data/15210-lecture12}"
BATCH_SIZE="${BATCH_SIZE:-16}"

cd "$REPO"
mkdir -p logs

module load anaconda3 2>/dev/null || true
source activate scs-learn                # the AUDIO env: whisperx, torch,
                                         # pyannote, anthropic. NOT scs-video,
                                         # whose onnxruntime-gpu conflicts with
                                         # the plain onnxruntime insightface
                                         # pulls in.

# transcription.py shells out to ffmpeg to make the wav. Prefer the conda
# build for the same reason face_anon does: PSC's ffmpeg module is a
# singularity wrapper whose /bil bind mount fails on some nodes, and a stage
# that pipes into ffmpeg then dies with BrokenPipeError long after the
# expensive work succeeded.
if [ -x "${CONDA_PREFIX:-}/bin/ffmpeg" ]; then
    export PATH="$CONDA_PREFIX/bin:$PATH"
    echo "ffmpeg  : $CONDA_PREFIX/bin/ffmpeg (native conda build)"
else
    module load ffmpeg 2>/dev/null || true
    echo "ffmpeg  : $(command -v ffmpeg) (PSC module -- singularity wrapper)"
fi
command -v ffmpeg >/dev/null || { echo "ERROR: no ffmpeg on PATH" >&2; exit 1; }

# Two secrets, and NEITHER can come from the repo: .env is gitignored and
# excluded from psc.sh sync on purpose, so nothing ever carries a key onto a
# shared filesystem. They must already be in the environment, or in a .env
# placed on PSC by hand, with permissions to match.
if [ -f "$REPO/.env" ]; then
    set -a; . "$REPO/.env"; set +a
    echo "secrets : sourced from \$REPO/.env"
fi
: "${HF_TOKEN:?HF_TOKEN is not set.
pyannote/speaker-diarization-3.1 is gated: accept its conditions AND those of
pyannote/speaker-diarization-community-1, which 3.1 pulls PLDA files from, or
the download 403s. Then put HF_TOKEN in $REPO/.env on PSC (chmod 600).}"
: "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY is not set -- the question
classification calls the API. Put it in $REPO/.env on PSC (chmod 600).}"

# /jet/home is 25 GB and typically ~90% full, and the whisper large-v2 plus
# pyannote weights are several GB. Left at its default the cache lands in
# ~/.cache and the job dies mid-download with 'Disk quota exceeded'.
export HF_HOME="${HF_HOME:-/ocean/projects/cis260220p/$USER/hf}"
mkdir -p "$HF_HOME"

echo "=============================================================="
echo "node    : $(hostname)"
echo "lecture : $LECTURE_DIR"
echo "HF_HOME : $HF_HOME"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo "=============================================================="

# Fail loudly rather than transcribing on CPU for two hours and timing out.
# transcription.py falls back to CPU silently when cuda is unavailable, which
# on an 82-minute lecture is the difference between minutes and not finishing.
python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("ERROR: torch reports no CUDA on this node. Do not run whisper "
             "large-v2 on CPU here -- it will not finish inside the walltime.")
print("CUDA available:", torch.cuda.get_device_name(0))
PY

srun python -m src.audio.transcription \
  --lecture-dir "$LECTURE_DIR" \
  --device cuda \
  --batch-size "$BATCH_SIZE"

echo "done: $LECTURE_DIR/transcript_classified.json"
