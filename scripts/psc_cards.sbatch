#!/bin/bash
# Burn student-question cards into the screen recording, on Bridges-2.
#
# This runs on RM (CPU), *not* a GPU node, on purpose:
#
#   Bridges-2 GPU nodes are Tesla V100s, and NVIDIA's compute-class datacenter
#   parts (P100/V100/A100/H100) ship NVDEC but no NVENC encoder block. So
#   h264_nvenc cannot work there no matter how ffmpeg was built -- CUDA compute
#   availability says nothing about video-encode silicon. With the encode
#   necessarily on CPU, an RM node is strictly better: 2x AMD EPYC 7742 = 128
#   cores vs. 40 Xeon cores on a GPU node, and it charges Regular SUs (5000 in
#   grant cis260220p) instead of the scarce GPU SUs (~499) that transcription
#   and face anonymization actually need.
#
# Submit:  sbatch scripts/psc_cards.sbatch
# Watch:   squeue -u $USER ; tail -f logs/cards-<jobid>.out
#
#SBATCH --job-name=cards
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=01:00:00
#SBATCH --output=logs/cards-%j.out
#SBATCH --error=logs/cards-%j.out

set -euo pipefail

PROJECT="${PROJECT:-$PROJECT}"          # /ocean/projects/cis260220p/$USER
REPO="${REPO:-$PROJECT/scs-learn-video-pipeline}"
LECTURE_DIR="${LECTURE_DIR:-data/15210-lecture12}"

cd "$REPO"
mkdir -p logs

module load anaconda3 2>/dev/null || true
source activate scs-learn

# Bridges-2 ships ffmpeg as a module; fall back to whatever is on PATH (e.g. a
# conda-forge ffmpeg in the scs-learn env).
module load ffmpeg 2>/dev/null || true
command -v ffmpeg >/dev/null || {
  echo "ERROR: no ffmpeg on PATH. Try: conda install -c conda-forge ffmpeg" >&2
  exit 1
}

CORES="${SLURM_CPUS_ON_NODE:-${SLURM_NTASKS:-32}}"

echo "=============================================================="
echo "node        : $(hostname)"
echo "cores       : $CORES"
echo "ffmpeg      : $(command -v ffmpeg) ($(ffmpeg -version | head -1 | cut -d' ' -f3))"
echo "nvenc listed: $(ffmpeg -hide_banner -encoders 2>/dev/null | grep -c nvenc) encoder(s)"
echo "=============================================================="

# --encoder libx264 is explicit rather than auto: on a CPU node the probe would
# reach the same answer, but stating it keeps the log unambiguous and skips the
# smoke tests.
srun python -m src.audio.cards \
  --lecture-dir "$LECTURE_DIR" \
  --encoder libx264 \
  --jobs "$CORES"

echo "done: $LECTURE_DIR/screen_with_cards.mp4"
