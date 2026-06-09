#!/bin/bash
# ============================================================
#  DTU Sophia HPC — LSF job script
#  Trains the Latent Distance Model on the hematopoiesis dataset
#
#  Submit from repo root on the login node:
#      bsub < scripts/run_ldm.sh
#
#  Monitor:
#      bjobs              — list your jobs
#      bpeek <JOBID>      — live stdout
#      bkill <JOBID>      — cancel a job
# ============================================================

#BSUB -J ldm_hematopoiesis
#BSUB -o logs/ldm_%J.out
#BSUB -e logs/ldm_%J.err
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 08:00

# Uncomment to get email when job ends:
##BSUB -u s245829@dtu.dk
##BSUB -N

set -e

echo "============================="
echo "Job ID : $LSB_JOBID"
echo "Node   : $HOSTNAME"
echo "Start  : $(date)"
echo "============================="

# ---- Modules ----------------------------------------------------------------
# Check available versions with: module avail python cuda
module load python3/3.11.3
module load cuda/12.1

# ---- uv ---------------------------------------------------------------------
# Install uv if not already present (only needed once)
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

export PATH="$HOME/.local/bin:$PATH"

# Move to repo root and sync the venv from pyproject.toml
REPO="$HOME/scatac-ldm-project"
cd "$REPO"
uv sync

# ---- Directories ------------------------------------------------------------
mkdir -p logs results/ldm_run

# ---- Run --------------------------------------------------------------------
uv run python src/train.py \
    --data         "$HOME/data/hematopoiesis_GSE129785_FACS_sorted.h5ad" \
    --latent_dim   8 \
    --epochs       200 \
    --batch_size   4096 \
    --neg_ratio    10 \
    --lr           1e-3 \
    --weight_decay 1e-4 \
    --seed         42 \
    --out_dir      results/ldm_run \
    --val_frac     0.10 \
    --eval_every   10

echo "============================="
echo "Done : $(date)"
echo "============================="
