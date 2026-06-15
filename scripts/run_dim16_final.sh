#!/bin/bash
# ============================================================
#  DTU HPC — LSF job script
#  Retrain LDM dim=16 with step-level BCE logging (--log_every 100)
#  for smooth training curves in the report.
#
#  Identical config to the sweep dim=16 run (same seed, same flags),
#  so results are reproducible. Only addition: --log_every 100 saves
#  BCE to history.json every 100 steps (~125 recordings per epoch).
#
#  Submit from repo root:
#      bsub < scripts/run_dim16_final.sh
# ============================================================

#BSUB -J ldm_dim16_final
#BSUB -o logs/dim16_final_%J.out
#BSUB -e logs/dim16_final_%J.err
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=48GB]"
#BSUB -W 04:00

set -e

echo "============================="
echo "Job ID : $LSB_JOBID"
echo "Node   : $HOSTNAME"
echo "Start  : $(date)"
echo "============================="

module load python3/3.12.11
module load cuda/11.8

if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

REPO="$HOME/scatac-ldm-project"
cd "$REPO"
uv sync

mkdir -p logs results/ldm_dim16_final

SCRATCH="${TMPDIR:-/tmp}/${USER}_${LSB_JOBID}"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT
cp "$HOME/data/hematopoiesis_GSE129785_FACS_sorted.h5ad" "$SCRATCH/"
DATA="$SCRATCH/hematopoiesis_GSE129785_FACS_sorted.h5ad"
echo "Done copying. Starting training..."

uv run python src/train.py \
    --data          "$DATA" \
    --out_dir       results/ldm_dim16_final \
    --latent_dim    16 \
    --init_std      1.0 \
    --weight_decay  0.0 \
    --lr            1e-2 \
    --epochs        25 \
    --batch_size    32768 \
    --neg_ratio     70 \
    --eval_max_pos  200000 \
    --log_every     100 \
    --val_frac      0.10 \
    --seed          42 \
    --min_cells_pct 0.001

echo "============================="
echo "Done : $(date)"
echo "============================="
