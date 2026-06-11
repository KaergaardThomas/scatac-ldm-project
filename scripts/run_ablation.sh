#!/bin/bash
# ============================================================
#  DTU HPC — LSF job script
#  Ablation: LDM without intercepts (geometry forced to do the work)
#
#  Submit from repo root:
#      bsub < scripts/run_ablation.sh
# ============================================================

#BSUB -J ldm_ablation
#BSUB -o logs/ablation_%J.out
#BSUB -e logs/ablation_%J.err
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

mkdir -p logs results/ldm_noint

# ---- Node-local scratch -----------------------------------------------------
SCRATCH="${TMPDIR:-/tmp}/${USER}_${LSB_JOBID}"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT
echo "Scratch: $SCRATCH"
cp "$HOME/data/hematopoiesis_GSE129785_FACS_sorted.h5ad" "$SCRATCH/"
DATA="$SCRATCH/hematopoiesis_GSE129785_FACS_sorted.h5ad"
echo "Done copying. Starting training..."

# ---- Run ablation -----------------------------------------------------------
# No intercepts: geometry is forced to explain accessibility.
# Large init_std (1.0) and no weight decay give the embeddings room to move.
# 30 epochs is sufficient — Opus confirmed it converges in a few epochs.
uv run python src/train.py \
    --data          "$DATA" \
    --out_dir       results/ldm_noint \
    --no_intercepts \
    --init_std      1.0 \
    --weight_decay  0.0 \
    --lr            1e-2 \
    --epochs        30 \
    --batch_size    16384 \
    --neg_ratio     10 \
    --val_frac      0.10 \
    --seed          42 \
    --min_cells_pct 0.001

echo "============================="
echo "Done : $(date)"
echo "============================="
