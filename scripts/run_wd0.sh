#!/bin/bash
#BSUB -J ldm_wd0
#BSUB -o logs/wd0_%J.out
#BSUB -e logs/wd0_%J.err
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

mkdir -p logs results/ldm_wd0

SCRATCH="${TMPDIR:-/tmp}/${USER}_${LSB_JOBID}"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT
cp "$HOME/data/hematopoiesis_GSE129785_FACS_sorted.h5ad" "$SCRATCH/"
DATA="$SCRATCH/hematopoiesis_GSE129785_FACS_sorted.h5ad"
echo "Done copying. Starting training..."

# Original config but with ONLY weight_decay changed to 0.
# Tests whether weight_decay alone was suppressing the geometry.
uv run python src/train.py \
    --data          "$DATA" \
    --out_dir       results/ldm_wd0 \
    --init_std      0.1 \
    --weight_decay  0.0 \
    --lr            1e-3 \
    --epochs        30 \
    --batch_size    16384 \
    --neg_ratio     10 \
    --val_frac      0.10 \
    --seed          42 \
    --min_cells_pct 0.001

echo "============================="
echo "Done : $(date)"
echo "============================="
