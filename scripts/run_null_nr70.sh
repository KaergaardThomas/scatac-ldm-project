#!/bin/bash
#BSUB -J null_nr70
#BSUB -o logs/null_nr70_%J.out
#BSUB -e logs/null_nr70_%J.err
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

mkdir -p logs results/null_nr70

SCRATCH="${TMPDIR:-/tmp}/${USER}_${LSB_JOBID}"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT
cp "$HOME/data/hematopoiesis_GSE129785_FACS_sorted.h5ad" "$SCRATCH/"
DATA="$SCRATCH/hematopoiesis_GSE129785_FACS_sorted.h5ad"
echo "Done copying. Starting training..."

uv run python src/train_null.py \
    --data          "$DATA" \
    --out_dir       results/null_nr70 \
    --neg_ratio     70 \
    --eval_max_pos  200000 \
    --epochs        25 \
    --batch_size    32768 \
    --lr            1e-3 \
    --seed          42 \
    --val_frac      0.10 \
    --min_cells_pct 0.001

echo "============================="
echo "Done : $(date)"
echo "============================="
