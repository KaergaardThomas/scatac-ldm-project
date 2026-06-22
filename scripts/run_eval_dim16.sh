#!/bin/bash
#BSUB -J eval_dim16
#BSUB -o logs/eval_dim16_%J.out
#BSUB -e logs/eval_dim16_%J.err
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=32GB]"
#BSUB -W 02:00

set -e

module load python3/3.12.11

if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/scatac-ldm-project"
uv sync
mkdir -p logs results/evaluation_dim16

uv run python src/evaluate.py \
    --data          "$HOME/data/hematopoiesis_GSE129785_FACS_sorted.h5ad" \
    --ldm_dir       results/ldm_dim16 \
    --null_dir      results/null_nr70 \
    --out_dir       results/evaluation_dim16 \
    --min_cells_pct 0.001

echo "Done : $(date)"
