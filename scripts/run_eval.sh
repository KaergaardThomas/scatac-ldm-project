#!/bin/bash
# ============================================================
#  DTU HPC — LSF job script
#  Runs evaluation (RQ1 + RQ2) after run_ldm.sh and run_null.sh have finished.
#  No GPU is needed — UMAP/K-means/LSA run on CPU — so this uses a CPU queue.
#
#  Submit from repo root:
#      bsub < scripts/run_eval.sh
# ============================================================

#BSUB -J eval_hematopoiesis
#BSUB -o logs/eval_%J.out
#BSUB -e logs/eval_%J.err
#BSUB -q hpc            # CPU queue — replace with your cluster's batch queue if different
#BSUB -n 4
#BSUB -R "rusage[mem=32GB]"
#BSUB -W 02:00

set -e

echo "============================="
echo "Job ID : $LSB_JOBID"
echo "Node   : $HOSTNAME"
echo "Start  : $(date)"
echo "============================="

module load python3/3.12.11

if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

REPO="$HOME/scatac-ldm-project"
cd "$REPO"
uv sync

mkdir -p logs results/evaluation

# The .h5ad is read once into memory, so read it straight from $HOME.
DATA="$HOME/data/hematopoiesis_GSE129785_FACS_sorted.h5ad"

# --min_cells_pct 0.001 must match the training runs so the LSA baseline uses
#   the same peak set as the LDM.
# --k is omitted on purpose: it defaults to the number of FACS cell types found
#   in the data. Add e.g. "--k 5" only to force a specific cluster count.
uv run python src/evaluate.py \
    --data          "$DATA" \
    --ldm_dir       results/ldm_run \
    --null_dir      results/null_run \
    --out_dir       results/evaluation \
    --seed          42 \
    --min_cells_pct 0.001

echo "============================="
echo "Done : $(date)"
echo "============================="
