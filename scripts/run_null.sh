#!/bin/bash
# ============================================================
#  DTU HPC — LSF job script
#  Trains the NULL model (intercepts only) for the nested comparison.
#
#  IMPORTANT: the flags below (--seed, --val_frac, --neg_ratio,
#  --min_cells_pct) MUST match run_ldm.sh. The train/val split and the
#  evaluation negative set are seeded from these, so matching them is what
#  makes the held-out evaluation identical to the full LDM — the whole point
#  of the nested comparison.
#
#  Submit from repo root:
#      bsub < scripts/run_null.sh
# ============================================================

#BSUB -J null_hematopoiesis
#BSUB -o logs/null_%J.out
#BSUB -e logs/null_%J.err
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

mkdir -p logs results/null_run

# ---- Node-local scratch (unique per job, auto-cleaned) ----------------------
SCRATCH="${TMPDIR:-/tmp}/${USER}_${LSB_JOBID}"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT
echo "Scratch: $SCRATCH"
cp "$HOME/data/hematopoiesis_GSE129785_FACS_sorted.h5ad" "$SCRATCH/"
DATA="$SCRATCH/hematopoiesis_GSE129785_FACS_sorted.h5ad"

# ---- Run (flags MUST match run_ldm.sh for an identical eval set) ------------
uv run python src/train_null.py \
    --data          "$DATA" \
    --epochs        200 \
    --batch_size    16384 \
    --neg_ratio     10 \
    --lr            1e-3 \
    --seed          42 \
    --out_dir       results/null_run \
    --val_frac      0.10 \
    --eval_every    10 \
    --min_cells_pct 0.001

echo "============================="
echo "Done : $(date)"
echo "============================="
