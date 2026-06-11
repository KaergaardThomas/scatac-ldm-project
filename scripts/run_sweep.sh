#!/bin/bash
# ============================================================
#  DTU HPC — LSF job ARRAY: latent-dimension sweep for the LDM
#
#  One submit launches all six dimensions in parallel, each on its own GPU:
#      bsub < scripts/run_sweep.sh
#
#  Array index -> latent_dim via the DIMS list below.
#  To cap how many run at once (e.g. 5 concurrent), change the job name to
#  "ldm_sweep[1-6]%5".  To skip dim=8 (already have it as ldm_tuned), use
#  "[1-5]" and drop 8 from DIMS.
#
#  Monitor:  bjobs        live status of all array elements
#            bpeek <ID>   live stdout (includes the per-100-step BCE)
# ============================================================

#BSUB -J "ldm_sweep[1-6]"
#BSUB -o logs/sweep_%J_%I.out
#BSUB -e logs/sweep_%J_%I.err
#BSUB -q gpua100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=48GB]"
#BSUB -W 08:00

set -e

# ---- Map array index -> latent dimension ------------------------------------
DIMS=(2 4 8 16 32 64)
DIM=${DIMS[$((LSB_JOBINDEX - 1))]}

echo "============================="
echo "Array job : $LSB_JOBID[$LSB_JOBINDEX]"
echo "latent_dim: $DIM"
echo "Node      : $HOSTNAME"
echo "Start     : $(date)"
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

mkdir -p logs "results/ldm_dim${DIM}"

# ---- Node-local scratch (unique per array element) --------------------------
SCRATCH="${TMPDIR:-/tmp}/${USER}_${LSB_JOBID}_${LSB_JOBINDEX}"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT
cp "$HOME/data/hematopoiesis_GSE129785_FACS_sorted.h5ad" "$SCRATCH/"
DATA="$SCRATCH/hematopoiesis_GSE129785_FACS_sorted.h5ad"
echo "Done copying. Starting training (dim=$DIM)..."

# ---- Run --------------------------------------------------------------------
# neg_ratio 70 ≈ (1 - density)/density at 98.59% sparsity, so the sampled
#   class balance matches the real data. NOTE: this changes the AUC-PR scale
#   (prevalence ~1/71 ≈ 0.014, vs ~0.091 at neg_ratio 10), so these numbers are
#   comparable across dims but NOT to the earlier neg_ratio=10 runs.
# log_every 100 prints the running train BCE every 100 steps.
uv run python src/train.py \
    --data          "$DATA" \
    --out_dir       "results/ldm_dim${DIM}" \
    --latent_dim    ${DIM} \
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
echo "Done dim=$DIM : $(date)"
echo "============================="
