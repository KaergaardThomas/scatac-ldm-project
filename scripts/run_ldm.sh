#!/bin/bash
# ============================================================
#  DTU HPC — LSF job script
#  Trains the Latent Distance Model on the hematopoiesis dataset.
#
#  The null model and evaluation are separate steps:
#      bsub < scripts/run_null.sh      (GPU, fast)
#      bsub < scripts/run_eval.sh      (CPU queue — no GPU needed)
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
#BSUB -R "rusage[mem=48GB]"
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
module load python3/3.12.11
module load cuda/11.8

# ---- uv ---------------------------------------------------------------------
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

REPO="$HOME/scatac-ldm-project"
cd "$REPO"
uv sync

# ---- Directories ------------------------------------------------------------
mkdir -p logs results/ldm_run

# ---- Verify GPU -------------------------------------------------------------
echo "PyTorch version and GPU:"
uv run python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# ---- Copy data to node-local scratch (unique per job, auto-cleaned) ---------
# $TMPDIR can be empty/unset on the compute node, so fall back to /tmp and use
# a per-job subdirectory to avoid collisions. The trap removes it on any exit
# (normal completion or a `set -e` abort), so no multi-GB files are left behind.
#
# Alternatively, skip the copy entirely — the .h5ad is read once into memory,
# so reading straight from $HOME over NFS is fine:
#     DATA="$HOME/data/hematopoiesis_GSE129785_FACS_sorted.h5ad"
SCRATCH="${TMPDIR:-/tmp}/${USER}_${LSB_JOBID}"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT
echo "Scratch: $SCRATCH"
echo "Copying data to node-local scratch..."
cp "$HOME/data/hematopoiesis_GSE129785_FACS_sorted.h5ad" "$SCRATCH/"
DATA="$SCRATCH/hematopoiesis_GSE129785_FACS_sorted.h5ad"
echo "Done copying. Starting training..."

# ---- Run --------------------------------------------------------------------
# --min_cells_pct 0.001 matches PeakVI's preprocessing of this exact dataset
#   (GSE129785: 571,400 -> 133,962 peaks). The previous 0.05 was 50x too
#   aggressive and dropped most cell-type-specific peaks.
# --batch_size 16384: negatives are now sampled per-batch on the GPU, so a
#   large batch is cheap and cuts the number of steps per epoch. Raise further
#   (e.g. 65536) for more speed, or lower if you see underfitting.
# Evaluation defaults (unset here): eval_max_pos=1,000,000 held-out positives,
#   observed positives filtered out of the evaluation negatives.
uv run python src/train.py \
    --data          "$DATA" \
    --latent_dim    8 \
    --epochs        200 \
    --batch_size    16384 \
    --neg_ratio     10 \
    --lr            1e-3 \
    --weight_decay  1e-4 \
    --seed          42 \
    --out_dir       results/ldm_run \
    --val_frac      0.10 \
    --eval_every    10 \
    --min_cells_pct 0.001

echo "============================="
echo "Done : $(date)"
echo "============================="
