# scatac-ldm-project

**Modeling the Structure of DNA: Latent Distance Models for Sparse Single-Cell Data**  
*DTU (02466) Fagprojekt 1*

Applies a [Latent Distance Model (Hoff et al., 2002)](https://doi.org/10.1198/016214502388618906)
to single-cell ATAC-seq data, treating the cell-by-peak chromatin accessibility matrix
as a bipartite graph and learning a joint low-dimensional embedding where geometric
distance predicts accessibility probability.

---

## Repository layout

```
scatac-ldm-project/
├── src/
│   ├── ldm.py            # LDM nn.Module
│   ├── prepare_data.py   # Data loading, binarisation, train/val split
│   └── train.py          # Training loop, evaluation, CLI entry point
├── scripts/
│   ├── smoke_test.py     # End-to-end test on synthetic data (no real data needed)
│   └── run_ldm.sh        # LSF job script for DTU Sophia HPC
├── data/                 # (gitignored) — place .h5ad files here
├── results/              # (gitignored) — model checkpoints, embeddings, history
├── logs/                 # (gitignored) — HPC job logs
├── pyproject.toml
└── README.md
```

---

## Local setup

```bash
git clone https://github.com/YOUR_USERNAME/scatac-ldm-project.git
cd scatac-ldm-project
uv sync
```

### Smoke test (no data required — runs in ~15 seconds)

```bash
uv run python scripts/smoke_test.py
```

### Train on real data locally

```bash
uv run python src/train.py --data data/hematopoiesis_GSE129785_FACS_sorted.h5ad
```

---

## DTU Sophia HPC

### One-time setup (on the login node)

```bash
ssh sXXXXXX@login.hpc.dtu.dk

git clone https://github.com/YOUR_USERNAME/scatac-ldm-project.git ~/scatac-ldm-project
cd ~/scatac-ldm-project

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Check available Python/CUDA modules, then load them
module avail python
module load python3/3.11.3
module load cuda/12.1

uv sync
```

### Upload data (from your local machine)

```bash
scp data/hematopoiesis_GSE129785_FACS_sorted.h5ad \
    sXXXXXX@login.hpc.dtu.dk:~/data/
```

### Submit

```bash
mkdir -p logs
bsub < scripts/run_ldm.sh
```

### Monitor

```bash
bjobs                        # list jobs
bpeek <JOBID>                # live stdout
cat logs/ldm_<JOBID>.out     # full log after completion
```

---

## Outputs

After training, `results/ldm_run/` contains:

| File              | Description                                           |
|-------------------|-------------------------------------------------------|
| `best_model.pt`   | State dict at epoch with best validation AUC-PR       |
| `final_model.pt`  | State dict at final epoch                             |
| `z_cells.npy`     | Cell embeddings, shape `(n_cells, latent_dim)`        |
| `z_peaks.npy`     | Peak embeddings, shape `(n_peaks, latent_dim)`        |
| `history.json`    | Per-epoch train/val metrics (BCE, AUC-ROC, AUC-PR)   |

---

## Key hyperparameters

| Flag             | Default | Description                               |
|------------------|---------|-------------------------------------------|
| `--latent_dim`   | `16`    | Embedding dimensionality                  |
| `--epochs`       | `25`    | Training epochs                           |
| `--batch_size`   | `32768` | Edges per mini-batch                      |
| `--neg_ratio`    | `70`    | Negative edges sampled per positive       |
| `--lr`           | `1e-2`  | Adam learning rate                        |
| `--weight_decay` | `0.0`   | L2 regularisation (embeddings only)       |
| `--eval_every`   | `10`    | Validate every N epochs                   |
