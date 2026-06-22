uv run python src/train_ldm.py --device 0  --epochs 25 --batch_size 1024000 --latent_dim 16 --seeds 42
uv run python src/train_ldm.py --device 0  --epochs 25 --batch_size 1024000 --latent_dim 16 --seeds 237
uv run python src/train_ldm.py --device 0  --epochs 25 --batch_size 1024000 --latent_dim 16 --seeds 284
uv run python src/train_ldm.py --device 0  --epochs 25 --batch_size 1024000 --latent_dim 16 --seeds 156
uv run python src/train_ldm.py --device 0  --epochs 25 --batch_size 1024000 --latent_dim 16 --seeds 708
uv run python src/train_ldm.py --device 0  --epochs 25 --batch_size 1024000 --latent_dim 16 --seeds 844
uv run python src/train_ldm.py --device 0  --epochs 25 --batch_size 1024000 --latent_dim 16 --seeds 444
uv run python src/train_ldm.py --device 0  --epochs 25 --batch_size 1024000 --latent_dim 16 --seeds 397
uv run python src/train_ldm.py --device 0  --epochs 25 --batch_size 1024000 --latent_dim 16 --seeds 431
uv run python src/train_ldm.py --device 0  --epochs 25 --batch_size 1024000 --latent_dim 16 --seeds 437

uv run python src/plot_ldm.py \
     --base_model_dir results/ldm_model \
     --base_out_data data/hematopoiesis_with_ldm \
     --latent_dim 2 4 8 16 32 64 \
     --threads 4

uv run python src/print_results.py \
     --base_model_dir results/ldm_model \
     --base_out_data data/hematopoiesis_with_ldm \
     --latent_dim 2 4 8 16 32 64

uv run python src/evaluate.py \
    --data data/hematopoiesis_GSE129785_FACS_sorted.h5ad \
    --ldm_dir results/ldm_model/seed_42_dim_16 \
    --out_dir results/evaluation


42 237 284 156 708 844 444 397 431 437
