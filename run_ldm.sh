uv run python src/train_ldm.py --device 0 1 --epochs 10 --batch_size 8388608 --latent_dim 8
uv run python src/plot_ldm.py \
     --base_model_dir results/ldm_model \
     --base_out_data data/hematopoiesis_with_ldm \
     --latent_dim 2 4 8 16 32 64 \
     --threads 4
