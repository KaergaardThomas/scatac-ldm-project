uv run python src/train_ldm.py --device 0 1 --epochs 10 --batch_size 8388608 --latent_dim 8
uv run python src/plot_ldm.py --peakvi_dir results/ldm_model --out_dir results/ldm_evaluation
