uv run python src/train_ldm.py --data data/hematopoiesis_GSE129785_FACS_sorted.h5ad --out_data data/hematopoiesis_with_ldm.h5ad --model_dir results/ldm_model --devices 1 --epochs 10 --batch_size 8388608
uv run python src/plot_ldm.py --peakvi_dir results/ldm_model --out_dir results/ldm_evaluation
