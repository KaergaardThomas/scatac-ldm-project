# So i don't forget
uv run python src/train_peakvi.py --data data/hematopoiesis_GSE129785_FACS_sorted.h5ad --out_data data/hematopoiesis_with_peakvi.h5ad --model_dir results/peakvi_model --devices 1 --epochs 25
uv run python src/plot_peakvi.py --peakvi_dir results/peakvi_model --out_dir results/peakvi_evaluation
