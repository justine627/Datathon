# Beijing Multi-Site Air Quality — Next-Hour PM2.5 Prediction

This notebook demonstrates the culmination of our efforts to produce a leak-proof model in order to predict `PM2_5_next_hour` for the "Inter-Uni Datathon Stream 2 — Beijing Multi-Site Air Quality" Kaggle competition,
using LightGBM and XGBoost with a chronological validation split and a weighted blend.

## Required Files

Input data (from the Kaggle competition, expected under `/kaggle/input/competitions/inter-uni-datathon-stream-2-beijing-multi-site-air-quality/`):
- `train.csv` — historical hourly readings per station, includes the target `PM2_5_next_hour`
- `test(1).csv` — hourly readings requiring predictions (no target column)
- `sample_submission.csv` — submission format reference (listed but not read in the notebook)

Files produced by the notebook (written to the working directory):
- `/kaggle/working/train_cleaned.csv` — cleaned training data (exploratory pass, cell 3)
- `/kaggle/working/test_cleaned.csv` — cleaned test data (exploratory pass, cell 3)
- `train_cleaned_with_features.csv` — final engineered training features (main pipeline)
- `submission.csv` — final predictions in `id, PM2_5_next_hour` format

## Execution Order

The notebook has two loosely related parts that must each run top to bottom:

1. **Cells 1–3 — EDA and a simpler feature pass**
   - Cell 1: Load `train.csv`/`test(1).csv`, check duplicates/missingness, correlations, target distribution.
   - Cell 2: Impossible-value checks, cyclical hour/month encodings, categorical setup, log-transform check on the target.
   - Cell 3: Apply the same preprocessing to `test_df`, align `station`/`wd` categories, save `train_cleaned.csv` / `test_cleaned.csv`.
   - This section is exploratory — its outputs are not consumed by the modeling pipeline in cell 4.

2. **Cell 4 — Full modeling pipeline (self-contained; re-loads raw data)**
   1. Load raw `train.csv` / `test(1).csv`
   2. Leakage-safe missing-value handling (forward-fill within station, then medians learned only from permitted reference periods)
   3. Feature engineering (wind vectors, cyclical time features, pollutant ratios, lags, rolling stats, expanding z-scores/hour-of-day averages) — all strictly causal (past-looking only)
   4. Build validation features (train portion + validation portion only)
   5. Build final train/test features (full train + test)
   6. Time-based train/validation split (80th percentile timestamp cutoff)
   7. Hyperparameter search over a small LightGBM grid using early stopping
   8. Fit final validation-stage LightGBM and XGBoost models, report RMSE/MAE
   9. Search blend weight between LightGBM and XGBoost validation predictions
   10. Retrain LightGBM (and XGBoost, if the blend uses it) on the full training set, predict on test
   11. Export `submission.csv`

Run all cells sequentially; do not skip cells 1–3, as they define `df`/`test_df` used for the exploratory analysis, even though cell 4 reloads data independently for modeling.

## Key Dependencies / Packages

- `numpy`
- `pandas`
- `matplotlib` (cells 1 EDA plots)
- `lightgbm` (`lgb.LGBMRegressor`, `lgb.early_stopping`)
- `xgboost` (`XGBRegressor`, with `enable_categorical=True`, `tree_method="hist"`)
- `scikit-learn` (`mean_squared_error`, `mean_absolute_error`)
- `kagglehub` (imported but unused beyond the commented example)
- Standard library: `os`, `time`

## Important Random Seeds / Settings

- `random_state=42` is set on every LightGBM (`LGBMRegressor`) and XGBoost (`XGBRegressor`) model instance, for both the hyperparameter search and final fits.
- `DEBUG = False` flag in cell 4: if set to `True`, restricts training/test data to only the first 2 stations for fast iteration.
- Chronological validation split: `cutoff = train_raw[TIMESTAMP].quantile(0.8)` — the last 20% of timestamps (by date, not row order) form the validation set.
- Missing-value forward-fill limit: `ffill(limit=6)` (up to 6 hours) before falling back to reference-period medians/modes.
- LightGBM hyperparameter grid searched: `num_leaves ∈ {31, 63, 127}` × `learning_rate ∈ {0.05, 0.1}` (4 combinations), `n_estimators=2000` with `early_stopping(stopping_rounds=50)`.
- XGBoost fixed settings: `n_estimators=800`, `learning_rate=0.03`, `max_depth=6`, `early_stopping_rounds=50`.
- Blend weight search: `w` swept from 0.0 to 1.0 in steps of 0.1 (LightGBM weight), choosing the `w` minimizing validation RMSE.
- Final predictions are clipped to be non-negative (`np.clip(..., 0, None)`), and any unmatched submission rows are filled with the training target mean.

## Notebook with final prediction
The notebook with the final prediction is Final_Version.ipynb in this repository.
