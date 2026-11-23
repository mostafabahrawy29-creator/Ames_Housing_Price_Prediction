# Ames Housing Price Prediction

> **End-to-end Machine Learning regression system** on the Ames, Iowa housing dataset.  
> Models: Ridge · KNN · SVR · XGBoost · LightGBM · CatBoost — optimized with Optuna Bayesian search.

---

## Project Structure

```
Ames_Housing_Price_Prediction/
├── data/
│   └── train.csv                          # Ames Housing dataset (1,460 records, 81 features)
├── notebooks/
│   └── Ames_Housing_End_to_End_ML.ipynb   # Main interactive notebook
├── src/
│   ├── __init__.py
│   ├── data_loader.py                     # Dataset loading & structural-NA handling
│   ├── feature_engineering.py             # Domain composite features
│   ├── preprocessing.py                   # Scikit-Learn ColumnTransformer pipelines
│   ├── feature_selection.py               # Mutual Information & Variance Threshold
│   ├── evaluation.py                      # Metric computation & benchmark table
│   ├── visualizations.py                  # Plotly interactive chart generators
│   ├── notebook_builder.py                # Automated Jupyter Notebook generator
│   └── run_pipeline.py                    # Full end-to-end execution script
├── outputs/
│   ├── figures/                           # Interactive Plotly HTML charts
│   │   ├── price_distribution.html
│   │   ├── living_area_vs_price.html
│   │   ├── feature_selection_mi.html
│   │   ├── model_comparison_r2.html
│   │   ├── model_comparison_errors.html
│   │   ├── actual_vs_predicted.html
│   │   ├── permutation_importance.html
│   │   └── optuna_convergence.html
│   └── tables/
│       ├── model_evaluation_summary.csv
│       ├── feature_selection_summary.csv
│       └── best_hyperparameters.json
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Workflow Overview

```
Raw CSV → Structural-NA Fix → Outlier Removal → Feature Engineering
       → Train/Val/Test Split (70/15/15)
       → Leak-Free ColumnTransformer Pipelines
         ├── Skewed Numerical  : Median Impute → log1p → RobustScaler
         ├── Normal Numerical  : Median Impute → StandardScaler
         ├── Nominal Categorical: MostFreq Impute → OneHotEncoder (drop first)
         └── Ordinal Categorical: MostFreq Impute → OrdinalEncoder → StandardScaler
       → Feature Selection (Mutual Information + VarianceThreshold)
       → Optuna Bayesian Hyperparameter Optimization (5-fold CV, training set only)
       → 6-Model Benchmark (Train / Val / Test: R², RMSE, MAE in $)
       → Winning Model Selection + Permutation Importance
```

---

## Models Compared

| Model | Tuner | Notes |
|---|---|---|
| **Ridge Regression** | **Optuna** | **L2-regularized linear model, lowest Test Log-RMSE & Dollar RMSE — Winner** |
| CatBoost | Optuna | Ordered boosting with symmetric trees, top tree-based model |
| XGBoost | Optuna | Second-order regularized gradient boosting |
| SVR (RBF) | Optuna | Robust epsilon-insensitive loss, lowest Test Dollar MAE |
| LightGBM | Optuna | Leaf-wise gradient boosting, fast training |
| KNN Regressor | Optuna | Distance-weighted nearest neighbors baseline |

---

## Key Results

| Model | Train R² | Val R² | Test R² | Test RMSE (Log) | Test MAE ($) | Test RMSE ($) | Overfit Gap (R²) |
|---|---|---|---|---|---|---|---|
| **Ridge Regression** | **0.9375** | 0.9147 | **0.9239** | **0.1138** | $13,076.52 | **$18,653.21** | **0.0228** |
| CatBoost | 0.9937 | 0.9076 | 0.9124 | 0.1221 | $13,735.65 | $19,112.83 | 0.0861 |
| XGBoost | 0.9794 | 0.9005 | 0.9098 | 0.1239 | $13,726.27 | $19,808.74 | 0.0789 |
| SVR | 0.9632 | **0.9168** | 0.9079 | 0.1252 | **$12,641.71** | $18,715.69 | 0.0464 |
| LightGBM | 0.9848 | 0.8893 | 0.8971 | 0.1323 | $14,512.48 | $22,079.21 | 0.0955 |
| KNN Regressor | 1.0000 | 0.8400 | 0.8156 | 0.1771 | $20,873.41 | $30,109.97 | 0.1600 |

> Metrics are evaluated on the **log-transformed target** (equivalent to RMSLE). Dollar values are back-transformed via `expm1`.

---

## Why Ridge Regression Won

1. **Primary Metric Superiority**: Achieved the lowest Test Log-RMSE (`0.1138`), lowest Dollar RMSE (`$18,653.21`), and highest Test R² (`0.9239`), directly minimizing relative error across mid-range and high-value homes.
2. **Linearity After Transformation**: Log-transforming skewed numeric features and `SalePrice`, combined with domain-engineered interaction terms (`QualArea`, `TotalSF`, `OverallGrade`, `TotalBath`), linearized feature-target relationships so L2 regularization captures true signal without memorizing noise.
3. **Minimal Overfitting Gap**: Maintained the tightest generalization gap (`0.0228` R² gap / `0.0149` Log-RMSE gap), avoiding the wider variance (`0.0789–0.0955` R² overfit gap) observed in deeper tree ensembles.
4. **Interpretability & Low Latency**: Exposes transparent feature weights for stakeholder explainability and provides near-instantaneous inference speed compared to kernel or boosted models.

---

## Privacy & Security

> [!CAUTION]
> This repository does **not** contain any personal credentials, API keys, tokens, or private configuration files.  
> The `.gitignore` excludes `.env`, `secrets.json`, `venv/`, `catboost_info/`, and all IDE-specific metadata.

The dataset (`data/train.csv`) is derived from the publicly available **Ames Housing Dataset** (Dean De Cock, 2011) obtained via `sklearn.datasets.fetch_openml(name='house_prices')`. It contains no personally identifiable information (PII).

---

## Quick Start

```bash
# 1. Clone the repository
git clone [https://github.com/mostafabahrawy29-creator/Ames_Housing_Price_Prediction.git](https://github.com/mostafabahrawy29-creator/Ames_Housing_Price_Prediction.git)
cd Ames_Housing_Price_Prediction

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate    # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full end-to-end pipeline
python src/run_pipeline.py

# 5. Or open the interactive notebook
jupyter notebook notebooks/Ames_Housing_End_to_End_ML.ipynb
```

---

## Requirements

```
numpy, pandas, scipy, scikit-learn
matplotlib, seaborn, plotly
xgboost, lightgbm, catboost
optuna
ipykernel, nbformat
```

See [`requirements.txt`](requirements.txt) for pinned versions.

---

## Author

**Mostafa Tarek Mohamed Hassan**  
GitHub: [@mostafabahrawy29-creator](https://github.com/mostafabahrawy29-creator)

---

## License

This project is released for academic and educational purposes.  
Dataset source: [OpenML House Prices](https://www.openml.org/search?type=data&id=43926)