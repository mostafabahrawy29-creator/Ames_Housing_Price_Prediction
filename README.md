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
│   ├── data_loader.py                     # Dataset loading & structural-NA handling
│   ├── feature_engineering.py             # Domain composite features
│   ├── preprocessing.py                   # Scikit-Learn ColumnTransformer pipelines
│   ├── feature_selection.py               # Mutual Information & Variance Threshold
│   ├── evaluation.py                      # Metric computation & benchmark table
│   ├── visualizations.py                  # Plotly interactive chart generators
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
| **Ridge Regression** | **Optuna** | **L2-regularized linear model, lowest Test Log-RMSE — Winner** |
| SVR (RBF) | Optuna | Robust epsilon-insensitive loss, lowest Dollar RMSE |
| CatBoost | Optuna | Ordered boosting with symmetric trees |
| LightGBM | Optuna | Leaf-wise gradient boosting, fast training |
| XGBoost | Optuna | Second-order regularized gradient boosting |
| KNN Regressor | Optuna | Distance-weighted nearest neighbors baseline |

---

## Key Results

| Model | Test R² | Test RMSE (Log) | Test RMSE ($) | Test MAE ($) |
|---|---|---|---|---|
| **Ridge Regression** | **0.9241** | **0.1136** | **$18,627.60** | — |
| SVR | — | — | $18,616.18 | $12,605.34 |
| CatBoost | — | — | $18,885.55 | — |
| LightGBM | — | — | — | — |
| XGBoost | — | — | — | — |
| KNN | — | — | — | — |

> Metrics are evaluated on the **log-transformed target** (equivalent to RMSLE). Dollar values are back-transformed via `expm1`.

---

## Why Ridge Regression Won

1. **Primary Metric Superiority**: Achieved the lowest Test Log-RMSE (`0.1136`) and highest R² (`0.9241`), directly minimizing relative error across mid-range and high-value homes.
2. **Linearity After Transformation**: Log-transforming skewed numeric features and `SalePrice`, combined with domain-engineered interaction terms, linearized feature-target relationships so L2 regularization captures true signal without memorizing noise.
3. **Minimal Overfitting Gap**: Maintained tight generalization between validation and test splits, avoiding the wider train-to-test variance (`0.0609–0.0978` Log-RMSE gap) observed in deeper tree ensembles.
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
git clone https://github.com/mostafabahrawy29-creator/Ames_Housing_Price_Prediction.git
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