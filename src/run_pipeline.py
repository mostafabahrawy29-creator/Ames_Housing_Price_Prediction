"""
Ames Housing Price Prediction - Full End-to-End Pipeline
Steps: Load -> Clean -> Engineer -> Preprocess -> Feature Selection
       -> Optuna Tuning -> Benchmark -> Plotly Charts
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna

from data_loader import load_dataset, handle_structural_missing, remove_documented_outliers
from feature_engineering import construct_domain_features
from preprocessing import build_preprocessor
from feature_selection import calculate_mutual_information, apply_variance_threshold
from evaluation import build_benchmark_table
from visualizations import (
    plot_price_distribution,
    plot_outlier_scatter,
    plot_mi_ranking,
    plot_model_comparison_r2,
    plot_model_errors,
    plot_actual_vs_predicted,
    plot_permutation_importance,
    plot_optuna_history,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
np.random.seed(SEED)
N_TRIALS = 15
N_CV_FOLDS = 5

DATA_PATH = os.path.join(PROJECT_DIR, "data", "train.csv")
OUT_FIGS = os.path.join(PROJECT_DIR, "outputs", "figures")
OUT_TABLES = os.path.join(PROJECT_DIR, "outputs", "tables")
os.makedirs(OUT_FIGS, exist_ok=True)
os.makedirs(OUT_TABLES, exist_ok=True)

# 1. Load & Clean
print("Step 1: Loading and cleaning dataset...")
df_raw = load_dataset(DATA_PATH)
df_clean = handle_structural_missing(df_raw)
df_clean = remove_documented_outliers(df_clean)
print(f"  Raw shape: {df_raw.shape} -> Cleaned shape: {df_clean.shape}")

# 2. Feature Engineering
print("Step 2: Constructing domain features...")
df_fe = construct_domain_features(df_clean)
y_log = np.log1p(df_fe["SalePrice"].values)
X_all = df_fe.drop(columns=["SalePrice"])
print(f"  Total features after engineering: {X_all.shape[1]}")

# 3. Train / Val / Test Split
print("Step 3: Splitting data (70% Train / 15% Val / 15% Test)...")
X_tv, X_te, y_tv, y_te = train_test_split(X_all, y_log, test_size=0.15, random_state=SEED)
X_tr, X_va, y_tr, y_va = train_test_split(X_tv, y_tv, test_size=(0.15 / 0.85), random_state=SEED)
print(f"  Train: {X_tr.shape[0]} | Val: {X_va.shape[0]} | Test: {X_te.shape[0]}")

# 4. Preprocessing
print("Step 4: Fitting leak-free ColumnTransformer...")
preprocessor, _, _, _, _ = build_preprocessor(X_tr, skew_threshold=0.75)
X_tr_p = preprocessor.fit_transform(X_tr)
X_va_p = preprocessor.transform(X_va)
X_te_p = preprocessor.transform(X_te)
feat_names = preprocessor.get_feature_names_out().tolist()
print(f"  Encoded feature count: {X_tr_p.shape[1]}")

# 5. Feature Selection
print("Step 5: Applying variance filter and Mutual Information ranking...")
X_tr_p, X_va_p, X_te_p, feat_names = apply_variance_threshold(
    X_tr_p, X_va_p, X_te_p, feat_names, threshold=0.005
)
print(f"  Retained {len(feat_names)} features after variance filtering.")

mi_df = calculate_mutual_information(X_tr_p, y_tr, feat_names, random_state=SEED)
mi_df.to_csv(os.path.join(OUT_TABLES, "feature_selection_summary.csv"), index=False)
print(f"  Top feature: {mi_df.iloc[0]['Feature']} (MI={mi_df.iloc[0]['MI_Score']:.4f})")

# 6. Optuna Bayesian Tuning
print(f"\nStep 6: Running Optuna optimization ({N_TRIALS} trials, {N_CV_FOLDS}-fold CV)...")
kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)


def cv_rmse(model, X, y):
    scores = []
    for ti, vi in kf.split(X):
        model.fit(X[ti], y[ti])
        scores.append(float(np.sqrt(mean_squared_error(y[vi], model.predict(X[vi])))))
    return float(np.mean(scores))


studies = {}


def tune(name, make_model_fn):
    def obj(trial):
        return cv_rmse(make_model_fn(trial), X_tr_p, y_tr)

    st = optuna.create_study(direction="minimize")
    st.optimize(obj, n_trials=N_TRIALS)
    studies[name] = st
    print(f"  {name:20s} | Best CV-RMSE = {st.best_value:.4f}")
    return st.best_params


p = tune("Ridge Regression", lambda t: Ridge(alpha=t.suggest_float("alpha", 0.1, 100.0, log=True), random_state=SEED))
best_ridge = Ridge(**p, random_state=SEED).fit(X_tr_p, y_tr)

p = tune(
    "KNN Regressor",
    lambda t: KNeighborsRegressor(
        n_neighbors=t.suggest_int("n_neighbors", 3, 21, step=2),
        weights=t.suggest_categorical("weights", ["uniform", "distance"]),
        p=t.suggest_int("p", 1, 2),
    ),
)
best_knn = KNeighborsRegressor(**p).fit(X_tr_p, y_tr)

p = tune(
    "SVR",
    lambda t: SVR(
        C=t.suggest_float("C", 0.5, 30.0, log=True),
        epsilon=t.suggest_float("epsilon", 0.01, 0.2),
        gamma=t.suggest_categorical("gamma", ["scale", "auto"]),
    ),
)
best_svr = SVR(**p).fit(X_tr_p, y_tr)


def make_xgb(t):
    return xgb.XGBRegressor(
        n_estimators=t.suggest_int("n_estimators", 150, 450, step=50),
        max_depth=t.suggest_int("max_depth", 3, 6),
        learning_rate=t.suggest_float("learning_rate", 0.02, 0.15, log=True),
        subsample=t.suggest_float("subsample", 0.7, 0.95),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.6, 0.95),
        reg_alpha=t.suggest_float("reg_alpha", 0.01, 5.0, log=True),
        reg_lambda=t.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        random_state=SEED,
        n_jobs=1,
        verbosity=0,
    )


p = tune("XGBoost", make_xgb)
best_xgb = xgb.XGBRegressor(**p, random_state=SEED, n_jobs=1, verbosity=0).fit(X_tr_p, y_tr)


def make_lgb(t):
    return lgb.LGBMRegressor(
        n_estimators=t.suggest_int("n_estimators", 150, 450, step=50),
        max_depth=t.suggest_int("max_depth", 3, 8),
        num_leaves=t.suggest_int("num_leaves", 15, 45),
        learning_rate=t.suggest_float("learning_rate", 0.02, 0.15, log=True),
        subsample=t.suggest_float("subsample", 0.7, 0.95),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.6, 0.95),
        reg_alpha=t.suggest_float("reg_alpha", 0.01, 5.0, log=True),
        reg_lambda=t.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        random_state=SEED,
        n_jobs=1,
        verbose=-1,
    )


p = tune("LightGBM", make_lgb)
best_lgb = lgb.LGBMRegressor(**p, random_state=SEED, n_jobs=1, verbose=-1).fit(X_tr_p, y_tr)


def make_cat(t):
    return cb.CatBoostRegressor(
        iterations=t.suggest_int("iterations", 200, 500, step=50),
        depth=t.suggest_int("depth", 4, 7),
        learning_rate=t.suggest_float("learning_rate", 0.02, 0.15, log=True),
        l2_leaf_reg=t.suggest_float("l2_leaf_reg", 2.0, 10.0),
        random_seed=SEED,
        verbose=0,
        thread_count=1,
        allow_writing_files=False,
    )


p = tune("CatBoost", make_cat)
best_cat = cb.CatBoostRegressor(
    **p, random_seed=SEED, verbose=0, thread_count=1, allow_writing_files=False
).fit(X_tr_p, y_tr)

with open(os.path.join(OUT_TABLES, "best_hyperparameters.json"), "w") as f:
    json.dump({k: studies[k].best_params for k in studies}, f, indent=4)

# 7. Multi-Split Benchmark
print("\nStep 7: Evaluating all models across Train / Val / Test...")
all_models = {
    "Ridge Regression": best_ridge,
    "KNN Regressor": best_knn,
    "SVR": best_svr,
    "XGBoost": best_xgb,
    "LightGBM": best_lgb,
    "CatBoost": best_cat,
}

benchmark_df = build_benchmark_table(all_models, X_tr_p, y_tr, X_va_p, y_va, X_te_p, y_te)
benchmark_df.to_csv(os.path.join(OUT_TABLES, "model_evaluation_summary.csv"), index=False)

print("\n" + "=" * 78)
print(benchmark_df[["Model", "Train R2", "Val R2", "Test R2", "Test RMSE (Log)", "Test MAE ($)"]].to_string(index=False))
print("=" * 78)

winner_name = benchmark_df.iloc[0]["Model"]
winner_mdl = all_models[winner_name]
print(f"\nWinning Model: {winner_name}")

# 8. Interactive Plotly Visualizations
print("\nStep 8: Generating interactive Plotly visualizations...")
plot_price_distribution(df_clean, OUT_FIGS)
plot_outlier_scatter(df_raw, OUT_FIGS)
plot_mi_ranking(mi_df, OUT_FIGS, top_n=25)
plot_model_comparison_r2(benchmark_df, OUT_FIGS)
plot_model_errors(benchmark_df, OUT_FIGS)
plot_actual_vs_predicted(y_te, winner_mdl.predict(X_te_p), winner_name, OUT_FIGS)
plot_optuna_history(studies, OUT_FIGS)

perm = permutation_importance(winner_mdl, X_te_p, y_te, n_repeats=10, random_state=SEED, n_jobs=1)
perm_df = pd.DataFrame(
    {"Feature": feat_names, "Importance": perm.importances_mean}
).sort_values("Importance", ascending=False).reset_index(drop=True)
plot_permutation_importance(perm_df, winner_name, OUT_FIGS)

print("\nPipeline completed successfully. All tables and figures saved.")