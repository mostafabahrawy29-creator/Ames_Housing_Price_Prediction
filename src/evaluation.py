"""
Comprehensive evaluation metrics for regression models.
Supports log-scale and original dollar-scale metric computation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict


def compute_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> Dict[str, float]:
    """
    Computes evaluation metrics in both log-scale and original dollar scale.

    Parameters
    ----------
    y_true_log : array-like, ground-truth log1p-transformed target values
    y_pred_log : array-like, model predicted log1p-transformed values

    Returns
    -------
    dict with R2, RMSE (log), MAE (log), MAE ($), RMSE ($)
    """
    r2 = r2_score(y_true_log, y_pred_log)
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    mae_log = mean_absolute_error(y_true_log, y_pred_log)

    y_true_dollar = np.expm1(y_true_log)
    y_pred_dollar = np.expm1(y_pred_log)

    mae_dollar = mean_absolute_error(y_true_dollar, y_pred_dollar)
    rmse_dollar = np.sqrt(mean_squared_error(y_true_dollar, y_pred_dollar))

    return {
        "R2": round(r2, 4),
        "RMSE (Log)": round(rmse_log, 4),
        "MAE (Log)": round(mae_log, 4),
        "MAE ($)": round(mae_dollar, 2),
        "RMSE ($)": round(rmse_dollar, 2),
    }


def build_benchmark_table(
    models: dict,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test
) -> pd.DataFrame:
    """
    Evaluates a dict of fitted models across Train/Val/Test splits.
    Returns a sorted DataFrame with all metric columns.
    """
    rows = []
    for name, model in models.items():
        tr_metrics = compute_metrics(y_train, model.predict(X_train))
        va_metrics = compute_metrics(y_val, model.predict(X_val))
        te_metrics = compute_metrics(y_test, model.predict(X_test))

        rows.append({
            "Model": name,
            "Train R2": tr_metrics["R2"],
            "Val R2": va_metrics["R2"],
            "Test R2": te_metrics["R2"],
            "Train RMSE (Log)": tr_metrics["RMSE (Log)"],
            "Val RMSE (Log)": va_metrics["RMSE (Log)"],
            "Test RMSE (Log)": te_metrics["RMSE (Log)"],
            "Test MAE ($)": te_metrics["MAE ($)"],
            "Test RMSE ($)": te_metrics["RMSE ($)"],
            "Overfit Gap (R2)": round(tr_metrics["R2"] - va_metrics["R2"], 4),
        })

    df = pd.DataFrame(rows).sort_values("Test RMSE (Log)", ascending=True).reset_index(drop=True)
    return df
