"""
Interactive Plotly visualization module.
Generates all project charts and saves them as self-contained HTML files.
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_price_distribution(df: pd.DataFrame, out_dir: str):
    """Raw vs log-transformed SalePrice distribution."""
    raw_skew = df["SalePrice"].skew()
    log_skew = np.log1p(df["SalePrice"]).skew()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Raw SalePrice  (skewness={raw_skew:.2f})",
            f"log1p(SalePrice)  (skewness={log_skew:.2f})"
        )
    )
    fig.add_trace(go.Histogram(x=df["SalePrice"], nbinsx=50,
                               marker_color="#1f77b4", name="Raw"), row=1, col=1)
    fig.add_trace(go.Histogram(x=np.log1p(df["SalePrice"]), nbinsx=50,
                               marker_color="#2ca02c", name="Log"), row=1, col=2)
    fig.update_layout(
        title_text="SalePrice Distribution: Raw vs log1p Transformation",
        template="plotly_white", showlegend=False
    )
    fig.write_html(os.path.join(out_dir, "price_distribution.html"))
    print("  Saved: price_distribution.html")


def plot_outlier_scatter(df: pd.DataFrame, out_dir: str):
    """GrLivArea vs SalePrice scatter with outlier rectangle."""
    fig = px.scatter(
        df, x="GrLivArea", y="SalePrice",
        color="OverallQual",
        hover_data=["Neighborhood", "YearBuilt"],
        title="Living Area vs SalePrice: Documented Extreme Outlier Region",
        labels={"GrLivArea": "Above Grade Living Area (sq ft)", "SalePrice": "Sale Price ($)"},
        color_continuous_scale="Viridis",
        template="plotly_white"
    )
    fig.add_shape(type="rect", x0=4000, y0=0, x1=5800, y1=300000,
                  line=dict(color="Red", width=2, dash="dash"),
                  fillcolor="rgba(255,0,0,0.12)")
    fig.add_annotation(x=4700, y=180000, text="Extreme Outliers (Removed)",
                       showarrow=True, arrowhead=2, ax=50, ay=-40, font=dict(color="red"))
    fig.write_html(os.path.join(out_dir, "living_area_vs_price.html"))
    print("  Saved: living_area_vs_price.html")


def plot_mi_ranking(mi_df: pd.DataFrame, out_dir: str, top_n: int = 25):
    """Horizontal bar chart of Mutual Information scores."""
    top = mi_df.head(top_n).copy()
    fig = px.bar(
        top, x="MI_Score", y="Feature", orientation="h",
        title=f"Top {top_n} Predictors by Mutual Information Regression Score",
        color="MI_Score", color_continuous_scale="Blues",
        labels={"MI_Score": "Mutual Information Score"},
        template="plotly_white"
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), height=620)
    fig.write_html(os.path.join(out_dir, "feature_selection_mi.html"))
    print("  Saved: feature_selection_mi.html")


def plot_model_comparison_r2(benchmark_df: pd.DataFrame, out_dir: str):
    """Grouped bar chart: Train / Val / Test R2 for each model."""
    fig = go.Figure(data=[
        go.Bar(name="Train R2", x=benchmark_df["Model"], y=benchmark_df["Train R2"], marker_color="#1f77b4"),
        go.Bar(name="Val R2",   x=benchmark_df["Model"], y=benchmark_df["Val R2"],   marker_color="#ff7f0e"),
        go.Bar(name="Test R2",  x=benchmark_df["Model"], y=benchmark_df["Test R2"],  marker_color="#2ca02c"),
    ])
    fig.update_layout(
        barmode="group",
        title="Model Benchmark: R² Comparison across Train / Val / Test",
        yaxis=dict(range=[0.70, 1.0]),
        template="plotly_white"
    )
    fig.write_html(os.path.join(out_dir, "model_comparison_r2.html"))
    print("  Saved: model_comparison_r2.html")


def plot_model_errors(benchmark_df: pd.DataFrame, out_dir: str):
    """Side-by-side: Test RMSE (log) and Test MAE ($)."""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Test RMSE  (Log Scale / RMSLE)", "Test MAE  (USD $)"))
    fig.add_trace(go.Bar(x=benchmark_df["Model"], y=benchmark_df["Test RMSE (Log)"],
                         marker_color="#d62728"), row=1, col=1)
    fig.add_trace(go.Bar(x=benchmark_df["Model"], y=benchmark_df["Test MAE ($)"],
                         marker_color="#9467bd"), row=1, col=2)
    fig.update_layout(title_text="Error Metrics Comparison Across Models",
                      template="plotly_white", showlegend=False)
    fig.write_html(os.path.join(out_dir, "model_comparison_errors.html"))
    print("  Saved: model_comparison_errors.html")


def plot_actual_vs_predicted(y_true_log: np.ndarray, y_pred_log: np.ndarray,
                              winner_name: str, out_dir: str):
    """Scatter: actual vs predicted SalePrice ($) for the winning model."""
    true_dollar = np.expm1(y_true_log)
    pred_dollar = np.expm1(y_pred_log)
    min_v = min(true_dollar.min(), pred_dollar.min())
    max_v = max(true_dollar.max(), pred_dollar.max())

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=true_dollar, y=pred_dollar, mode="markers",
                             marker=dict(color="#1f77b4", opacity=0.65, size=7),
                             name="Predictions"))
    fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines",
                             line=dict(color="red", dash="dash"), name="Perfect Fit"))
    fig.update_layout(
        title=f"Winning Model ({winner_name}): Actual vs Predicted SalePrice ($)",
        xaxis_title="Actual SalePrice ($)",
        yaxis_title="Predicted SalePrice ($)",
        template="plotly_white"
    )
    fig.write_html(os.path.join(out_dir, "actual_vs_predicted.html"))
    print("  Saved: actual_vs_predicted.html")


def plot_permutation_importance(perm_df: pd.DataFrame, winner_name: str, out_dir: str):
    """Horizontal bar: permutation importance of top features."""
    fig = px.bar(
        perm_df.head(20), x="Importance", y="Feature", orientation="h",
        title=f"Top 20 Permutation-Important Features ({winner_name})",
        color="Importance", color_continuous_scale="Teal",
        labels={"Importance": "Mean R² Loss on Permutation"},
        template="plotly_white"
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), height=600)
    fig.write_html(os.path.join(out_dir, "permutation_importance.html"))
    print("  Saved: permutation_importance.html")


def plot_optuna_history(studies: dict, out_dir: str):
    """Line chart: Optuna optimization history for all models."""
    fig = go.Figure()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for (name, study), color in zip(studies.items(), colors):
        values = [t.value for t in study.trials if t.value is not None]
        best_so_far = [min(values[:i+1]) for i in range(len(values))]
        fig.add_trace(go.Scatter(x=list(range(1, len(best_so_far)+1)),
                                 y=best_so_far, mode="lines+markers",
                                 name=name, line=dict(color=color)))
    fig.update_layout(
        title="Optuna Optimization Convergence: Best CV RMSE per Trial",
        xaxis_title="Trial Number",
        yaxis_title="Best 5-Fold CV RMSE (Log Scale)",
        template="plotly_white"
    )
    fig.write_html(os.path.join(out_dir, "optuna_convergence.html"))
    print("  Saved: optuna_convergence.html")
