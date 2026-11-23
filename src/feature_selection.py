"""
Feature selection module.
Includes variance thresholding, Mutual Information regression ranking,
and feature subset extraction.
"""

from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, VarianceThreshold


def calculate_mutual_information(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    random_state: int = 42
) -> pd.DataFrame:
    """
    Computes Mutual Information (MI) scores between predictors and the continuous target.
    Captures both linear and non-linear associations.
    """
    scores = mutual_info_regression(X, y, random_state=random_state)
    mi_df = pd.DataFrame({
        'Feature': feature_names,
        'MI_Score': scores
    }).sort_values('MI_Score', ascending=False).reset_index(drop=True)
    return mi_df


def apply_variance_threshold(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    threshold: float = 0.005
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Filters out quasi-constant features with variance below threshold.
    """
    selector = VarianceThreshold(threshold=threshold)
    X_train_sel = selector.fit_transform(X_train)
    X_val_sel = selector.transform(X_val)
    X_test_sel = selector.transform(X_test)

    mask = selector.get_support()
    selected_names = [name for name, keep in zip(feature_names, mask) if keep]

    return X_train_sel, X_val_sel, X_test_sel, selected_names
