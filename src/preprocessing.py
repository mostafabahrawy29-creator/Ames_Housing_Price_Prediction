"""
Preprocessing pipelines and ColumnTransformer construction.
Builds leak-free transformations for skewed numerical, standard numerical,
nominal categorical, and ordinal categorical features.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    OneHotEncoder,
    OrdinalEncoder,
    FunctionTransformer
)

ORDINAL_COLS = [
    'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
    'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual',
    'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond',
    'PoolQC', 'Fence', 'Functional'
]

QUALITY_ORDER = ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
ORDINAL_CATEGORIES = {
    'ExterQual': QUALITY_ORDER,
    'ExterCond': QUALITY_ORDER,
    'BsmtQual': QUALITY_ORDER,
    'BsmtCond': QUALITY_ORDER,
    'BsmtExposure': ['None', 'No', 'Mn', 'Av', 'Gd'],
    'BsmtFinType1': ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
    'BsmtFinType2': ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
    'HeatingQC': QUALITY_ORDER,
    'KitchenQual': QUALITY_ORDER,
    'FireplaceQu': QUALITY_ORDER,
    'GarageFinish': ['None', 'Unf', 'RFn', 'Fin'],
    'GarageQual': QUALITY_ORDER,
    'GarageCond': QUALITY_ORDER,
    'PoolQC': QUALITY_ORDER,
    'Fence': ['None', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv'],
    'Functional': ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'],
}


def build_preprocessor(
    X_train: pd.DataFrame,
    skew_threshold: float = 0.75
) -> Tuple[ColumnTransformer, List[str], List[str], List[str], List[str]]:
    """
    Builds a ColumnTransformer tailored to the training set schema.
    Skewness is calculated exclusively on X_train to prevent data leakage.
    """
    available_cols = X_train.columns.tolist()
    ord_cols = [c for c in ORDINAL_COLS if c in available_cols]
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    nom_cols = [c for c in cat_cols if c not in ord_cols]

    # Calculate skewness on training data
    train_skew = X_train[num_cols].skew().abs()
    skewed_num = train_skew[train_skew > skew_threshold].index.tolist()
    normal_num = [c for c in num_cols if c not in skewed_num]

    ord_cats_list = [ORDINAL_CATEGORIES[c] for c in ord_cols if c in ORDINAL_CATEGORIES]

    pipe_num_skewed = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('log1p', FunctionTransformer(np.log1p, validate=False, feature_names_out='one-to-one')),
        ('scaler', RobustScaler())
    ])

    pipe_num_normal = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    pipe_nominal = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])

    pipe_ordinal = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('oe', OrdinalEncoder(categories=ord_cats_list, handle_unknown='use_encoded_value', unknown_value=-1)),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num_skewed', pipe_num_skewed, skewed_num),
        ('num_normal', pipe_num_normal, normal_num),
        ('nominal', pipe_nominal, nom_cols),
        ('ordinal', pipe_ordinal, ord_cols)
    ], remainder='drop')

    return preprocessor, skewed_num, normal_num, nom_cols, ord_cols
