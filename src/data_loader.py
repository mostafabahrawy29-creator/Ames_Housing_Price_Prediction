"""
Data loading and initial sanity verification module.
Handles loading train/test data and basic structural checks.
"""

import os
import pandas as pd


def load_dataset(file_path: str = None) -> pd.DataFrame:
    """
    Load dataset from csv file.
    Defaults to relative path data/train.csv.
    """
    if file_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(base_dir, "data", "train.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at expected location: {file_path}")

    df = pd.read_csv(file_path)
    return df


def handle_structural_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    In the Ames housing dataset, 'NA' in specific columns signifies
    the physical absence of a feature (e.g. No Garage, No Basement, No Pool)
    rather than unrecorded or missing data.
    """
    structural_none_columns = [
        'Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
        'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType',
        'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature'
    ]
    df_clean = df.copy()
    for col in structural_none_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('None')

    if 'Id' in df_clean.columns:
        df_clean = df_clean.drop(columns=['Id'])

    return df_clean


def remove_documented_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes the documented extreme outliers recommended by Dean De Cock:
    Properties with above grade living area > 4,000 sq ft and SalePrice < 300,000.
    """
    if 'GrLivArea' in df.columns and 'SalePrice' in df.columns:
        mask = (df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)
        return df[~mask].reset_index(drop=True)
    return df
