"""
Domain-specific feature engineering module for housing data.
Derives composite area, bath counts, property age, and interaction terms.
"""

import pandas as pd


def construct_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs informative domain features:
    1. TotalSF: Total square footage combining basement, 1st, and 2nd floor.
    2. TotalBath: Weighted bathroom sum (full baths + 0.5 * half baths).
    3. HouseAge: Age of property at year of sale.
    4. RemodAge: Years since remodel at year of sale.
    5. TotalPorchSF: Aggregate porch and deck surface areas.
    6. Binary amenity indicators (HasPool, HasGarage, HasBasement, WasRemodeled).
    7. Interaction terms (OverallGrade = Qual * Cond, QualArea = Qual * GrLivArea).
    """
    d = df.copy()

    # Total square footage
    if {'TotalBsmtSF', '1stFlrSF', '2ndFlrSF'}.issubset(d.columns):
        d['TotalSF'] = d['TotalBsmtSF'] + d['1stFlrSF'] + d['2ndFlrSF']

    # Total bathrooms
    bath_cols = {'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath'}
    if bath_cols.issubset(d.columns):
        d['TotalBath'] = (
            d['FullBath'] + 0.5 * d['HalfBath'] +
            d['BsmtFullBath'] + 0.5 * d['BsmtHalfBath']
        )

    # Age variables
    if {'YrSold', 'YearBuilt'}.issubset(d.columns):
        d['HouseAge'] = d['YrSold'] - d['YearBuilt']
    if {'YrSold', 'YearRemodAdd'}.issubset(d.columns):
        d['RemodAge'] = d['YrSold'] - d['YearRemodAdd']

    # Porch area
    porch_cols = {'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'WoodDeckSF'}
    if porch_cols.issubset(d.columns):
        d['TotalPorchSF'] = (
            d['OpenPorchSF'] + d['EnclosedPorch'] +
            d['3SsnPorch'] + d['ScreenPorch'] + d['WoodDeckSF']
        )

    # Binary amenity indicators
    if 'PoolArea' in d.columns:
        d['HasPool'] = (d['PoolArea'] > 0).astype(int)
    if 'GarageArea' in d.columns:
        d['HasGarage'] = (d['GarageArea'] > 0).astype(int)
    if 'TotalBsmtSF' in d.columns:
        d['HasBasement'] = (d['TotalBsmtSF'] > 0).astype(int)
    if {'YearRemodAdd', 'YearBuilt'}.issubset(d.columns):
        d['WasRemodeled'] = (d['YearRemodAdd'] != d['YearBuilt']).astype(int)

    # Interactions
    if {'OverallQual', 'OverallCond'}.issubset(d.columns):
        d['OverallGrade'] = d['OverallQual'] * d['OverallCond']
    if {'OverallQual', 'GrLivArea'}.issubset(d.columns):
        d['QualArea'] = d['OverallQual'] * d['GrLivArea']

    return d
