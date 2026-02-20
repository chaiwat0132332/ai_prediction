# src/data/clean.py

import pandas as pd
import numpy as np


def clean_series(df: pd.DataFrame, target_col: str) -> pd.Series:
    """
    Clean and validate time series data

    Steps:
    - convert to numeric
    - remove NaN
    - reset index
    - ensure float
    """

    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found")

    series = pd.to_numeric(
        df[target_col],
        errors="coerce"
    )

    # remove NaN
    series = series.dropna()

    if len(series) == 0:
        raise ValueError("No valid numeric data found")

    # reset index
    series = series.reset_index(drop=True)

    return series.astype(float)


def validate_series_length(series: pd.Series, lag: int):
    """
    Ensure enough data for training
    """

    if len(series) <= lag:
        raise ValueError(
            f"Data length ({len(series)}) must be > lag ({lag})"
        )