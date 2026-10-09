"""
Preprocessing and time-series feature engineering module.
Engineered by M. Abdullah.
Features chronological train-test splitting to strictly prevent temporal lookahead leakage.
"""
from typing import Tuple
import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """Load sales data, convert dates, sort chronologically, and aggregate by date if needed."""
    df = pd.read_csv(filepath, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate calendar and lag features for supervised regression forecasting.
    Includes day-of-week, day-of-month, month, lag_1, lag_7, lag_14, and rolling window aggregations.
    """
    df = df.copy()
    df["dayofweek"] = df["date"].dt.dayofweek
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    # Autoregressive lag features
    df["lag_1"] = df["sales"].shift(1)
    df["lag_7"] = df["sales"].shift(7)
    df["lag_14"] = df["sales"].shift(14)

    # Rolling statistics (using closed='left' shift to avoid target leakage)
    df["rolling_mean_7"] = df["sales"].shift(1).rolling(window=7).mean()
    df["rolling_std_7"] = df["sales"].shift(1).rolling(window=7).std()
    df["rolling_mean_14"] = df["sales"].shift(1).rolling(window=14).mean()

    # Drop burn-in rows with missing lag values
    df = df.dropna().reset_index(drop=True)
    return df


def train_test_split_temporal(df: pd.DataFrame, test_days: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological out-of-sample partition.
    Takes the final `test_days` rows as holdout evaluation period.
    """
    if len(df) <= test_days:
        raise ValueError(f"Dataframe length ({len(df)}) must exceed test period ({test_days}).")
    train_df = df.iloc[:-test_days].copy()
    test_df = df.iloc[-test_days:].copy()
    return train_df, test_df
