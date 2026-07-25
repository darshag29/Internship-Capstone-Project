"""
Task 5: Feature engineering.

Builds price, trend, volatility, time and volume features, plus the
next-day UP/DOWN target used for classification.
"""
import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    # Price features
    "daily_return", "log_return", "price_to_ma7", "price_to_ma14", "high_low_ratio",
    # Trend features
    "ma_7", "ma_14", "ma_30", "ma_crossover",
    # Volatility features
    "volatility_7", "volatility_14",
    # Time features
    "day_of_week", "month", "quarter", "is_monday", "is_friday",
    # Volume features
    "volume_ma_7", "volume_ratio",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Take a cleaned OHLCV DataFrame (DatetimeIndex) and return it enriched
    with 15+ engineered features and a next-day target column ('target').
    """
    df = df.copy()

    # ---- Price features ----
    df["daily_return"] = df["Close"].pct_change() * 100
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["high_low_ratio"] = df["High"] / df["Low"]

    # ---- Trend features ----
    df["ma_7"] = df["Close"].rolling(7).mean()
    df["ma_14"] = df["Close"].rolling(14).mean()
    df["ma_30"] = df["Close"].rolling(30).mean()
    df["price_to_ma7"] = df["Close"] / df["ma_7"]
    df["price_to_ma14"] = df["Close"] / df["ma_14"]
    df["ma_crossover"] = (df["ma_7"] > df["ma_14"]).astype(int)

    # ---- Volatility features ----
    df["volatility_7"] = df["daily_return"].rolling(7).std()
    df["volatility_14"] = df["daily_return"].rolling(14).std()

    # ---- Time features ----
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["is_monday"] = (df["day_of_week"] == 0).astype(int)
    df["is_friday"] = (df["day_of_week"] == 4).astype(int)

    # ---- Volume features ----
    df["volume_ma_7"] = df["Volume"].rolling(7).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_ma_7"]

    # ---- Target: next-day direction (1 = UP, 0 = DOWN) ----
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # Rolling windows create leading NaNs; the shifted target creates a
    # trailing NaN on the last row. Drop both.
    df = df.dropna()
    return df
