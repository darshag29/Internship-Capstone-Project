"""
Task 3: Clean stock and news data.
"""
import pandas as pd


def clean_stock_df(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten columns, fix dtypes, handle missing values for a single asset's OHLCV data."""
    df = df.copy()

    # Flatten MultiIndex columns if present (common with yfinance multi-ticker downloads)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure the index is a proper datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
        else:
            df.index = pd.to_datetime(df.index)

    df = df.sort_index()

    # Report missing values before handling
    n_missing = df.isna().sum().sum()
    if n_missing:
        print(f"  found {n_missing} missing values -> forward-filling, then dropping remainder")

    # Forward-fill small gaps (e.g. holidays merged incorrectly), then drop any
    # rows that are still incomplete (e.g. leading NaNs before the series starts)
    df = df.ffill().dropna()

    # Make sure numeric columns are actually numeric
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()
    return df


def clean_news_df(news_df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with no usable text, fix dtypes, dedupe."""
    df = news_df.copy()
    if df.empty:
        return df

    df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
    df = df.dropna(subset=["title"])
    df = df.drop_duplicates(subset=["title", "ticker"])
    df["description"] = df["description"].fillna("")
    df["content"] = df["content"].fillna("")
    return df.sort_values("publishedAt", ascending=False)
