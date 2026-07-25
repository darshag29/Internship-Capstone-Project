"""
Task 1: Fetch 5 years of historical OHLCV data for all assets via yfinance.
Task 2: Fetch financial news headlines for all assets via NewsAPI.
"""
import time
import requests
import pandas as pd
import yfinance as yf

from src.config import ASSETS, NEWS_QUERY_MAP, NEWSAPI_KEY, DATA_DIR


def fetch_stock_data(assets=None, period="5y") -> dict:
    """Download OHLCV history for every ticker and save to data/data_<ticker>.csv.

    Returns a dict of {ticker: DataFrame}.
    """
    assets = assets or ASSETS
    results = {}
    for ticker in assets:
        print(f"Downloading {ticker} ...")
        df = yf.download(ticker, period=period, auto_adjust=True)

        # yfinance can return MultiIndex columns when downloading a single
        # ticker in some versions -- flatten defensively.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty:
            print(f"  WARNING: no data returned for {ticker}")
            continue

        df.index.name = "Date"
        out_path = DATA_DIR / f"data_{ticker.replace('^', '').replace('.', '_')}.csv"
        df.to_csv(out_path)
        results[ticker] = df
        print(f"  saved {len(df)} rows -> {out_path.name}")
    return results


def fetch_news(company: str, api_key: str = None, page_size: int = 50) -> list:
    """Fetch recent news articles mentioning `company` from NewsAPI."""
    api_key = api_key or NEWSAPI_KEY
    if not api_key:
        raise ValueError(
            "NEWSAPI_KEY is not set. Add it to your .env file "
            "(see .env.example) or pass api_key explicitly."
        )

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": company,
        "apiKey": api_key,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
    }
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    payload = response.json()
    return payload.get("articles", [])


def fetch_all_news(assets=None) -> pd.DataFrame:
    """Fetch news for every asset and save a combined CSV.

    NewsAPI free tier is rate-limited, so we sleep briefly between calls.
    """
    assets = assets or ASSETS
    rows = []
    for ticker in assets:
        query = NEWS_QUERY_MAP.get(ticker, ticker)
        print(f"Fetching news for {ticker} ({query}) ...")
        try:
            articles = fetch_news(query)
        except Exception as exc:  # noqa: BLE001 - surface but keep going
            print(f"  ERROR fetching news for {ticker}: {exc}")
            continue

        for a in articles:
            rows.append(
                {
                    "ticker": ticker,
                    "title": a.get("title"),
                    "description": a.get("description"),
                    "content": a.get("content"),
                    "source": (a.get("source") or {}).get("name"),
                    "publishedAt": a.get("publishedAt"),
                    "url": a.get("url"),
                }
            )
        time.sleep(1)  # be polite to the free tier rate limit

    news_df = pd.DataFrame(rows)
    if not news_df.empty:
        news_df["publishedAt"] = pd.to_datetime(news_df["publishedAt"], errors="coerce")
    out_path = DATA_DIR / "news_data.csv"
    news_df.to_csv(out_path, index=False)
    print(f"Saved {len(news_df)} news rows -> {out_path.name}")
    return news_df


if __name__ == "__main__":
    fetch_stock_data()
    fetch_all_news()
