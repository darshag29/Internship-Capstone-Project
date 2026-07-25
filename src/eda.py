"""
Task 4: Exploratory Data Analysis helpers (Plotly figures, reused by app.py).
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_price_trend(df: pd.DataFrame, title: str = "Price Trend"):
    fig = px.line(df, x=df.index, y="Close", title=title)
    fig.update_layout(xaxis_title="Date", yaxis_title="Close Price")
    return fig


def plot_returns_distribution(df: pd.DataFrame, title: str = "Daily Returns Distribution"):
    returns = df["Close"].pct_change().dropna() * 100
    fig = px.histogram(returns, nbins=60, title=title)
    fig.update_layout(xaxis_title="Daily Return (%)", yaxis_title="Count", showlegend=False)
    return fig


def plot_volume_trend(df: pd.DataFrame, title: str = "Volume Trend"):
    fig = px.bar(df, x=df.index, y="Volume", title=title)
    fig.update_layout(xaxis_title="Date", yaxis_title="Volume")
    return fig


def plot_correlation_heatmap(price_dict: dict, title: str = "Asset Return Correlation"):
    """price_dict: {display_name: DataFrame} -> heatmap of daily return correlations."""
    returns = pd.DataFrame(
        {name: df["Close"].pct_change() for name, df in price_dict.items()}
    ).dropna()
    corr = returns.corr()
    fig = go.Figure(
        data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="RdBu", zmid=0)
    )
    fig.update_layout(title=title)
    return fig


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    stats = df[["Open", "High", "Low", "Close", "Volume"]].describe().T
    stats["daily_return_mean_%"] = df["Close"].pct_change().mean() * 100
    stats["daily_return_std_%"] = df["Close"].pct_change().std() * 100
    return stats
