"""
Task 6: process_asset(ticker) - full pipeline for a single ticker:
        download -> clean -> engineer features -> time split -> train -> evaluate.
Task 7: run_all_assets() - loop process_asset over every configured ticker.
"""
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import ASSETS, MODELS_DIR, TEST_SIZE_FRACTION, RANDOM_STATE
from src.data_cleaning import clean_stock_df
from src.feature_engineering import engineer_features, FEATURE_COLUMNS


def process_asset(ticker: str, period: str = "5y") -> dict:
    """Build and evaluate a Random Forest direction-classifier for `ticker`.

    Returns a dict with keys: ticker, model, metrics, X_test, y_test, y_pred,
    y_proba, feature_df (full engineered frame, for SHAP/plots), test_dates.
    """
    # 1. Download
    raw = yf.download(ticker, period=period, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}")

    # 2. Clean
    clean = clean_stock_df(raw)

    # 3. Feature engineering
    feat_df = engineer_features(clean)

    X = feat_df[FEATURE_COLUMNS]
    y = feat_df["target"]

    # 4. Time-based split (NOT random) - last TEST_SIZE_FRACTION of rows are test
    split_idx = int(len(feat_df) * (1 - TEST_SIZE_FRACTION))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    test_dates = feat_df.index[split_idx:]

    # 5. Pipeline: Scaler + Random Forest
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])
    pipeline.fit(X_train, y_train)

    # 6. Evaluate
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_proba) if y_test.nunique() > 1 else float("nan"),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    # Save model to disk for reuse by the Streamlit app
    model_path = MODELS_DIR / f"model_{ticker.replace('^', '').replace('.', '_')}.joblib"
    joblib.dump(pipeline, model_path)

    return {
        "ticker": ticker,
        "model": pipeline,
        "metrics": metrics,
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "y_pred": y_pred, "y_proba": y_proba,
        "feature_df": feat_df,
        "test_dates": test_dates,
        "model_path": str(model_path),
    }


def run_all_assets(assets=None) -> dict:
    """Task 7: process every configured asset, returning {ticker: result_dict}."""
    assets = assets or ASSETS
    results = {}
    for ticker in assets:
        print(f"Processing {ticker} ...")
        try:
            results[ticker] = process_asset(ticker)
            print(f"  accuracy={results[ticker]['metrics']['accuracy']:.3f} "
                  f"auc={results[ticker]['metrics']['auc']:.3f}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR processing {ticker}: {exc}")
    return results


def metrics_comparison_table(results: dict) -> pd.DataFrame:
    """Task 8: build a comparison table of business metrics across assets."""
    rows = []
    for ticker, r in results.items():
        row = {"ticker": ticker, **r["metrics"]}
        rows.append(row)
    return pd.DataFrame(rows).set_index("ticker").sort_values("accuracy", ascending=False)


def predict_next_direction(result: dict) -> dict:
    """Use the most recent row of features to predict tomorrow's direction."""
    latest_row = result["feature_df"][FEATURE_COLUMNS].iloc[[-1]]
    proba_up = result["model"].predict_proba(latest_row)[0, 1]
    direction = "UP" if proba_up >= 0.5 else "DOWN"
    return {"direction": direction, "confidence": float(proba_up if direction == "UP" else 1 - proba_up)}


if __name__ == "__main__":
    all_results = run_all_assets()
    print(metrics_comparison_table(all_results))
