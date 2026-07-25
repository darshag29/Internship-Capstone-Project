"""
Run the full pipeline end-to-end from the command line (Tasks 1-10),
without launching the Streamlit app. Useful for a first data pull,
debugging, or generating a written report.

Usage:
    python main.py
"""
from src.data_fetch import fetch_stock_data, fetch_all_news
from src.model import run_all_assets, metrics_comparison_table, predict_next_direction
from src.business_metrics import find_optimal_threshold
from src.explainability import get_shap_values, global_feature_importance
from src.config import ASSET_DISPLAY_NAMES


def main():
    print("=" * 60)
    print("Task 1-2: Fetching stock + news data")
    print("=" * 60)
    fetch_stock_data()
    try:
        fetch_all_news()
    except Exception as exc:
        print(f"News fetch skipped/failed: {exc}")

    print("\n" + "=" * 60)
    print("Task 6-7: Training models for all assets")
    print("=" * 60)
    results = run_all_assets()

    print("\n" + "=" * 60)
    print("Task 8: Business metrics comparison table")
    print("=" * 60)
    print(metrics_comparison_table(results))

    print("\n" + "=" * 60)
    print("Task 9: Optimal thresholds per asset")
    print("=" * 60)
    for ticker, r in results.items():
        tuning = find_optimal_threshold(r["y_test"], r["y_proba"])
        print(f"{ASSET_DISPLAY_NAMES.get(ticker, ticker):12s} "
              f"optimal_threshold={tuning['optimal_threshold']:.2f}  "
              f"savings_vs_default=${tuning['savings_vs_default']:.0f}")

    print("\n" + "=" * 60)
    print("Task 10: Top SHAP features per asset")
    print("=" * 60)
    for ticker, r in results.items():
        shap_values, _ = get_shap_values(r["model"], r["X_test"])
        top_features = global_feature_importance(shap_values).head(5)
        print(f"\n{ASSET_DISPLAY_NAMES.get(ticker, ticker)}:")
        print(top_features.to_string(index=False))

    print("\n" + "=" * 60)
    print("Tomorrow's predictions")
    print("=" * 60)
    for ticker, r in results.items():
        pred = predict_next_direction(r)
        print(f"{ASSET_DISPLAY_NAMES.get(ticker, ticker):12s} -> "
              f"{pred['direction']} (confidence {pred['confidence']*100:.1f}%)")


if __name__ == "__main__":
    main()
