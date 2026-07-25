"""
Task 9: Threshold tuning based on business costs.

False positive  = model says UP, actually DOWN -> a bad trade is taken (costly)
False negative = model says DOWN, actually UP -> a good trade is missed (cheaper, opportunity cost)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.config import COST_FALSE_POSITIVE, COST_FALSE_NEGATIVE


def cost_at_threshold(y_test, y_proba, threshold, cost_fp=COST_FALSE_POSITIVE, cost_fn=COST_FALSE_NEGATIVE):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    total_cost = (fp * cost_fp) + (fn * cost_fn)
    return total_cost, {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def find_optimal_threshold(y_test, y_proba, cost_fp=COST_FALSE_POSITIVE, cost_fn=COST_FALSE_NEGATIVE):
    """Task 9: sweep thresholds 0.1-0.9 and return the one that minimizes total cost."""
    thresholds = np.arange(0.1, 0.91, 0.05)
    records = []
    for thresh in thresholds:
        cost, cm = cost_at_threshold(y_test, y_proba, thresh, cost_fp, cost_fn)
        records.append({"threshold": round(thresh, 2), "total_cost": cost, **cm})

    cost_df = pd.DataFrame(records)
    best_row = cost_df.loc[cost_df["total_cost"].idxmin()]
    default_cost, _ = cost_at_threshold(y_test, y_proba, 0.5, cost_fp, cost_fn)

    return {
        "cost_table": cost_df,
        "optimal_threshold": float(best_row["threshold"]),
        "optimal_cost": float(best_row["total_cost"]),
        "default_threshold_cost": float(default_cost),
        "savings_vs_default": float(default_cost - best_row["total_cost"]),
    }
