"""
Task 10: SHAP explainability for the Random Forest models.
"""
import shap
import numpy as np
import pandas as pd


def get_shap_values(pipeline, X_test: pd.DataFrame):
    """Return a shap.Explanation for the classifier step of the sklearn Pipeline.

    We explain on the *scaled* features because that's what the RF was trained on,
    but we keep the original column names / raw values for readable plots.
    """
    scaler = pipeline.named_steps["scaler"]
    clf = pipeline.named_steps["clf"]

    X_scaled = scaler.transform(X_test)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_test.columns, index=X_test.index)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer(X_scaled_df)

    # For binary classifiers TreeExplainer can return shape (n, features, 2);
    # keep the "UP" (class 1) contributions.
    if shap_values.values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    return shap_values, X_scaled_df


def global_feature_importance(shap_values) -> pd.DataFrame:
    """Mean |SHAP value| per feature, sorted descending - used for the bar chart."""
    importance = np.abs(shap_values.values).mean(axis=0)
    df = pd.DataFrame({"feature": shap_values.feature_names, "mean_abs_shap": importance})
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def explain_single_prediction(shap_values, index: int = -1):
    """Return the shap.Explanation slice for a single row (default: most recent)."""
    return shap_values[index]
