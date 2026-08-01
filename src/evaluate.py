"""Reusable evaluation functions for trained regression models.

All plotting functions return the matplotlib Figure they draw on (rather
than calling plt.show() internally) so a notebook can decide whether to
display it, save it, or both — this also keeps the functions usable
outside a notebook context, e.g. from a future evaluation script.
"""

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true, y_pred) -> dict:
    """Compute RMSE, MAE, and R² for a set of predictions.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted target values, same length as y_true.

    Returns:
        Dict with keys "rmse", "mae", "r2".
    """
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_comparison_table(results: dict) -> pd.DataFrame:
    """Assemble a model comparison table from a results dict.

    Args:
        results: Mapping of model name -> dict of metric name -> value
            (e.g. "train_r2", "test_r2", "test_rmse", "test_mae",
            "cv_r2_mean", "cv_r2_std").

    Returns:
        DataFrame with one row per model, sorted by test_r2 descending.
    """
    table = pd.DataFrame(results).T
    return table.sort_values("test_r2", ascending=False)


def plot_actual_vs_predicted(
    y_true, y_pred, title: str = "Actual vs Predicted", price_label: str = "Price (USD)"
) -> Figure:
    """Plot actual vs predicted values with a perfect-prediction reference line.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted target values, same length as y_true.
        title: Plot title.
        price_label: Axis label, stating the currency explicitly.

    Returns:
        The matplotlib Figure containing the plot.
    """
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.3, s=12)
    limits = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(limits, limits, color="red", linestyle="--", label="Perfect prediction (y = x)")
    ax.set_xlabel(f"Actual {price_label}")
    ax.set_ylabel(f"Predicted {price_label}")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_residuals(
    y_true, y_pred, title: str = "Residual Plot", price_label: str = "Price (USD)"
) -> Figure:
    """Plot residuals (actual - predicted) against predicted values.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted target values, same length as y_true.
        title: Plot title.
        price_label: Axis label, stating the currency explicitly.

    Returns:
        The matplotlib Figure containing the plot.
    """
    import matplotlib.pyplot as plt

    residuals = np.asarray(y_true) - np.asarray(y_pred)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(y_pred, residuals, alpha=0.3, s=12)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel(f"Predicted {price_label}")
    ax.set_ylabel(f"Residual ({price_label}) = Actual - Predicted")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_error_distribution(
    y_true, y_pred, title: str = "Error Distribution", price_label: str = "Price (USD)"
) -> Figure:
    """Plot a histogram of prediction errors (actual - predicted).

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted target values, same length as y_true.
        title: Plot title.
        price_label: Axis label, stating the currency explicitly.

    Returns:
        The matplotlib Figure containing the plot.
    """
    import matplotlib.pyplot as plt

    residuals = np.asarray(y_true) - np.asarray(y_pred)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(residuals, bins=60)
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel(f"Error ({price_label}) = Actual - Predicted")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_feature_importance(
    feature_names, importances, top_n: int = 20, title: str = "Feature Importance"
) -> Figure:
    """Plot a horizontal bar chart of the top-N most important features.

    Args:
        feature_names: Sequence of feature names, aligned with importances.
        importances: Sequence of importance scores, aligned with feature_names.
        top_n: Number of top features to display.
        title: Plot title.

    Returns:
        The matplotlib Figure containing the plot.
    """
    import matplotlib.pyplot as plt

    feature_names = np.asarray(feature_names)
    importances = np.asarray(importances)
    order = np.argsort(importances)[::-1][:top_n]
    top_names = feature_names[order]
    top_scores = importances[order]

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.3)))
    ax.barh(top_names[::-1], top_scores[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    return fig
