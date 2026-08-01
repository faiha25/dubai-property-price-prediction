"""SHAP explainability for the final tuned XGBoost model.

The saved model (models/best_model_xgboost.joblib) is a
TransformedTargetRegressor wrapping Pipeline(preprocessor, XGBRegressor),
trained on log1p(price_usd) (see notebooks/03_modeling.ipynb, Section 7).
SHAP values computed here explain the underlying XGBRegressor's
contribution to that LOG-transformed target, not raw USD price — a
feature's SHAP value is in log-price units, not dollars. This is stated
wherever SHAP output is shown, so magnitudes aren't misread as dollar
amounts.

Functions here operate on the fitted pipeline extracted from a loaded
model, using the same feature schema built in src/feature_engineering.py
and notebooks/03_modeling.ipynb. They are reusable from a notebook, a
script, or app.py alike.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from matplotlib.figure import Figure
from sklearn.compose import TransformedTargetRegressor


def load_trained_model(path: Path) -> TransformedTargetRegressor:
    """Load the saved tuned model pipeline from disk.

    Args:
        path: Path to the joblib file (e.g. models/best_model_xgboost.joblib).

    Returns:
        The fitted TransformedTargetRegressor, ready for .predict() or explanation.
    """
    return joblib.load(path)


def transform_features(
    model: TransformedTargetRegressor, X: pd.DataFrame
) -> tuple[np.ndarray, list[str]]:
    """Run X through the model's fitted preprocessing pipeline only (no prediction).

    Args:
        model: A fitted TransformedTargetRegressor from load_trained_model().
        X: Raw, feature-engineered rows — same schema as the feature_cols
            used in notebooks/03_modeling.ipynb (output of
            src.feature_engineering.engineer_static_features, still
            containing the raw "zone" column for LocationTierEncoder).

    Returns:
        Tuple of (dense numeric array XGBoost actually sees, feature
        names with ColumnTransformer step prefixes stripped for
        readability).
    """
    preprocessor = model.regressor_.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(X)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    feature_names = [name.split("__", 1)[-1] for name in preprocessor.get_feature_names_out()]
    return X_transformed, feature_names


def compute_shap_values(
    model: TransformedTargetRegressor, X: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Compute SHAP values for the underlying XGBoost regressor.

    Uses shap.TreeExplainer, which is exact (not approximate) for
    tree-based models like XGBoost.

    Args:
        model: A fitted TransformedTargetRegressor from load_trained_model().
        X: Raw, feature-engineered rows to explain (see transform_features).

    Returns:
        Tuple of (shap_values array, transformed feature matrix, feature names).
    """
    xgb_model = model.regressor_.named_steps["model"]
    X_transformed, feature_names = transform_features(model, X)
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_transformed)
    return shap_values, X_transformed, feature_names


def plot_shap_summary(
    shap_values: np.ndarray,
    X_transformed: np.ndarray,
    feature_names: list[str],
    max_display: int = 20,
) -> Figure:
    """Generate a SHAP beeswarm summary plot.

    Args:
        shap_values: Array from compute_shap_values().
        X_transformed: Transformed feature matrix from compute_shap_values().
        feature_names: Feature names from compute_shap_values().
        max_display: Maximum number of features to show.

    Returns:
        The matplotlib Figure containing the plot.
    """
    import matplotlib.pyplot as plt

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def save_shap_summary(fig: Figure, path: Path) -> None:
    """Save a SHAP summary figure to disk.

    Args:
        fig: Figure returned by plot_shap_summary().
        path: Destination PNG path. Parent directory must already exist.
    """
    fig.savefig(path, dpi=150, bbox_inches="tight")
