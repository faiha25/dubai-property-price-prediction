"""Feature engineering functions for the Dubai secondary sales modeling dataset.

These functions are kept separate from cleaning (src/preprocessing.py) per
the project's cleaning/engineering split: cleaning fixes problems with the
raw data, engineering creates new information from already-clean data.

Two categories of feature behave differently with respect to the target
variable:

- Row-level, target-independent features (size_category, property_age,
  has_floor_info, log_area_sqft, luxury_indicator) are pure functions of a
  single row and are safe to compute on the full dataset before any
  train/test split — none of them reference price_usd.
- location_tier is derived from historical median price per zone, which
  means it must be fit on training data only (per fold, during
  cross-validation) to avoid leaking target information. It is implemented
  as a scikit-learn-compatible transformer (LocationTierEncoder) so it can
  be embedded directly inside a Pipeline/ColumnTransformer, where
  scikit-learn automatically enforces the correct fit/transform boundary
  on every cross-validation split.

Two features suggested by the original project blueprint are NOT
implemented because the required columns do not exist in this dataset:
bedroom_bathroom_ratio (no `bathrooms` column) and amenities_density (no
`amenities_count` column). Fabricating them would misrepresent data that
was never collected.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SIZE_CATEGORY_BINS = [0, 500, 1000, 2000, 3500, np.inf]
SIZE_CATEGORY_LABELS = ["studio", "small", "medium", "large", "very_large"]


def add_size_category(df: pd.DataFrame, area_col: str = "area_sqft") -> pd.DataFrame:
    """Bin property area into named size categories.

    Property value does not scale perfectly linearly with size — a
    4,000 sqft home does not cost exactly twice a 2,000 sqft home — so a
    categorical size band lets models split on broad size tiers directly,
    alongside the continuous area feature.

    Args:
        df: DataFrame containing area_col.
        area_col: Name of the numeric area column to bin.

    Returns:
        A copy of df with a new "size_category" column.
    """
    df = df.copy()
    df["size_category"] = pd.cut(
        df[area_col], bins=SIZE_CATEGORY_BINS, labels=SIZE_CATEGORY_LABELS
    )
    return df


def add_property_age(
    df: pd.DataFrame,
    year_built_col: str = "year_built",
    date_col: str = "date_listed",
) -> pd.DataFrame:
    """Compute property age at the time of listing.

    Args:
        df: DataFrame containing year_built_col and date_col.
        year_built_col: Column with the construction year.
        date_col: Column with the listing date (datetime-parseable).

    Returns:
        A copy of df with a new "property_age" column, computed as the
        listing year minus year_built. The listing year is used instead
        of a single fixed "current year", since listings in this dataset
        span 2020-2026 and a fixed reference year would compute a
        different, inconsistent age for the same building depending only
        on when the notebook happens to be re-run.
    """
    df = df.copy()
    df["property_age"] = pd.to_datetime(df[date_col]).dt.year - df[year_built_col]
    return df


def add_has_floor_info(
    df: pd.DataFrame, category_col: str = "property_category"
) -> pd.DataFrame:
    """Flag whether floor/total_floors is structurally available for a row.

    floor and total_floors are missing for every villa (the concept does
    not apply to a standalone house) and present for every apartment.
    This flag makes that missingness explicit to the model, rather than
    relying on it to infer the pattern from an imputed placeholder value
    alone.

    Args:
        df: DataFrame containing category_col.
        category_col: Column identifying apartment vs villa.

    Returns:
        A copy of df with a new binary "has_floor_info" column (1 for
        apartment, 0 for villa).
    """
    df = df.copy()
    df["has_floor_info"] = (df[category_col] == "apartment").astype(int)
    return df


def add_log_area(df: pd.DataFrame, area_col: str = "area_sqft") -> pd.DataFrame:
    """Add a log-transformed area feature.

    Property price typically scales sub-linearly with size (diminishing
    marginal value per additional sqft), so log(area) is often a more
    linear predictor of price than raw area.

    Args:
        df: DataFrame containing area_col.
        area_col: Name of the numeric area column to transform.

    Returns:
        A copy of df with a new "log_area_sqft" column.
    """
    df = df.copy()
    df["log_area_sqft"] = np.log1p(df[area_col])
    return df


def add_luxury_indicator(
    df: pd.DataFrame,
    property_type_col: str = "property_type",
    bedrooms_col: str = "bedrooms",
    size_category_col: str = "size_category",
) -> pd.DataFrame:
    """Flag listings in the luxury segment, using structural attributes only.

    Deliberately built without any price-derived signal: a "luxury" flag
    built from a price-per-sqft threshold (as the original blueprint
    suggests) would be definitionally correlated with the target and
    would leak. Instead, luxury status is inferred from attributes a
    buyer or agent would know before ever seeing a price: penthouse
    listings, 5+ bedroom properties, and very-large villas.

    Args:
        df: DataFrame containing property_type_col, bedrooms_col, and
            size_category_col (from add_size_category).
        property_type_col: Column with the fine-grained property type.
        bedrooms_col: Column with bedroom count.
        size_category_col: Column with the size band from add_size_category.

    Returns:
        A copy of df with a new binary "luxury_indicator" column.
    """
    df = df.copy()
    is_penthouse = df[property_type_col].str.contains("penthouse", case=False, na=False)
    is_large_bedroom = df[bedrooms_col] >= 5
    is_very_large_villa = (df[size_category_col] == "very_large") & (
        df[property_type_col].str.contains("villa", case=False, na=False)
    )
    df["luxury_indicator"] = (is_penthouse | is_large_bedroom | is_very_large_villa).astype(int)
    return df


def engineer_static_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all target-independent feature engineering steps in sequence.

    Safe to call on the full dataset before any train/test split: none of
    these features reference price_usd or any dataset-wide statistic —
    each is a deterministic, row-level transformation.

    Args:
        df: Cleaned DataFrame (output of 01_data_cleaning.ipynb).

    Returns:
        A copy of df with size_category, property_age, has_floor_info,
        log_area_sqft, and luxury_indicator added.
    """
    df = add_size_category(df)
    df = add_property_age(df)
    df = add_has_floor_info(df)
    df = add_log_area(df)
    df = add_luxury_indicator(df)
    return df


class LocationTierEncoder(BaseEstimator, TransformerMixin):
    """Encode a zone column into price-based tiers, fit on training data only.

    Unlike the static features above, location tiers are inherently
    derived from historical price — grouping zones by their median price
    is the entire point of the feature (project blueprint Part 21).
    Computing that grouping from the full dataset once, before a
    train/test split, would leak target information into the feature: a
    zone's tier would then partly encode price information from rows the
    model is later evaluated against.

    Implementing this as a scikit-learn transformer lets it live inside a
    Pipeline/ColumnTransformer, where scikit-learn calls fit() on the
    training fold only and transform() on the held-out fold automatically
    — including during cross_val_score and GridSearchCV/
    RandomizedSearchCV, where a manually precomputed tier column would
    otherwise leak validation-fold price information across folds.
    """

    def __init__(self, zone_col: str = "zone", n_tiers: int = 4):
        self.zone_col = zone_col
        self.n_tiers = n_tiers

    def fit(self, X, y=None):
        """Compute the zone -> tier mapping from median price per zone.

        Args:
            X: DataFrame (or array) containing zone_col.
            y: Target values positionally aligned with X, used to compute
                median price per zone.

        Returns:
            self, with zone_tier_map_ and fallback_tier_ fitted attributes.
        """
        zone_values = X[self.zone_col] if isinstance(X, pd.DataFrame) else pd.Series(np.asarray(X)[:, 0])
        zone_median = pd.Series(np.asarray(y), index=zone_values.index).groupby(zone_values).median()

        tier_labels = [f"tier_{i + 1}" for i in range(self.n_tiers)]
        # Rank before qcut so duplicate zone-median values can't produce
        # non-unique bin edges.
        tier_assignment = pd.qcut(
            zone_median.rank(method="first"), q=self.n_tiers, labels=tier_labels
        )

        self.zone_tier_map_ = tier_assignment.to_dict()
        self.fallback_tier_ = tier_labels[len(tier_labels) // 2]
        return self

    def transform(self, X) -> pd.DataFrame:
        """Map each row's zone to its fitted price tier.

        Args:
            X: DataFrame (or array) containing zone_col.

        Returns:
            Single-column DataFrame "location_tier" with tier labels;
            zones not seen during fit fall back to the middle tier.
        """
        zone_values = X[self.zone_col] if isinstance(X, pd.DataFrame) else pd.Series(np.asarray(X)[:, 0])
        tiers = zone_values.map(self.zone_tier_map_).fillna(self.fallback_tier_)
        return tiers.to_frame(name="location_tier")

    def get_feature_names_out(self, input_features=None):
        """Return the fixed output column name, for ColumnTransformer compatibility."""
        return np.array(["location_tier"])


def save_feature_engineered_data(df: pd.DataFrame, path: Path) -> None:
    """Save a feature-engineered DataFrame to disk as CSV, without the pandas index.

    Args:
        df: DataFrame to save.
        path: Destination CSV path. Parent directory must already exist.
    """
    df.to_csv(path, index=False)
