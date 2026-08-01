"""Reusable data-cleaning functions for the Dubai secondary sales dataset.

These functions are intentionally conservative: the source dataset is
already well-structured (no duplicates, no whitespace/casing issues,
internally consistent derived columns), so most functions here validate
and document rather than transform. Structural missingness (floor /
total_floors for villas) is preserved, not imputed, and statistical
outliers are reported, not removed, unless a genuine data-quality defect
is found.
"""

from pathlib import Path

import pandas as pd


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load a raw CSV listing file from disk.

    Args:
        path: Path to the raw CSV file.

    Returns:
        The loaded DataFrame, unmodified.
    """
    return pd.read_csv(path)


def check_duplicate_rows(df: pd.DataFrame, id_column: str = "id") -> dict:
    """Check for exact duplicate rows and duplicate IDs.

    Args:
        df: DataFrame to check.
        id_column: Name of the unique-identifier column.

    Returns:
        Dict with counts of duplicate rows and duplicate IDs, and whether
        the ID column is fully unique.
    """
    return {
        "n_duplicate_rows": int(df.duplicated().sum()),
        "n_duplicate_ids": int(df[id_column].duplicated().sum()),
        "id_is_unique": bool(df[id_column].is_unique),
    }


def fix_data_types(df: pd.DataFrame, date_columns: list[str]) -> pd.DataFrame:
    """Convert specified string columns to pandas datetime.

    Args:
        df: DataFrame to correct.
        date_columns: Column names to parse as dates.

    Returns:
        A copy of df with the specified columns converted to datetime64.
    """
    df = df.copy()
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])
    return df


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize missing values per column.

    Args:
        df: DataFrame to inspect.

    Returns:
        DataFrame indexed by column name with missing_count and
        missing_pct, sorted descending by missing_count. Columns with
        zero missing values are still included.
    """
    missing_count = df.isna().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)
    report = pd.DataFrame(
        {"missing_count": missing_count, "missing_pct": missing_pct}
    )
    return report.sort_values("missing_count", ascending=False)


def check_structural_missingness(
    df: pd.DataFrame, value_columns: list[str], group_column: str
) -> pd.DataFrame:
    """Check whether missingness in value_columns is fully explained by a category.

    Args:
        df: DataFrame to inspect.
        value_columns: Columns whose missingness pattern is being checked.
        group_column: Categorical column hypothesized to explain the pattern.

    Returns:
        DataFrame of missing-rate (0-1) per value_column, grouped by
        group_column. A rate of 1.0 for one category and 0.0 for another
        confirms the missingness is structural rather than random.
    """
    return df.groupby(group_column)[value_columns].apply(lambda g: g.isna().mean())


def validate_string_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Check string columns for whitespace issues and list their categories.

    Args:
        df: DataFrame to inspect.
        columns: String/categorical column names to validate.

    Returns:
        DataFrame indexed by column name with n_unique, has_whitespace
        (count of values differing from their stripped form), and the
        sorted list of unique categories for manual review.
    """
    rows = []
    for col in columns:
        values = df[col].astype(str)
        whitespace_issues = int((values != values.str.strip()).sum())
        rows.append(
            {
                "column": col,
                "n_unique": df[col].nunique(),
                "has_whitespace": whitespace_issues,
                "categories": sorted(df[col].unique().tolist()),
            }
        )
    return pd.DataFrame(rows).set_index("column")


def check_floor_consistency(df: pd.DataFrame) -> int:
    """Count rows where floor exceeds total_floors, among rows with both values.

    Args:
        df: DataFrame containing floor and total_floors columns.

    Returns:
        Count of rows violating floor <= total_floors.
    """
    valid = df.dropna(subset=["floor", "total_floors"])
    return int((valid["floor"] > valid["total_floors"]).sum())


def check_price_per_sqft_consistency(
    df: pd.DataFrame, tolerance: float = 1.0
) -> dict:
    """Verify price_per_sqft_usd matches price_usd / area_sqft within tolerance.

    Args:
        df: DataFrame containing price_usd, area_sqft, price_per_sqft_usd.
        tolerance: Maximum allowed absolute difference before flagging a row.

    Returns:
        Dict with the max observed difference and the count of rows that
        exceed tolerance.
    """
    implied = df["price_usd"] / df["area_sqft"]
    diff = (implied - df["price_per_sqft_usd"]).abs()
    return {
        "max_abs_diff": float(diff.max()),
        "n_rows_exceeding_tolerance": int((diff > tolerance).sum()),
    }


def check_area_unit_consistency(
    df: pd.DataFrame, sqft_to_m2: float = 0.092903, tolerance: float = 0.5
) -> dict:
    """Verify area_m2 matches area_sqft converted to square metres.

    Args:
        df: DataFrame containing area_sqft and area_m2.
        sqft_to_m2: Conversion factor from square feet to square metres.
        tolerance: Maximum allowed absolute difference before flagging a row.

    Returns:
        Dict with the max observed difference and the count of rows that
        exceed tolerance.
    """
    implied = df["area_sqft"] * sqft_to_m2
    diff = (implied - df["area_m2"]).abs()
    return {
        "max_abs_diff": float(diff.max()),
        "n_rows_exceeding_tolerance": int((diff > tolerance).sum()),
    }


def save_clean_data(df: pd.DataFrame, path: Path) -> None:
    """Save a cleaned DataFrame to disk as CSV, without the pandas index.

    Args:
        df: DataFrame to save.
        path: Destination CSV path. Parent directory must already exist.
    """
    df.to_csv(path, index=False)
