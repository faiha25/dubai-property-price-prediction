# Data Cleaning Report

**Project:** Dubai Residential Property Price Prediction
**Phase:** Data Cleaning
**Source notebook:** [`notebooks/01_data_cleaning.ipynb`](../notebooks/01_data_cleaning.ipynb)
**Source module:** [`src/preprocessing.py`](../src/preprocessing.py)
**Input:** `data/raw/secondary_sales.csv` (50,000 rows x 29 columns)
**Output:** `data/processed/clean_secondary_sales.csv` (50,000 rows x 29 columns)
**Date:** 2026-08-01

## 1. Scope

This phase covers **data cleaning only**: duplicate validation, datatype
correction, missing-value review, string validation, business-rule
consistency checks, and outlier assessment. It deliberately does not
include feature engineering, feature selection, encoding, scaling,
train/test splitting, or model training — those belong to later phases
(`02_eda.ipynb` and `03_modeling.ipynb`).

Of the five files shipped in the source Kaggle dataset
(`sergionefedov/dubai-real-estate-sales-and-rentals-2020-2026`), only
`secondary_sales.csv` is used as the primary modeling dataset for this
first model version. `rentals.csv` and `off_plan.csv` are out of scope.
`area_prices_monthly.csv` and `metro_stations.csv` are reserved as
potential future enrichment sources only and are not touched in this
phase.

## 2. Dataset Overview

| | |
|---|---|
| Rows | 50,000 |
| Columns | 29 |
| Target variable | `price_usd` (continuous, USD) |
| Unique key | `id` (confirmed 100% unique) |
| Date range | `date_listed`: 2020-01-01 to 2026-04-29 |

## 3. Transformations Performed

| Column | Change | Justification |
|---|---|---|
| `date_listed` | `object` (string) → `datetime64[ns]` | Enables correct chronological sorting/filtering in later phases; the string form cannot be compared or bucketed by date reliably. |

**No other column was modified.** This is stated explicitly because it is
easy to assume a cleaning phase must transform many columns — here, the
dataset was already well-formed, and modifying more than what the
evidence required would be an unjustified transformation.

## 4. Transformations Intentionally NOT Performed

| Candidate action | Why it was rejected |
|---|---|
| Impute `floor` / `total_floors` with 0, mean, or median | Missingness is 100% explained by `property_category == 'villa'` (villas structurally have no floor number — the concept doesn't apply). Imputing would fabricate a false floor value for every villa in the dataset. See Section 6. |
| Remove duplicate rows | None exist (0 duplicate rows, 0 duplicate IDs). Nothing to remove. |
| Standardize string casing/whitespace | None found. All 10 categorical columns checked clean. |
| Remove or cap high-price / high-area outliers | Investigated and found to be genuine ultra-luxury market segments (e.g. Bulgari Resort, Jumeirah Bay Island) with internally consistent price-per-sqft values, not data-entry errors. See Section 8. |
| Correct `price_per_sqft_usd` / `area_m2` for rounding differences | Differences are sub-$4 and sub-0.1 m² — expected floating-point/integer rounding, not genuine inconsistencies. See Section 7. |
| Drop `price_per_sqft_usd` / `price_per_m2_usd` (target-leakage columns) | Retained in the cleaned dataset for documentation and future evaluation benchmarking. They must be **excluded from the model's feature set** at feature-selection time in `03_modeling.ipynb` — that exclusion is a modeling-phase decision, not a cleaning-phase one. |
| Convert `price_usd` to AED | Source data is natively in USD; converting introduces an FX-rate assumption that belongs to a documented, deliberate decision rather than an implicit cleaning step. Deferred pending explicit sign-off. |

## 5. Duplicate Validation

| Check | Result |
|---|---|
| Exact duplicate rows | 0 |
| Duplicate `id` values | 0 |
| `id` uniqueness | 100% unique |

No rows removed.

## 6. Missing Value Review

Only two columns contain missing values:

| Column | Missing count | Missing % |
|---|---|---|
| `floor` | 17,820 | 35.64% |
| `total_floors` | 17,820 | 35.64% |
| all other columns | 0 | 0.00% |

**Root cause analysis:** grouping missingness by `property_category`
shows a missing rate of **1.0 for every `villa` row and 0.0 for every
`apartment` row**, for both columns. This is not random — it is
structural missingness (MNAR by design): a villa is a standalone house
and does not have a "floor number" in the way an apartment unit does.

**Decision: preserve as `NaN`. No imputation applied.**

Rationale:
- Filling with `0` would falsely encode every villa as ground-floor,
  creating a spurious pattern a model could learn as signal.
- Filling with the apartment-derived mean/median would inject
  apartment-market floor levels into villa rows — information that does
  not exist for those properties.
- `property_category` already exists as a column and is sufficient for
  a downstream model (tree-based models handle `NaN` natively; a
  `has_floor_info` indicator can be engineered later if needed). Both
  are feature-engineering decisions explicitly deferred to
  `03_modeling.ipynb`.

## 7. Consistency Checks

| Rule | Result |
|---|---|
| `floor <= total_floors` (rows with both values) | 0 violations / 32,180 checked |
| `price_per_sqft_usd == price_usd / area_sqft` (tolerance $1) | max diff $3.67; 278/50,000 rows (0.56%) exceed tolerance |
| `area_m2` == `area_sqft` converted to m² (tolerance 0.5 m²) | max diff 0.096 m²; 0 rows exceed tolerance |

The `price_per_sqft_usd` deviations are explained by that column being
stored as an integer (rounded), so rounding error on a per-unit value
gets amplified when multiplied back out over large `area_sqft` values.
Maximum deviation is under $4 on prices ranging up to $24.5M — this is
rounding noise, not a data-quality defect. **No correction applied to
any column as a result of these checks.**

## 8. Outlier Assessment

Distributions of `price_usd`, `area_sqft`, and `price_per_sqft_usd` were
visualized (histogram + KDE + boxplot) in the notebook.

| Variable | Skew | Mean | Median | Max |
|---|---|---|---|---|
| `price_usd` | 3.69 | $1,014,038 | $593,700 | $24,501,200 |
| `area_sqft` | 1.20 | 2,244 | 1,516 | 13,009 |
| `price_per_sqft_usd` | 2.89 | $446 | $325 | $4,967 |

All three are right-skewed, as expected for real-estate price and size
data. The top price outliers cluster in `Bulgari Resort` and `Jumeirah
Bay Island` — known ultra-luxury developments — with price-per-sqft
values internally consistent with that market segment rather than
disconnected from `area_sqft` (which is the signature of a data-entry
error). The lowest-priced listings are small units in `International
City` and `Dubai South`, Dubai's most affordable, high-supply areas.

**Decision: retain all rows, including extremes.** No statistical
outlier removal was performed. Whether to log-transform `price_usd` for
modeling is a `03_modeling.ipynb` decision, not a cleaning decision —
removing legitimate luxury or budget listings here would bias the
dataset before any modeling choice has been made.

## 9. String / Categorical Validation

All 10 categorical columns (`community`, `zone`, `property_category`,
`property_type`, `view`, `furnishing`, `condition`, `metro_station`,
`metro_line`, `metro_distance_type`) were checked for leading/trailing
whitespace and reviewed for casing consistency and unexpected
categories.

**Result: no issues found.** Zero whitespace problems across all
columns; all category values are consistent `snake_case` with no
duplicate variants (e.g. no `Unfurnished` vs. `unfurnished`) and no
garbage/unexpected values. This is stated explicitly rather than
silently skipped.

## 10. Summary

| Metric | Value |
|---|---|
| Rows in raw file | 50,000 |
| Rows in clean file | 50,000 |
| Rows removed | 0 |
| Columns modified | 1 (`date_listed`, dtype only) |
| Columns unchanged | 28 |
| New columns created | 0 |

No rows were removed at any point in this phase. Every check performed
confirmed the dataset was already internally consistent; the sole
missingness pattern found was structural and was preserved rather than
imputed.

## 11. Assumptions

- `price_usd` represents the **listed** price, not a confirmed
  transaction price. Listed and transacted prices may differ.
- `price_usd` is in USD, as delivered by the source dataset; no currency
  conversion has been applied.
- Structural missingness in `floor`/`total_floors` reflects the absence
  of the concept for villas, not a data-collection failure — this is an
  inference from the data (100% alignment with `property_category`),
  not a stated fact from the data source.
- The dataset appears to be **synthetically generated** rather than
  scraped from a live listings site: zero duplicates, zero string
  formatting issues, perfectly deterministic derived columns
  (`price_per_sqft_usd`, `area_m2`) up to rounding, and only 6 discrete
  `year_built` values across 50,000 rows. This is stated explicitly so
  that downstream readers do not mistake this cleanliness for evidence
  of real-world data-collection rigor.

## 12. Limitations

- Because the dataset is likely synthetic, cleaning decisions validated
  here (e.g. "no whitespace issues") may not generalize to a future
  real-world data refresh from Bayut/PropertyFinder-style sources, which
  would likely require the string-cleaning and deduplication logic this
  notebook currently has no need to exercise.
- `floor`/`total_floors` missingness being cleanly explained by
  `property_category` is convenient for this dataset; a real-world
  dataset might have partial or noisier missingness that would need a
  different, more careful treatment.
- No cross-validation against an external source (e.g. Dubai Land
  Department records) was performed to confirm `price_usd` values are
  plausible at the individual-listing level; only internal consistency
  (e.g. `price_per_sqft_usd` derivation) was checked.

## 13. Reproducibility

- All cleaning logic lives in reusable, type-hinted, docstringed
  functions in `src/preprocessing.py` — the notebook imports and calls
  these rather than duplicating logic inline.
- `notebooks/01_data_cleaning.ipynb` was executed top-to-bottom via
  `jupyter nbconvert --execute` after a fresh kernel start; zero errors
  were produced.
- No random processes are involved in this phase (`RANDOM_STATE = 42` is
  defined in the notebook for consistency with the rest of the pipeline
  but is unused here).
- File paths are relative (`Path("../data/raw/secondary_sales.csv")`
  from the `notebooks/` directory), not hardcoded to any local machine.
- Re-running the notebook against the same raw file will deterministically
  reproduce `data/processed/clean_secondary_sales.csv` byte-for-byte
  (no randomness, no external calls).
