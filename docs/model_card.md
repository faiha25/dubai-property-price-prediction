# Model Card: Dubai Property Price Prediction Model

**Model file:** `models/best_model_xgboost.joblib` (not committed to git — produced by running `notebooks/03_modeling.ipynb`)
**Date:** 2026-08-01

## 1. Model Overview

**Purpose:** estimate the listed sale price (USD) of a residential property in Dubai's secondary (resale) market, from its physical characteristics, location, and listing attributes.

**Prediction target:** `price_usd` — a continuous value, the listed sale price in USD.

**Intended users:** this is a portfolio/educational project. Within that scope, the intended use is illustrative — a buyer, seller, or agent exploring what a machine learning model estimates for a given set of property attributes, and a technical reviewer (e.g. a recruiter) evaluating the underlying ML engineering. It is **not** intended as a substitute for a professional property valuation — see [Ethical Considerations](#7-ethical-considerations).

## 2. Model Details

- **Algorithm:** XGBoost gradient-boosted tree regressor (`XGBRegressor`), tuned via `RandomizedSearchCV` (30 parameter combinations, 5-fold cross-validation).
- **Framework:** scikit-learn `Pipeline` / `ColumnTransformer` for preprocessing, wrapped in a `TransformedTargetRegressor` that trains on `log1p(price_usd)` and returns predictions already inverse-transformed to USD.
- **Training dataset size:** 50,000 listings total; 40,000 used for training (80%), 10,000 held out as the test set (20%), `RANDOM_STATE = 42`.
- **Tuned hyperparameters:** `n_estimators=500`, `max_depth=5`, `learning_rate=0.05`, `subsample=0.6`, `colsample_bytree=0.8`, `min_child_weight=5`.
- **Important features:** per the model's gain-based feature importance (`notebooks/03_modeling.ipynb`, Section 11), the top predictors are `bedrooms` (0.271), `location_tier_tier_4` (0.149, the highest price tier), `size_category_very_large` (0.099), `log_area_sqft` (0.089), and `location_tier_tier_3` (0.073). A separate SHAP analysis (`src/explain.py`, `images/shap_summary.png`) ranks `log_area_sqft` above `bedrooms` — the two importance measures disagree on the #1 feature, which is expected (gain-based importance reflects average loss reduction per split; SHAP reflects typical per-prediction impact) and is reported here rather than picking whichever ranking looks cleaner.

## 3. Performance

**Best model: XGBoost (tuned).**

| Metric | Value |
|---|---|
| Test R² | 0.888 |
| Test RMSE | $403,817 |
| Test MAE | $162,554 |
| Train R² | 0.913 |
| 5-fold CV R² | 0.878 ± 0.008 |

For context, a naive baseline (always predicting the training-set mean price) scores Test R² ≈ 0.000 and Test MAE = $809,911 — the tuned model reduces MAE by roughly 80% relative to that floor. Full model-by-model comparison (Linear Regression, Random Forest, untuned XGBoost) is in `README.md`, "Model Results". All figures above are taken directly from the executed `notebooks/03_modeling.ipynb`; none are estimated or invented for this document.

## 4. Training Data

- **Source:** [Kaggle — Dubai Real Estate: Sales & Rentals (2020–2026)](https://www.kaggle.com/datasets/sergionefedov/dubai-real-estate-sales-and-rentals-20202026), file `secondary_sales.csv` only (Apache-2.0 licensed).
- **Type of listings:** secondary-market (resale) residential listings — apartments and villas — in Dubai, dated 2020-01-01 through 2026-04-29. Off-plan sales and rental listings (present in the same source dataset as separate files) are explicitly out of scope for this model.
- **Target variable:** `price_usd`, the **listed** price — not a confirmed transaction price (see [Limitations](#6-limitations)).
- The dataset shows signs of being synthetically generated rather than scraped from a live listings site (zero duplicate rows, zero string-formatting inconsistencies, perfectly consistent derived columns, only 6 discrete `year_built` values); see `docs/data_cleaning_report.md` for the full evidence trail.

## 5. Features Used

24 features, engineered in `src/feature_engineering.py` and selected in `notebooks/03_modeling.ipynb`:

- **Location:** `zone` (via the leakage-safe `LocationTierEncoder`, fit on training data only), `lat`, `lon`, `to_burj_khalifa_km`, `metro_line`, `metro_distance_min`, `metro_distance_type`.
- **Size/type:** `log_area_sqft`, `size_category`, `bedrooms`, `property_type`, `property_category`, `luxury_indicator`.
- **Building:** `floor`, `total_floors`, `has_floor_info` (villas structurally have no floor number), `property_age`, `parking_spaces`.
- **Listing attributes:** `furnishing`, `condition`, `view`, `is_freehold`, `chiller_included`.
- **Market context:** `mortgage_rate_at_listing`.

**Explicitly excluded (leakage):** `price_per_sqft_usd` and `price_per_m2_usd` are deterministically derived from `price_usd` and are never used as model inputs.

## 6. Limitations

- **Listed price vs. transaction price:** the target is the listed sale price, not a confirmed sale/transaction price. Actual transaction prices can differ from listings, and this project has no way to measure that gap.
- **No building-level information:** there is no `developer`, `building name`, or building-quality field in this dataset. Two identically-specified units in different buildings (or by different developers) in the same zone are treated identically by this model, even though real buyer behavior differentiates them.
- **Market changes over time:** the model is trained on listings spanning 2020–2026 and does not explicitly model temporal price trends beyond what `mortgage_rate_at_listing` and `property_age` capture. A market shift after the training data's range would not be reflected in predictions.
- **Geographic limitations:** location is captured via a 4-tier `zone`-based encoding plus straight-line distance to one landmark (Burj Khalifa). This does not capture street-level or building-level location variation, nor proximity to other landmarks, schools, or beaches.
- **No `bathrooms` field:** this dataset has no bathroom-count column at all; bedroom count and area are used as size proxies instead.
- **Dataset authenticity:** given the evidence this dataset is likely synthetic (Section 4), model behavior — including which features matter most — describes this dataset's internal structure, not a validated claim about the real Dubai property market.
- **Error grows for high-end properties:** prediction error is largest for the most expensive listings (per the actual-vs-predicted plot in `notebooks/03_modeling.ipynb`, Section 11) — precisely where a wrong estimate matters most financially.

## 7. Ethical Considerations

- **This model should support decision-making, not replace it.** A predicted price is a data-driven estimate based on historical listings, not an appraisal. It should inform a conversation with a buyer, seller, or licensed valuer, not substitute for one.
- **It is not a substitute for professional valuation.** Real property valuation accounts for factors this model cannot see — physical condition on inspection, legal/title issues, building management quality, and current market sentiment — none of which exist in this dataset.
- **Possible bias from historical listings:** the model can only reproduce patterns present in its training data. If certain zones, property types, or price segments are over- or under-represented in the source dataset (as documented in `docs/data_cleaning_report.md` and `notebooks/02_eda.ipynb`), predictions for underrepresented segments (e.g. ultra-luxury properties, of which there are few examples) will be less reliable, and this model should not be treated as an unbiased arbiter of "fair" price across all segments equally.
- **Synthetic-data caveat repeated deliberately:** because this dataset is likely synthetic, any real-world business decision should not rely on this model without validation against real transaction data first.

## 8. Future Improvements

- Retrain on real Dubai Land Department transaction records instead of listed prices, to close the listed-vs-transacted-price gap.
- Add building-level and developer-level features if a data source providing them becomes available.
- Add geospatial features beyond `to_burj_khalifa_km` (e.g. distance to multiple landmarks, or unsupervised location clustering from `lat`/`lon`).
- Explore ensembling/stacking Linear Regression, Random Forest, and XGBoost predictions.
- Add prediction intervals (e.g. via quantile regression or Random Forest's variance estimates) instead of a single point estimate, so users see an uncertainty range rather than one number.
