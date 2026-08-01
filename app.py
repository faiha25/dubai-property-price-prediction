"""Streamlit demo for the Dubai Residential Property Price Prediction model.

Loads the tuned XGBoost pipeline saved by notebooks/03_modeling.ipynb
(models/best_model_xgboost.joblib) and lets a non-technical user get a
live price estimate for a hypothetical property. No retraining happens
here — this app is a thin UI wrapper around the already-trained,
already-persisted model.

The model predicts in USD (see docs/decisions.md, decision 0001: the
modeling pipeline never converts currency internally). This app converts
the USD prediction to AED for display only, using the UAE Central Bank's
official fixed peg — the underlying model and prediction stay in USD.
"""

import sys
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent))

from src.feature_engineering import engineer_static_features

# --- Configuration ---
MODEL_PATH = Path("models/best_model_xgboost.joblib")

# UAE Central Bank official USD/AED peg, fixed since 1997 — used for
# display only, never inside the model (see docs/decisions.md).
AED_PER_USD = 3.6725

# Model test-set MAE (USD), from notebooks/03_modeling.ipynb Section 9 —
# shown alongside the prediction so the estimate isn't presented as more
# precise than the model actually is.
TEST_MAE_USD = 162_554

# Columns dropped before modeling, mirroring notebooks/03_modeling.ipynb
# Section 4 EXACTLY — kept in sync manually since the fitted pipeline
# does not expose this list itself.
EXCLUDED_COLUMNS = {
    "price_usd", "price_per_sqft_usd", "price_per_m2_usd", "id",
    "date_listed", "year_built", "area_m2", "area_sqft",
    "community", "metro_station",
}

# Valid property_type options per property_category, and the bedroom
# count each type implies — confirmed 1:1 in the training data
# (data/processed/clean_secondary_sales.csv), so bedrooms is derived
# from property_type rather than asked separately, preventing
# nonsensical combinations (e.g. "studio" with 4 bedrooms) that the
# model never saw during training.
PROPERTY_TYPE_OPTIONS = {
    "apartment": ["studio", "1BR", "2BR", "3BR", "4BR_penthouse"],
    "villa": ["3BR_villa", "4BR_villa", "5BR_villa", "6BR_villa"],
}
BEDROOMS_BY_TYPE = {
    "studio": 0, "1BR": 1, "2BR": 2, "3BR": 3, "4BR_penthouse": 4,
    "3BR_villa": 3, "4BR_villa": 4, "5BR_villa": 5, "6BR_villa": 6,
}

VIEW_OPTIONS = ["community", "park", "city", "pool", "sea", "marina", "golf_course", "burj_khalifa"]
FURNISHING_OPTIONS = ["unfurnished", "semi_furnished", "fully_furnished"]
CONDITION_OPTIONS = ["vacant_on_transfer", "tenanted", "off_plan_resale"]
YEAR_BUILT_OPTIONS = [2008, 2012, 2015, 2018, 2021, 2023]  # only values seen in training

# Per-zone reference values (mean lat/lon, typical metro line/distance,
# median distance to Burj Khalifa), computed once from
# data/processed/clean_secondary_sales.csv. Hardcoded here rather than
# read from that CSV at runtime, since data/ is gitignored and this app
# must run from just the saved model file, without requiring the raw
# dataset to be present.
ZONE_REFERENCE = {
    "Al Barsha": {"lat": 25.0944, "lon": 55.2065, "metro_line": "Red", "metro_distance_type": "drive", "metro_distance_min": 22, "to_burj_khalifa_km": 13.49},
    "Al Furjan": {"lat": 25.0291, "lon": 55.1474, "metro_line": "Red", "metro_distance_type": "walk", "metro_distance_min": 3, "to_burj_khalifa_km": 22.69},
    "Al Mamzar": {"lat": 25.2944, "lon": 55.3487, "metro_line": "Green", "metro_distance_type": "drive", "metro_distance_min": 11, "to_burj_khalifa_km": 13.17},
    "Al Nahda": {"lat": 25.292, "lon": 55.3687, "metro_line": "Green", "metro_distance_type": "walk", "metro_distance_min": 6, "to_burj_khalifa_km": 14.21},
    "Al Qusais": {"lat": 25.2859, "lon": 55.3677, "metro_line": "Green", "metro_distance_type": "walk", "metro_distance_min": 12, "to_burj_khalifa_km": 13.61},
    "Al Warqa": {"lat": 25.1987, "lon": 55.3956, "metro_line": "Green", "metro_distance_type": "drive", "metro_distance_min": 50, "to_burj_khalifa_km": 12.26},
    "Al Wasl": {"lat": 25.2078, "lon": 55.2616, "metro_line": "Red", "metro_distance_type": "drive", "metro_distance_min": 20, "to_burj_khalifa_km": 2.05},
    "Arabian Ranches": {"lat": 25.0522, "lon": 55.2734, "metro_line": "Red", "metro_distance_type": "drive", "metro_distance_min": 65, "to_burj_khalifa_km": 16.2},
    "Bukadra": {"lat": 25.181, "lon": 55.3366, "metro_line": "Red", "metro_distance_type": "drive", "metro_distance_min": 39, "to_burj_khalifa_km": 6.53},
    "Bur Dubai": {"lat": 25.2512, "lon": 55.2994, "metro_line": "Green", "metro_distance_type": "walk", "metro_distance_min": 14, "to_burj_khalifa_km": 6.67},
    "Business Bay": {"lat": 25.1897, "lon": 55.2627, "metro_line": "Red", "metro_distance_type": "walk", "metro_distance_min": 6, "to_burj_khalifa_km": 1.74},
    "Creek": {"lat": 25.196, "lon": 55.351, "metro_line": "Green", "metro_distance_type": "drive", "metro_distance_min": 37, "to_burj_khalifa_km": 7.75},
    "DIFC": {"lat": 25.2101, "lon": 55.2807, "metro_line": "Red", "metro_distance_type": "walk", "metro_distance_min": 10, "to_burj_khalifa_km": 1.96},
    "DIP": {"lat": 25.0009, "lon": 55.1783, "metro_line": "Red", "metro_distance_type": "drive", "metro_distance_min": 20, "to_burj_khalifa_km": 23.91},
    "Damac": {"lat": 25.0494, "lon": 55.2634, "metro_line": "Red", "metro_distance_type": "drive", "metro_distance_min": 60, "to_burj_khalifa_km": 16.48},
    "Damac Hills": {"lat": 24.995, "lon": 55.2768, "metro_line": "Red", "metro_distance_type": "drive", "metro_distance_min": 78, "to_burj_khalifa_km": 24.28},
    "Deira": {"lat": 25.2699, "lon": 55.3101, "metro_line": "Green", "metro_distance_type": "walk", "metro_distance_min": 8, "to_burj_khalifa_km": 8.92},
    "Deira Islands": {"lat": 25.2865, "lon": 55.3113, "metro_line": "Red", "metro_distance_type": "drive", "metro_distance_min": 17, "to_burj_khalifa_km": 10.7},
    "Discovery Gardens": {"lat": 25.042, "lon": 55.1472, "metro_line": "Red", "metro_distance_type": "walk", "metro_distance_min": 11, "to_burj_khalifa_km": 21.505},
    "Downtown": {"lat": 25.197, "lon": 55.2748, "metro_line": "Red", "metro_distance_type": "walk", "metro_distance_min": 7, "to_burj_khalifa_km": 1.3},
    "Dubai Marina": {"lat": 25.0815, "lon": 55.1355, "metro_line": "Red", "metro_distance_type": "drive", "metro_distance_min": 28, "to_burj_khalifa_km": 19.04},
    "Dubai Silicon Oasis": {"lat": 25.1182, "lon": 55.3871, "metro_line": "Red", "metro_distance_type": "drive", "metro_distance_min": 76, "to_burj_khalifa_km": 14.38},
    "Dubai South": {"lat": 24.9338, "lon": 55.1556, "metro_line": "Red", "metro_distance_type": "drive", "metro_distance_min": 26, "to_burj_khalifa_km": 31.595},
    "Dubai Sports City": {"lat": 25.0426, "lon": 55.2206, "metro_line": "Red", "metro_distance_type": "drive", "metro_distance_min": 40, "to_burj_khalifa_km": 18.08},
    "Dubai Studio City": {"lat": 25.0412, "lon": 55.2248, "metro_line": "Red", "metro_distance_type": "drive", "metro_distance_min": 43, "to_burj_khalifa_km": 18.09},
    "Dubai-Al Ain Road": {"lat": 24.9647, "lon": 55.4918, "metro_line": "Red", "metro_distance_type": "drive", "metro_distance_min": 150, "to_burj_khalifa_km": 33.84},
    "Dubailand": {"lat": 25.0478, "lon": 55.2879, "metro_line": "Red", "metro_distance_type": "drive", "metro_distance_min": 73, "to_burj_khalifa_km": 16.635},
    "Emirates Hills": {"lat": 25.0499, "lon": 55.1693, "metro_line": "Red", "metro_distance_type": "drive", "metro_distance_min": 21, "to_burj_khalifa_km": 19.51},
    "Emirates Living": {"lat": 25.0531, "lon": 55.1743, "metro_line": "Red", "metro_distance_type": "drive", "metro_distance_min": 25, "to_burj_khalifa_km": 19.01},
    "Festival City": {"lat": 25.2278, "lon": 55.3521, "metro_line": "Green", "metro_distance_type": "walk", "metro_distance_min": 7, "to_burj_khalifa_km": 8.6},
    "IMPZ": {"lat": 25.0258, "lon": 55.2331, "metro_line": "Red", "metro_distance_type": "drive", "metro_distance_min": 48, "to_burj_khalifa_km": 19.5},
    "International City": {"lat": 25.1666, "lon": 55.4127, "metro_line": "Red", "metro_distance_type": "drive", "metro_distance_min": 64, "to_burj_khalifa_km": 14.33},
    "JGE": {"lat": 25.0138, "lon": 55.1851, "metro_line": "Red", "metro_distance_type": "drive", "metro_distance_min": 25, "to_burj_khalifa_km": 22.31},
    "JLT": {"lat": 25.0678, "lon": 55.1418, "metro_line": "Red", "metro_distance_type": "drive", "metro_distance_min": 23, "to_burj_khalifa_km": 19.65},
    "JVC": {"lat": 25.0537, "lon": 55.2109, "metro_line": "Red", "metro_distance_type": "drive", "metro_distance_min": 37, "to_burj_khalifa_km": 17.26},
    "JVT": {"lat": 25.0545, "lon": 55.1853, "metro_line": "Red", "metro_distance_type": "drive", "metro_distance_min": 28, "to_burj_khalifa_km": 18.23},
    "Jebel Ali": {"lat": 25.0141, "lon": 55.0977, "metro_line": "Red", "metro_distance_type": "drive", "metro_distance_min": 21, "to_burj_khalifa_km": 27.07},
    "Jumeirah": {"lat": 25.1999, "lon": 55.2353, "metro_line": "Red", "metro_distance_type": "drive", "metro_distance_min": 25, "to_burj_khalifa_km": 4.45},
    "Jumeirah Bay": {"lat": 25.2074, "lon": 55.2512, "metro_line": "Red", "metro_distance_type": "drive", "metro_distance_min": 22, "to_burj_khalifa_km": 2.82},
    "MBR City": {"lat": 25.1777, "lon": 55.2973, "metro_line": "Red", "metro_distance_type": "drive", "metro_distance_min": 32, "to_burj_khalifa_km": 3.375},
    "Maritime City": {"lat": 25.2663, "lon": 55.2569, "metro_line": "Red", "metro_distance_type": "drive", "metro_distance_min": 27, "to_burj_khalifa_km": 7.97},
    "Meydan": {"lat": 25.153, "lon": 55.2937, "metro_line": "Red", "metro_distance_type": "drive", "metro_distance_min": 44, "to_burj_khalifa_km": 5.43},
    "Mirdif": {"lat": 25.2197, "lon": 55.4188, "metro_line": "Green", "metro_distance_type": "drive", "metro_distance_min": 48, "to_burj_khalifa_km": 14.74},
    "Mohammed Bin Rashid": {"lat": 25.1393, "lon": 55.2782, "metro_line": "Red", "metro_distance_type": "drive", "metro_distance_min": 37, "to_burj_khalifa_km": 7.62},
    "Motor City": {"lat": 25.0463, "lon": 55.2385, "metro_line": "Red", "metro_distance_type": "drive", "metro_distance_min": 46, "to_burj_khalifa_km": 17.205},
    "Nshama": {"lat": 25.0086, "lon": 55.2693, "metro_line": "Red", "metro_distance_type": "drive", "metro_distance_min": 64, "to_burj_khalifa_km": 21.01},
    "Palm Jumeirah": {"lat": 25.1119, "lon": 55.139, "metro_line": "Red", "metro_distance_type": "drive", "metro_distance_min": 35, "to_burj_khalifa_km": 16.615},
    "Reem": {"lat": 25.0411, "lon": 55.2712, "metro_line": "Red", "metro_distance_type": "drive", "metro_distance_min": 64, "to_burj_khalifa_km": 17.38},
    "Sustainable City": {"lat": 24.9818, "lon": 55.2819, "metro_line": "Red", "metro_distance_type": "drive", "metro_distance_min": 74, "to_burj_khalifa_km": 23.95},
    "The Greens": {"lat": 25.1019, "lon": 55.1743, "metro_line": "Red", "metro_distance_type": "walk", "metro_distance_min": 6, "to_burj_khalifa_km": 14.7},
    "Tilal Al Ghaf": {"lat": 25.0169, "lon": 55.2602, "metro_line": "Red", "metro_distance_type": "drive", "metro_distance_min": 59, "to_burj_khalifa_km": 20.115},
    "World Islands": {"lat": 25.2239, "lon": 55.1604, "metro_line": "Red", "metro_distance_type": "drive", "metro_distance_min": 65, "to_burj_khalifa_km": 11.94},
}


@st.cache_resource
def load_model():
    """Load the saved tuned model pipeline once per app session.

    Returns:
        The fitted TransformedTargetRegressor loaded from MODEL_PATH.

    Raises:
        FileNotFoundError: if the model file doesn't exist locally —
            this app never trains a model itself.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. This app only loads an already-trained "
            "model — run notebooks/01_data_cleaning.ipynb, 02_eda.ipynb, then "
            "03_modeling.ipynb (in order) first to produce it."
        )
    return joblib.load(MODEL_PATH)


def build_input_row(
    zone: str,
    property_category: str,
    property_type: str,
    area_sqft: int,
    floor: float | None,
    total_floors: float | None,
    year_built: int,
    view: str,
    furnishing: str,
    condition: str,
    parking_spaces: int,
    is_freehold: bool,
    chiller_included: bool,
    mortgage_rate_at_listing: float,
) -> pd.DataFrame:
    """Assemble a single-row DataFrame in the exact schema the trained model expects.

    Mirrors notebooks/03_modeling.ipynb Section 12's manually-constructed
    example: raw columns in, engineer_static_features() applied, then
    subset to the model's feature_cols.

    Args:
        zone: Selected zone name (must be a key in ZONE_REFERENCE for a
            zone the model has seen; unseen zones still work, falling
            back to the model's middle location tier).
        property_category: "apartment" or "villa".
        property_type: One of PROPERTY_TYPE_OPTIONS[property_category].
        area_sqft: Property size in square feet.
        floor: Floor number, or None for villas (matches training data,
            where floor is structurally absent for villas).
        total_floors: Building height, or None for villas.
        year_built: Construction year.
        view: View category.
        furnishing: Furnishing status.
        condition: Listing condition.
        parking_spaces: Number of parking spaces.
        is_freehold: Whether the property is freehold.
        chiller_included: Whether AC chiller cost is bundled.
        mortgage_rate_at_listing: Prevailing mortgage rate, as a percentage.

    Returns:
        A one-row DataFrame with exactly the columns the saved model's
        preprocessing pipeline expects.
    """
    zone_info = ZONE_REFERENCE.get(zone)
    if zone_info is None:
        # Unseen zone: fall back to the dataset-wide mean location so the
        # row is still valid; LocationTierEncoder itself already falls
        # back to the middle price tier for zones it didn't see at fit time.
        zone_info = {"lat": 25.15, "lon": 55.28, "metro_line": "Red", "metro_distance_type": "drive", "metro_distance_min": 30, "to_burj_khalifa_km": 15.0}

    raw_row = pd.DataFrame([{
        "date_listed": pd.Timestamp(date.today()),
        "zone": zone,
        "is_freehold": int(is_freehold),
        "lat": zone_info["lat"],
        "lon": zone_info["lon"],
        "property_category": property_category,
        "property_type": property_type,
        "bedrooms": BEDROOMS_BY_TYPE[property_type],
        "area_sqft": area_sqft,
        "floor": floor,
        "total_floors": total_floors,
        "year_built": year_built,
        "view": view,
        "furnishing": furnishing,
        "condition": condition,
        "parking_spaces": parking_spaces,
        "chiller_included": int(chiller_included),
        "metro_line": zone_info["metro_line"],
        "metro_distance_min": zone_info["metro_distance_min"],
        "metro_distance_type": zone_info["metro_distance_type"],
        "to_burj_khalifa_km": zone_info["to_burj_khalifa_km"],
        "mortgage_rate_at_listing": mortgage_rate_at_listing,
    }])

    # floor/total_floors must be float64 to match the dtype the fitted
    # SimpleImputer(strategy="constant", fill_value=0) was fit on during
    # training (that column is float64 there because NaN — for villas —
    # forces the whole column to float). Without this cast, a plain
    # Python int (e.g. floor=22) makes a fresh single-row DataFrame infer
    # int64 instead, and the fitted imputer rejects the dtype mismatch.
    raw_row = raw_row.astype({"floor": "float64", "total_floors": "float64"})

    # Same static feature engineering used at training time (Section 3 of
    # notebooks/03_modeling.ipynb) — size_category, property_age,
    # has_floor_info, log_area_sqft, luxury_indicator.
    engineered = engineer_static_features(raw_row)

    feature_cols = [c for c in engineered.columns if c not in EXCLUDED_COLUMNS]
    return engineered[feature_cols]


def format_price(amount: float) -> str:
    """Format a price for display: rounded to the nearest 1,000, comma-separated.

    Rounding to the nearest 1,000 avoids implying false precision — the
    model's test-set MAE is roughly $162,554, so displaying exact-dollar
    predictions would overstate how precise any single estimate is.

    Args:
        amount: Raw predicted price.

    Returns:
        A formatted string like "1,850,000".
    """
    return f"{round(amount / 1_000) * 1_000:,.0f}"


# ============================== UI ============================== #

st.set_page_config(page_title="Dubai Property Price Predictor", page_icon="🏠", layout="centered")

st.title("Dubai Residential Property Price Predictor")
st.caption("Machine learning model estimating property prices using XGBoost trained on Dubai listings")

st.info(
    "This is a portfolio/educational project trained on a Kaggle dataset that shows signs of being "
    "synthetically generated (see the README). Predictions illustrate the trained model, not a "
    "professional property valuation.",
    icon="ℹ️",
)

# Load the model once; show a clear, actionable error if it's missing
# rather than letting the app crash with a raw traceback.
try:
    model = load_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

with st.form("property_form"):
    st.subheader("Property Details")

    col1, col2 = st.columns(2)
    with col1:
        zone = st.selectbox("Location (zone)", sorted(ZONE_REFERENCE.keys()), index=sorted(ZONE_REFERENCE.keys()).index("Dubai Marina"))
        property_category = st.selectbox("Property category", ["apartment", "villa"])
        property_type = st.selectbox("Property type", PROPERTY_TYPE_OPTIONS[property_category])
        area_sqft = st.number_input("Area (sqft)", min_value=300, max_value=15000, value=1200, step=50)
        year_built = st.selectbox("Year built", YEAR_BUILT_OPTIONS, index=3)
        view = st.selectbox("View", VIEW_OPTIONS)

    with col2:
        furnishing = st.selectbox("Furnishing status", FURNISHING_OPTIONS)
        condition = st.selectbox("Condition", CONDITION_OPTIONS)
        parking_spaces = st.slider("Parking spaces", min_value=1, max_value=4, value=1)
        is_freehold = st.checkbox("Freehold ownership", value=True)
        chiller_included = st.checkbox("AC chiller included", value=True)
        mortgage_rate_at_listing = st.slider("Prevailing mortgage rate (%)", min_value=1.9, max_value=6.9, value=5.5, step=0.05)

    # Floor / total_floors only apply to apartments — mirrors the
    # structural missingness documented in docs/data_cleaning_report.md
    # (villas have no floor number in the training data at all).
    if property_category == "apartment":
        col3, col4 = st.columns(2)
        with col3:
            total_floors = st.number_input("Building height (total floors)", min_value=4, max_value=90, value=30)
        with col4:
            floor = st.number_input("Floor number", min_value=1, max_value=int(total_floors), value=min(10, int(total_floors)))
    else:
        floor, total_floors = None, None
        st.caption("Floor number and building height don't apply to villas — this dataset has no floor "
                   "field for standalone houses, matching real-world Dubai villa listings.")

    st.caption(
        "Note: this dataset has no bathroom-count field, so bathrooms are not part of the model's "
        "inputs — bedroom count (derived from property type above) and area are used as size proxies "
        "instead. See docs/model_card.md for the full list of limitations."
    )

    submitted = st.form_submit_button("Estimate Price")

if submitted:
    try:
        input_row = build_input_row(
            zone=zone,
            property_category=property_category,
            property_type=property_type,
            area_sqft=area_sqft,
            floor=floor,
            total_floors=total_floors,
            year_built=year_built,
            view=view,
            furnishing=furnishing,
            condition=condition,
            parking_spaces=parking_spaces,
            is_freehold=is_freehold,
            chiller_included=chiller_included,
            mortgage_rate_at_listing=mortgage_rate_at_listing,
        )
        predicted_usd = float(model.predict(input_row)[0])
        predicted_aed = predicted_usd * AED_PER_USD

        st.success(f"Estimated Property Price: AED {format_price(predicted_aed)}")
        st.caption(f"Equivalent to USD {format_price(predicted_usd)} (the model's native currency — "
                   f"AED shown at the fixed peg of {AED_PER_USD} AED/USD, for display only)")
        st.caption(
            f"Typical model error on held-out test data: ± USD {TEST_MAE_USD:,} (MAE) — "
            "treat this as an estimate range, not an exact figure."
        )
    except Exception as e:
        # Broad catch is intentional here: this is the final user-facing
        # step, and any failure (bad input combination, a stale/corrupt
        # model file, etc.) should surface as a readable message rather
        # than a raw Streamlit traceback.
        st.error(f"Could not generate a prediction: {e}")

st.divider()
st.caption(
    "Model: tuned XGBoost regressor, test R² = 0.888 (see README.md and docs/model_card.md). "
    "Not a substitute for professional property valuation."
)
