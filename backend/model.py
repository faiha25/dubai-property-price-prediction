"""Model loading and inference for the FastAPI backend.

Loads the same tuned XGBoost pipeline used by app.py (Streamlit demo) —
models/best_model_xgboost.joblib — and rebuilds a single input row in the
exact schema that fitted pipeline expects. No training happens here.

This intentionally duplicates constants and logic from app.py
(ZONE_REFERENCE, BEDROOMS_BY_TYPE, PROPERTY_TYPE_OPTIONS,
EXCLUDED_COLUMNS, build_input_row) rather than importing app.py directly:
app.py executes Streamlit UI calls (st.set_page_config, st.form, ...) at
module import time, so importing it from a non-Streamlit process would
try to run that UI code outside of a Streamlit runtime and fail. Both
copies must stay in sync with notebooks/03_modeling.ipynb Section 4/12
manually, same as app.py's own comments already note.
"""

import sys
from datetime import date
from pathlib import Path

import joblib
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.feature_engineering import engineer_static_features

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model_xgboost.joblib"

# UAE Central Bank official USD/AED peg, fixed since 1997 — used for
# display only, never inside the model (see docs/decisions.md).
AED_PER_USD = 3.6725

# Model test-set MAE (USD), from notebooks/03_modeling.ipynb Section 9.
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
# count each type implies — confirmed 1:1 in the training data, so
# bedrooms is derived from property_type rather than accepted as a
# separate field.
PROPERTY_TYPE_OPTIONS = {
    "apartment": ["studio", "1BR", "2BR", "3BR", "4BR_penthouse"],
    "villa": ["3BR_villa", "4BR_villa", "5BR_villa", "6BR_villa"],
}
BEDROOMS_BY_TYPE = {
    "studio": 0, "1BR": 1, "2BR": 2, "3BR": 3, "4BR_penthouse": 4,
    "3BR_villa": 3, "4BR_villa": 4, "5BR_villa": 5, "6BR_villa": 6,
}

# Per-zone reference values (mean lat/lon, typical metro line/distance,
# median distance to Burj Khalifa), computed once from
# data/processed/clean_secondary_sales.csv. Hardcoded here rather than
# read from that CSV at runtime, since data/ is gitignored and this
# backend must run from just the saved model file, without requiring the
# raw dataset to be present.
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

_model = None


def get_model():
    """Load and cache the saved tuned model pipeline (loaded once per process).

    Returns:
        The fitted TransformedTargetRegressor loaded from MODEL_PATH.

    Raises:
        FileNotFoundError: if the model file doesn't exist locally — this
            backend never trains a model itself.
    """
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"{MODEL_PATH} not found. This backend only loads an already-trained "
                "model — run notebooks/01_data_cleaning.ipynb, 02_eda.ipynb, then "
                "03_modeling.ipynb (in order) first to produce it."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def build_input_row(request) -> pd.DataFrame:
    """Assemble a single-row DataFrame in the exact schema the trained model expects.

    Mirrors app.py's build_input_row() / notebooks/03_modeling.ipynb
    Section 12's manually-constructed example: raw columns in,
    engineer_static_features() applied, then subset to the model's
    feature_cols.

    Args:
        request: A validated schemas.PredictionRequest.

    Returns:
        A one-row DataFrame with exactly the columns the saved model's
        preprocessing pipeline expects.
    """
    zone_info = ZONE_REFERENCE.get(request.zone)
    if zone_info is None:
        # Unseen zone: fall back to the dataset-wide mean location so the
        # row is still valid; LocationTierEncoder itself already falls
        # back to the middle price tier for zones it didn't see at fit time.
        zone_info = {"lat": 25.15, "lon": 55.28, "metro_line": "Red", "metro_distance_type": "drive", "metro_distance_min": 30, "to_burj_khalifa_km": 15.0}

    raw_row = pd.DataFrame([{
        "date_listed": pd.Timestamp(date.today()),
        "zone": request.zone,
        "is_freehold": int(request.is_freehold),
        "lat": zone_info["lat"],
        "lon": zone_info["lon"],
        "property_category": request.property_category,
        "property_type": request.property_type,
        "bedrooms": BEDROOMS_BY_TYPE[request.property_type],
        "area_sqft": request.area_sqft,
        "floor": request.floor,
        "total_floors": request.total_floors,
        "year_built": request.year_built,
        "view": request.view,
        "furnishing": request.furnishing,
        "condition": request.condition,
        "parking_spaces": request.parking_spaces,
        "chiller_included": int(request.chiller_included),
        "metro_line": zone_info["metro_line"],
        "metro_distance_min": zone_info["metro_distance_min"],
        "metro_distance_type": zone_info["metro_distance_type"],
        "to_burj_khalifa_km": zone_info["to_burj_khalifa_km"],
        "mortgage_rate_at_listing": request.mortgage_rate_at_listing,
    }])

    # floor/total_floors must be float64 to match the dtype the fitted
    # SimpleImputer(strategy="constant", fill_value=0) was fit on during
    # training — see app.py's build_input_row for the full explanation.
    raw_row = raw_row.astype({"floor": "float64", "total_floors": "float64"})

    engineered = engineer_static_features(raw_row)

    feature_cols = [c for c in engineered.columns if c not in EXCLUDED_COLUMNS]
    return engineered[feature_cols]


def predict(request) -> tuple[float, float]:
    """Run inference for one prediction request.

    Args:
        request: A validated schemas.PredictionRequest.

    Returns:
        Tuple of (predicted_price_usd, predicted_price_aed).
    """
    model = get_model()
    input_row = build_input_row(request)
    predicted_usd = float(model.predict(input_row)[0])
    predicted_aed = predicted_usd * AED_PER_USD
    return predicted_usd, predicted_aed
