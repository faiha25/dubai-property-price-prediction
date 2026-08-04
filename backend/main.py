"""FastAPI backend serving the trained XGBoost Dubai property price model.

Thin API wrapper — no training happens here. Loads the same fitted
pipeline app.py (the Streamlit demo) uses, models/best_model_xgboost.joblib,
and exposes it over HTTP for a future frontend to call.

Run from the backend/ directory:
    uvicorn main:app --reload --port 8000
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import model as model_module
from model import TEST_MAE_USD, ZONE_REFERENCE
from schemas import HealthResponse, PredictionRequest, PredictionResponse

app = FastAPI(
    title="Dubai Property Price Prediction API",
    description="Serves predictions from the trained XGBoost pipeline (models/best_model_xgboost.joblib).",
    version="0.1.0",
)

# Wide open for local development with a future frontend on a different
# port (e.g. Vite's default localhost:5173). Tighten this to a specific
# origin before deploying either side publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report whether the trained model file is present and loadable."""
    try:
        model_module.get_model()
        model_loaded = True
    except FileNotFoundError:
        model_loaded = False
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        known_zones=len(ZONE_REFERENCE),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict a property's price from its characteristics.

    Args:
        request: Property attributes, validated against the same schema
            the trained pipeline was built on (see schemas.py).

    Returns:
        The predicted price in USD (the model's native currency) and its
        AED equivalent at the fixed peg, plus the model's test-set MAE
        for context on estimate precision.
    """
    try:
        predicted_usd, predicted_aed = model_module.predict(request)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not generate a prediction: {e}") from e

    return PredictionResponse(
        predicted_price_usd=predicted_usd,
        predicted_price_aed=predicted_aed,
        test_mae_usd=TEST_MAE_USD,
    )
