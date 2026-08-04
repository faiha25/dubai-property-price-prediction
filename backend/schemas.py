"""Pydantic request/response models for the prediction API.

Field constraints mirror the valid input space the trained pipeline was
built against (see notebooks/03_modeling.ipynb and app.py): the same
zones, property types, and categorical options, plus the same
apartment/villa floor-field rule enforced by app.py's form (floor and
total_floors only apply to apartments).
"""

import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

sys.path.append(str(Path(__file__).resolve().parent))

from model import BEDROOMS_BY_TYPE, PROPERTY_TYPE_OPTIONS, ZONE_REFERENCE

PropertyCategory = Literal["apartment", "villa"]
View = Literal[
    "community", "park", "city", "pool", "sea", "marina", "golf_course", "burj_khalifa"
]
Furnishing = Literal["unfurnished", "semi_furnished", "fully_furnished"]
Condition = Literal["vacant_on_transfer", "tenanted", "off_plan_resale"]


class PredictionRequest(BaseModel):
    """Raw property attributes needed to build one input row for the model.

    This is the full set of fields build_input_row() (model.py) needs to
    assemble a row in the exact schema the fitted pipeline expects.
    bedrooms is intentionally not a field here: like app.py, it's derived
    from property_type via BEDROOMS_BY_TYPE, since the training data has a
    strict 1:1 mapping and asking for it separately could produce
    combinations (e.g. "studio" with 4 bedrooms) the model never saw.
    """

    zone: str = Field(..., description="Zone name. Unseen zones fall back to a mean location.")
    property_category: PropertyCategory
    property_type: str
    area_sqft: int = Field(..., gt=0, le=15000)
    floor: float | None = Field(default=None, description="Required for apartments, must be null for villas.")
    total_floors: float | None = Field(default=None, description="Required for apartments, must be null for villas.")
    year_built: int
    view: View
    furnishing: Furnishing
    condition: Condition
    parking_spaces: int = Field(..., ge=1, le=4)
    is_freehold: bool
    chiller_included: bool
    mortgage_rate_at_listing: float = Field(..., ge=0)

    @model_validator(mode="after")
    def validate_property_type_and_floors(self) -> "PredictionRequest":
        valid_types = PROPERTY_TYPE_OPTIONS.get(self.property_category, [])
        if self.property_type not in valid_types:
            raise ValueError(
                f"property_type '{self.property_type}' is not valid for "
                f"property_category '{self.property_category}'. Valid options: {valid_types}"
            )

        if self.property_category == "villa":
            if self.floor is not None or self.total_floors is not None:
                raise ValueError("floor and total_floors must be null for villas (structurally absent in training data).")
        else:
            if self.floor is None or self.total_floors is None:
                raise ValueError("floor and total_floors are required for apartments.")
            if self.floor > self.total_floors:
                raise ValueError("floor cannot exceed total_floors.")

        return self

    @property
    def bedrooms(self) -> int:
        return BEDROOMS_BY_TYPE[self.property_type]


class PredictionResponse(BaseModel):
    """Predicted price, in the model's native currency and the display currency."""

    predicted_price_usd: float = Field(..., description="Raw model output — the model's native currency.")
    predicted_price_aed: float = Field(..., description="predicted_price_usd converted at the fixed AED/USD peg, for display only.")
    test_mae_usd: float = Field(..., description="Model's test-set MAE (USD), for context on estimate precision.")


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    known_zones: int = Field(..., description="Number of zones with hardcoded reference data available.")
