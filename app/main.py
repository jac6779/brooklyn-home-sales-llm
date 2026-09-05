from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
import pandas as pd
import joblib
from datetime import datetime

from app.llm_parser import parse_property_details


app = FastAPI(
    title="Brooklyn Home Price Prediction API",
)

pipeline = joblib.load("models/brooklyn_price_pipeline_raw_inputs.joblib")


# Sample JSON
class PredictionRequest(BaseModel):
    neighborhood: str
    building_class_category: str
    gross_sqft: float = Field(..., gt=0)
    dist_to_station: float = Field(..., gt=0)
    build_age_yrs: float = Field(..., ge=0)
    within_half_mi: int = Field(..., ge=0, le=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "neighborhood": "park_slope",
                "building_class_category": "one_family_dwellings",
                "gross_sqft": 1800,
                "dist_to_station": 0.2,
                "build_age_yrs": 65,
                "within_half_mi": 1
            }
        }
    }


# Sample prompt
class PromptRequest(BaseModel):
    user_prompt: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_prompt": "A one-family home in Bay Ridge with about 2,500 square feet, built in 1950, around 0.3 miles from the subway."
            }
        }
    }


def run_prediction(input_data: dict) -> float:
    df = pd.DataFrame([input_data])

    predicted_log_price = pipeline.predict(df)[0]
    predicted_price = 10 ** predicted_log_price

    return round(float(predicted_price), 2)


# Predict from text
@app.post("/predict-from-text")
def predict_from_text(payload: PromptRequest):
    raw_data = parse_property_details(payload.user_prompt)

    if not raw_data:
        raise HTTPException(
            status_code=400,
            detail="Could not parse property details."
        )

    year_built = raw_data.get("year_built")

    if year_built is None:
        raise HTTPException(
            status_code=400,
            detail="Missing year_built."
        )

    input_data = {
        "neighborhood": raw_data.get("neighborhood"),
        "building_class_category": raw_data.get("building_class_category"),
        "gross_sqft": raw_data.get("gross_sqft"),
        "dist_to_station": raw_data.get("distance_to_station"),
        "build_age_yrs": datetime.now().year - int(year_built),
        "within_half_mi": raw_data.get("within_half_mi")
    }

    try:
        validated_data = PredictionRequest(**input_data)
    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=e.errors()
        )

    input_data = validated_data.model_dump()
    predicted_price = run_prediction(input_data)

    return {
        "extracted_features": input_data,
        "predicted_price_usd": predicted_price
    }


# Prediction
@app.post("/predict")
def predict(payload: PredictionRequest):
    predicted_price = run_prediction(payload.model_dump())

    return {
        "predicted_price_usd": predicted_price
    }


# Health Check
@app.get("/health")
def health():
    return {"status": "ok"}