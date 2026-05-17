from fastapi import FastAPI, HTTPException, UploadFile, File
import easyocr
import io
import numpy as np
from PIL import Image
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime

from app.llm_parser import parse_property_details

app = FastAPI(
    title="Brooklyn Home Price Prediction API",
    version="1.2.0"
)

pipeline = joblib.load("models/brooklyn_price_pipeline_raw_inputs.joblib")

reader = easyocr.Reader(["en"])

class PredictionRequest(BaseModel):
    neighborhood: str
    building_class_category: str
    gross_sqft: float
    dist_to_station: float
    build_age_yrs: float
    within_half_mi: int
    
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
    neighborhood = input_data.get("neighborhood")
    building_class_category = input_data.get("building_class_category")
    gross_sqft = input_data.get("gross_sqft")
    dist_to_station = input_data.get("dist_to_station")
    build_age_yrs = input_data.get("build_age_yrs")
    within_half_mi = input_data.get("within_half_mi")

    if not neighborhood:
        raise HTTPException(status_code=400, detail="Missing neighborhood.")
    if not building_class_category:
        raise HTTPException(status_code=400, detail="Missing building_class_category.")
    if gross_sqft is None:
        raise HTTPException(status_code=400, detail="Missing gross_sqft.")
    if dist_to_station is None:
        raise HTTPException(status_code=400, detail="Missing dist_to_station.")
    if build_age_yrs is None:
        raise HTTPException(status_code=400, detail="Missing build_age_yrs.")
    if within_half_mi is None:
        raise HTTPException(status_code=400, detail="Missing within_half_mi.")

    gross_sqft = float(gross_sqft)
    dist_to_station = float(dist_to_station)
    build_age_yrs = float(build_age_yrs)
    within_half_mi = int(within_half_mi)

    if gross_sqft <= 0:
        raise HTTPException(status_code=400, detail="gross_sqft must be greater than 0.")
    if dist_to_station <= 0:
        raise HTTPException(status_code=400, detail="dist_to_station must be greater than 0.")
    if within_half_mi not in [0, 1]:
        raise HTTPException(status_code=400, detail="within_half_mi must be 0 or 1.")

    df = pd.DataFrame([{
        "neighborhood": neighborhood,
        "building_class_category": building_class_category,
        "gross_sqft": gross_sqft,
        "dist_to_station": dist_to_station,
        "build_age_yrs": build_age_yrs,
        "within_half_mi": within_half_mi
    }])

    predicted_log_price = float(pipeline.predict(df)[0])
    predicted_price = float(10 ** predicted_log_price)

    return round(predicted_price, 2)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictionRequest):
    predicted_price = run_prediction(payload.model_dump())
    return {"predicted_price_usd": predicted_price}


@app.post("/predict-from-text")
def predict_from_text(payload: PromptRequest):
    raw_data = parse_property_details(payload.user_prompt)

    if not raw_data:
        raise HTTPException(status_code=400, detail="Could not parse property details.")

    year_built = raw_data.get("year_built")
    if year_built is None:
        raise HTTPException(status_code=400, detail="Missing year_built.")

    current_year = datetime.now().year

    input_data = {
        "neighborhood": raw_data.get("neighborhood"),
        "building_class_category": raw_data.get("building_class_category"),
        "gross_sqft": raw_data.get("gross_sqft"),
        "dist_to_station": raw_data.get("distance_to_station"),
        "build_age_yrs": current_year - int(year_built),
        "within_half_mi": raw_data.get("within_half_mi")
    }

    predicted_price = run_prediction(input_data)

    return {
        "extracted_features": input_data,
        "predicted_price_usd": predicted_price
    }
    
@app.post("/predict-from-image")
async def predict_from_image(file: UploadFile = File(...)):
    data = await file.read()
    image = Image.open(io.BytesIO(data))
    
    text_results = reader.readtext(np.array(image), detail=0)
    user_prompt = " ".join(text_results)
    
    raw_data = parse_property_details(user_prompt)

    if not raw_data:
        raise HTTPException(status_code=400, detail="Could not parse property details.")

    year_built = raw_data.get("year_built")
    if year_built is None:
        raise HTTPException(status_code=400, detail="Missing year_built.")

    current_year = datetime.now().year

    input_data = {
        "neighborhood": raw_data.get("neighborhood"),
        "building_class_category": raw_data.get("building_class_category"),
        "gross_sqft": raw_data.get("gross_sqft"),
        "dist_to_station": raw_data.get("distance_to_station"),
        "build_age_yrs": current_year - int(year_built),
        "within_half_mi": raw_data.get("within_half_mi")
    }

    predicted_price = run_prediction(input_data)

    return {
        "ocr_text": user_prompt,
        "extracted_features": input_data,
        "predicted_price_usd": predicted_price
    }