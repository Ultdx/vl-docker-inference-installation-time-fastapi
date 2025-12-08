from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from pathlib import Path


# Path to the trained model (.pkl)
MODEL_PATH = Path(__file__).parent / "models" / "vl-sgd-installation-time-1.0.pkl"

# Load the trained pipeline (preprocessor + model)
model = joblib.load(MODEL_PATH)

app = FastAPI(title="VL Installation Time Inference API")


class InputFeatures(BaseModel):
    software: str
    version: str
    software_addon: str
    is_active_int: int
    has_addon: int


@app.get("/")
async def root():
    """Simple health/root endpoint."""
    return {"status": "ok"}


@app.post("/predict")
async def predict(data: InputFeatures):
    """Predict installation time from input features using the loaded model."""
    df = pd.DataFrame([data.dict()])
    pred = model.predict(df)
    # Ensure the result is JSON serializable
    return {"prediction": float(pred[0])}
