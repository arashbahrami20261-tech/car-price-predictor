"""FastAPI service that serves the trained used-car price model.

The model artifact is loaded once when the process starts, not on every
request, so a prediction costs a few milliseconds instead of a few seconds.

Run:
    uvicorn src.api:app --reload
Then open http://127.0.0.1:8000/docs
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"

# Filled in by the lifespan handler below.
artifact: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model into memory before the first request is served."""
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"{MODEL_PATH} not found. Run `python src/train.py` before starting the API."
        )
    artifact.update(joblib.load(MODEL_PATH))
    yield
    artifact.clear()


app = FastAPI(
    title="Car Price Predictor",
    description="Estimates the market price of a second-hand car.",
    version="1.0.0",
    lifespan=lifespan,
)


class CarFeatures(BaseModel):
    """Request schema.

    Every field is constrained, so a malformed request is rejected with a clear
    422 response before it ever reaches the model.
    """

    model: str = Field(..., examples=["peugeot 206"])
    year: int = Field(..., ge=1990, le=2026)
    mileage_km: int = Field(..., ge=0, le=700_000)
    engine_size_l: float = Field(..., gt=0.5, lt=8.0)
    fuel_type: Literal["petrol", "petrol_lpg"]
    transmission: Literal["manual", "automatic"]
    body_condition: Literal[
        "intact", "minor_paint", "major_paint", "accident_repaired"
    ]
    color: str = Field(..., examples=["white"])
    previous_owners: int = Field(..., ge=1, le=15)


class PredictionResponse(BaseModel):
    predicted_price_million_toman: float
    currency: str = "IRR (million toman)"
    model_version: str


@app.get("/health")
def health() -> dict:
    """Liveness probe. Container orchestrators call this, not humans."""
    return {"status": "ok", "model_loaded": bool(artifact)}


@app.get("/model-info")
def model_info() -> dict:
    """Expose training metrics and accepted categories for debugging."""
    if not artifact:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return {
        "trained_at": artifact["trained_at"],
        "metrics": artifact["metrics"],
        "known_categories": artifact["categories"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CarFeatures) -> PredictionResponse:
    if not artifact:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    payload = features.model_dump()
    # Apply the same text normalisation the training data went through.
    for column in artifact["categorical_features"]:
        payload[column] = str(payload[column]).strip().lower()

    frame = pd.DataFrame([payload])
    prediction = artifact["pipeline"].predict(frame)[0]

    return PredictionResponse(
        predicted_price_million_toman=round(float(prediction), 1),
        model_version=artifact["trained_at"],
    )
