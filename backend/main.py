
from pathlib import Path
from datetime import datetime, timedelta
import json
import math

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "xgb_model.joblib"
FEATURE_PATH = BASE_DIR / "models" / "feature_names.json"

# Load the trained model and feature names
model = joblib.load(MODEL_PATH)

with open(FEATURE_PATH, "r") as f:
    feature_names = json.load(f)

app = FastAPI(
    title="EcoForecast API",
    description="Hourly energy forecasting using XGBoost",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Development only
    allow_methods=["*"],
    allow_headers=["*"]
)


class HourlyObservation(BaseModel):
    timestamp: datetime
    value: float


class PredictionRequest(BaseModel):
    # Supply at least 168 consecutive hourly observations,
    # oldest first and newest last.
    observations: list[HourlyObservation]


@app.get("/")
def home():
    return {"message": "Welcome to EcoForecast API"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "feature_count": len(feature_names)
    }


def generate_features(observations):
    if len(observations) < 168:
        raise HTTPException(
            status_code=400,
            detail="At least 168 hourly observations are required."
        )

    timestamps = [item.timestamp for item in observations]
    values = [item.value for item in observations]

    if not all(math.isfinite(v) for v in values):
        raise HTTPException(
            status_code=400,
            detail="All energy values must be finite numbers."
        )

    # This version expects timezone-naive, hourly timestamps.
    if any(t.tzinfo is not None for t in timestamps):
        raise HTTPException(
            status_code=400,
            detail="Use timezone-naive timestamps, consistent with training data."
        )

    if any(
        timestamps[i] - timestamps[i - 1] != timedelta(hours=1)
        for i in range(1, len(timestamps))
    ):
        raise HTTPException(
            status_code=400,
            detail="Observations must be chronological and exactly one hour apart."
        )

    # Features are calculated for the hour after the latest reading.
    prediction_time = timestamps[-1] + timedelta(hours=1)
    target = pd.Series(values, dtype="float64")

    hour = prediction_time.hour
    day_of_week = prediction_time.weekday()
    month = prediction_time.month

    features = {
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": int(day_of_week >= 5),

        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),

        "day_sin": np.sin(2 * np.pi * day_of_week / 7),
        "day_cos": np.cos(2 * np.pi * day_of_week / 7),

        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
    }

    # Lags: lag_1 is the most recent observed hour.
    for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
        features[f"lag_{lag}"] = float(target.iloc[-lag])

    # Rolling statistics use observations up to the latest known hour.
    for window in [6, 12, 24, 48, 168]:
        recent = target.iloc[-window:]
        features[f"rolling_mean_{window}"] = float(recent.mean())
        features[f"rolling_std_{window}"] = float(recent.std())

    return features, prediction_time


@app.post("/predict")
def predict(request: PredictionRequest):
    features, prediction_time = generate_features(request.observations)

    missing = [name for name in feature_names if name not in features]
    if missing:
        raise HTTPException(
            status_code=500,
            detail={"missing_model_features": missing}
        )

    # Preserve the exact feature order used during training.
    input_df = pd.DataFrame(
        [[features[name] for name in feature_names]],
        columns=feature_names
    )

    prediction = float(model.predict(input_df)[0])

    return {
        "prediction_timestamp": prediction_time.isoformat(),
        "predicted_energy_value": prediction,
        "unit": "same unit as the training target"
    }
    # Path to the original household electricity dataset
DATA_PATH = BASE_DIR / "data" / "raw" / "latest_history.csv"


class ForecastRequest(BaseModel):
    horizon: int


def load_latest_history():
    """
    Load the latest 168 consecutive hourly observations
    from the original household power consumption dataset.
    """
    if not DATA_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Dataset not found at: {DATA_PATH}"
        )

    try:
        df = pd.read_csv(
            DATA_PATH,
            sep=";",
            usecols=["Date", "Time", "Global_active_power"],
            na_values=["?"],
            low_memory=False
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read dataset: {str(e)}"
        )

    # Combine date and time into one timestamp
    df["timestamp"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce"
    )

    # Convert energy readings to numeric values
    df["Global_active_power"] = pd.to_numeric(
        df["Global_active_power"],
        errors="coerce"
    )

    df = df.dropna(subset=["timestamp", "Global_active_power"])

    # Calculate hourly mean power
    hourly = (
        df.set_index("timestamp")["Global_active_power"]
        .resample("h")
        .mean()
        .dropna()
    )

    if len(hourly) < 168:
        raise HTTPException(
            status_code=500,
            detail="Not enough valid hourly data in the dataset."
        )

    # Identify continuous hourly sections
    run_ids = (
        hourly.index.to_series()
        .diff()
        .ne(pd.Timedelta(hours=1))
        .cumsum()
    )

    # Find the most recent section with at least 168 observations
    valid_runs = [
        run for _, run in hourly.groupby(run_ids)
        if len(run) >= 168
    ]

    if not valid_runs:
        raise HTTPException(
            status_code=500,
            detail="No 168-hour continuous section found in the dataset."
        )

    latest_run = valid_runs[-1].tail(168)

    # Convert the observations to the format used by the API
    history = [
        HourlyObservation(
            timestamp=timestamp.to_pydatetime(),
            value=float(value)
        )
        for timestamp, value in latest_run.items()
    ]

    return history


@app.get("/history")
def get_history():
    """Return the latest 168 consecutive hourly observations."""
    history = load_latest_history()

    return {
        "count": len(history),
        "unit": "kW",
        "observations": [
            {
                "timestamp": item.timestamp.isoformat(),
                "value": item.value
            }
            for item in history
        ]
    }


@app.post("/forecast")
def forecast(request: ForecastRequest):
    """Generate recursive multi-step hourly forecasts."""

    if request.horizon not in [24, 48, 72, 168]:
        raise HTTPException(
            status_code=400,
            detail="Horizon must be 24, 48, 72, or 168 hours."
        )

    # Load historical observations automatically
    history = load_latest_history()
    predictions = []

    # Check that the model's required features are available
    # before generating any predictions
    missing = [
        name for name in feature_names
        if name not in generate_features(history)[0]
    ]

    if missing:
        raise HTTPException(
            status_code=500,
            detail={"missing_model_features": missing}
        )

    # Generate predictions one hour at a time
    for _ in range(request.horizon):
        features, prediction_time = generate_features(history)

        input_df = pd.DataFrame(
            [[features[name] for name in feature_names]],
            columns=feature_names
        )

        prediction = float(model.predict(input_df)[0])

        if not math.isfinite(prediction):
            raise HTTPException(
                status_code=500,
                detail="The model generated a non-finite prediction."
            )

        predictions.append({
            "timestamp": prediction_time.isoformat(),
            "value": prediction
        })

        # Add the predicted value to history for the next step
        history.append(
            HourlyObservation(
                timestamp=prediction_time,
                value=prediction
            )
        )

    values = [item["value"] for item in predictions]

    return {
        "model": "XGBoost",
        "horizon": request.horizon,
        "unit": "kW",
        "average": float(np.mean(values)),
        "peak": float(np.max(values)),
        "predictions": predictions
    }
