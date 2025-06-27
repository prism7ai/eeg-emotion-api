from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from datetime import datetime
import csv
import os

# === FastAPI App ===
app = FastAPI(title="EEG Emotion Predictor API")

# === Enable CORS for Frontend Access ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set this to your deployed frontend URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load Model, Scaler, Encoder ===
model = joblib.load("emotion_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# === Define Input Schema ===
class EEGInput(BaseModel):
    features: list[float]  # Must be 2548 floats

# === CSV Log File ===
LOG_FILE = "logs.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "predicted_emotion", "input_preview"])

# === Root Route ===
@app.get("/")
def read_root():
    return {"message": "EEG Emotion Predictor is running."}

# === Prediction Endpoint ===
@app.post("/predict")
def predict_emotion(input_data: EEGInput):
    if len(input_data.features) != 2548:
        raise HTTPException(status_code=400, detail="Expected 2548 EEG features")

    # Normalize and predict
    X = np.array(input_data.features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    emotion = label_encoder.inverse_transform(y_pred)[0]

    # Log to CSV
    timestamp = datetime.now().isoformat()
    input_preview = input_data.features[:5]  # just show first 5 values
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, emotion, input_preview])

    return {"predicted_emotion": emotion}
