from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
import cv2

from .model import EmotionModel
from .utils import detect_faces, safe_crop

app = FastAPI(title="Emotion Recognition API", version="0.1.0")

# Allow Streamlit on localhost and typical dev ports; relax during dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # consider restricting to ["http://localhost:8501"] later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model (dummy for now)
model = EmotionModel()

class FacePrediction(BaseModel):
    bbox: List[int]     # [x, y, w, h]
    label: str
    score: float

class PredictResponse(BaseModel):
    num_faces: int
    predictions: List[FacePrediction]

@app.get("/")
def home() -> Dict[str, str]:
    return {"message": "Emotion Recognition API is running"}

@app.get("/labels")
def labels() -> Dict[str, List[str]]:
    return {"labels": model.get_labels()}

@app.post("/predict/", response_model=PredictResponse)
async def predict_emotion(file: UploadFile = File(...)):
    """
    Accepts an image and returns face predictions.
    """
    # Read -> decode into OpenCV BGR image
    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"num_faces": 0, "predictions": []}

    # Detect faces
    faces = detect_faces(frame)

    # Predict per face
    predictions: List[Dict[str, Any]] = []
    for (x, y, w, h) in faces:
        crop = safe_crop(frame, (x, y, w, h))
        pred = model.predict_on_crop(crop)
        predictions.append({
            "bbox": [x, y, w, h],
            "label": pred["label"],
            "score": pred["score"],
        })

    return {"num_faces": len(faces), "predictions": predictions}
