from fastapi import FastAPI, UploadFile
import cv2
import numpy as np
from .model import EmotionModel
from .utils import detect_faces



app = FastAPI()
model = EmotionModel(num_classes=7)  # e.g., happy, sad, angry, neutral, surprise, fear, disgust

@app.get("/")
def home():
    return {"message": "Emotion Recognition API is running"}

@app.post("/predict/")
async def predict_emotion(file: UploadFile):
    # Read image from request
    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Detect faces
    faces = detect_faces(frame)

    # For now, just return number of faces
    return {"faces_detected": len(faces)}
