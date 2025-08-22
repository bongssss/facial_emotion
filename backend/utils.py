import cv2
import numpy as np
from typing import List, Tuple

# Load Haar Cascade face detector once.
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_faces(frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Returns a list of (x, y, w, h) for detected faces in BGR frame.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

def safe_crop(frame_bgr: np.ndarray, box: Tuple[int, int, int, int], pad: int = 10) -> np.ndarray:
    """
    Safely crop a face region with optional padding.
    """
    h, w = frame_bgr.shape[:2]
    x, y, bw, bh = box
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + bw + pad)
    y2 = min(h, y + bh + pad)
    crop = frame_bgr[y1:y2, x1:x2].copy()
    return crop
