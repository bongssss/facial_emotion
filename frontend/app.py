import os
import time
import json
import cv2
import numpy as np
import requests
import streamlit as st

# ====== Config ======
DEFAULT_API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
PREDICT_ENDPOINT = "/predict/"

# ====== UI ======
st.set_page_config(page_title="Face & Emotion Recognition", layout="wide")
st.title("😊 Real-Time Face & Emotion Recognition")

with st.sidebar:
    st.subheader("Settings")
    api_url = st.text_input("Backend API URL", value=DEFAULT_API_URL, help="FastAPI base URL")
    conf_thresh = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
    frame_stride = st.number_input("Send every Nth frame", min_value=1, max_value=10, value=2, help="Reduce API calls")
    draw_thickness = st.number_input("Box thickness", min_value=1, max_value=5, value=2)
    font_scale = st.slider("Label font scale", 0.4, 2.0, 0.7, 0.1)
    start = st.button("▶️ Start")
    stop = st.button("⏹ Stop")

# Keep state across reruns
if "running" not in st.session_state:
    st.session_state.running = False

if start:
    st.session_state.running = True
if stop:
    st.session_state.running = False

frame_area = st.empty()
status_area = st.empty()

def encode_jpeg(frame_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()

def call_predict_api(api_base: str, frame_bgr: np.ndarray) -> dict:
    url = api_base.rstrip("/") + PREDICT_ENDPOINT
    payload = encode_jpeg(frame_bgr)
    files = {"file": ("frame.jpg", payload, "image/jpeg")}
    resp = requests.post(url, files=files, timeout=10)
    resp.raise_for_status()
    return resp.json()

def draw_predictions(frame_bgr: np.ndarray, predictions: list, conf: float, thickness: int, font_scale: float) -> np.ndarray:
    out = frame_bgr.copy()
    for p in predictions:
        score = float(p.get("score", 0.0))
        if score < conf:
            continue
        x, y, w, h = map(int, p["bbox"])
        label = p.get("label", "?")
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), thickness)
        text = f"{label} {score:.2f}"
        # Text background for legibility
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(out, (x, y - th - 6), (x + tw + 4, y), (0, 255, 0), -1)
        cv2.putText(out, text, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
    return out

# ====== Main loop ======
cap = None
try:
    if st.session_state.running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            status_area.error("Could not access webcam (device 0).")
            st.session_state.running = False

    frame_idx = 0
    last_fps_t = time.time()
    frames_shown = 0

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            status_area.error("Failed to grab frame from webcam.")
            break

        # Send to API every Nth frame to control load
        predictions = []
        if frame_idx % int(frame_stride) == 0:
            try:
                result = call_predict_api(api_url, frame)
                predictions = result.get("predictions", [])
                status_area.info(
                    f"Faces: {result.get('num_faces', 0)} | API: {api_url}{PREDICT_ENDPOINT}"
                )
            except requests.exceptions.RequestException as e:
                status_area.error(f"API error: {e}")
                # keep showing raw webcam if API down

        # Draw overlays
        frame_annot = draw_predictions(frame, predictions, conf_thresh, int(draw_thickness), float(font_scale))

        # Convert BGR -> RGB for Streamlit display
        frame_rgb = cv2.cvtColor(frame_annot, cv2.COLOR_BGR2RGB)
        frame_area.image(frame_rgb, channels="RGB")

        # Simple FPS display
        frames_shown += 1
        now = time.time()
        if now - last_fps_t >= 1.0:
            st.caption(f"Approx FPS: {frames_shown/(now-last_fps_t):.1f}")
            frames_shown = 0
            last_fps_t = now

        frame_idx += 1

except Exception as e:
    status_area.error(f"Unexpected error: {e}")

finally:
    if cap is not None:
        cap.release()
