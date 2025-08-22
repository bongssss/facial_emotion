import streamlit as st
import cv2

st.title("😊 Real-Time Face & Emotion Recognition")

run = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to grab frame")
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
