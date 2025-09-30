
---

### **2. Face & Emotion Recognition**  
```markdown
#  Real-Time Face & Emotion Recognition

##  Overview
This project is a **deep learning system for face & emotion recognition**.  
It detects faces from a webcam feed and classifies emotions (happy, sad, neutral, angry, etc.).

##  Features
- Real-time face detection (OpenCV).
- Emotion classification with CNN/ResNet.
- Overlay bounding boxes + labels.
- Logs emotions with timestamps.
- Web interface for live demo.
- API endpoint that returns **per-face emotion predictions** (dummy for now).
- Streamlit frontend to preview webcam (integration to API in next step).
- CORS-enabled backend for easy local dev.
- Clear, swappable model interface.


##  Tech Stack (with docs)
- **Python** — https://www.python.org/  
- **PyTorch** — https://pytorch.org/  
- **Torchvision** — https://pytorch.org/vision/stable/index.html  
- **NumPy** — https://numpy.org/  
- **Pandas** — https://pandas.pydata.org/  
- **scikit-learn** — https://scikit-learn.org/stable/  
- **OpenCV** — https://opencv.org/  
  - **Haar Cascades** — https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html  
- **FastAPI** — https://fastapi.tiangolo.com/  
- **Uvicorn** — https://www.uvicorn.org/  
- **Pydantic** — https://docs.pydantic.dev/  
- **Streamlit** — https://streamlit.io/  
- **Matplotlib** — https://matplotlib.org/ (optional for plots)
- **SQLite** — https://www.sqlite.org/index.html (optional for logging)
- **Pillow** — https://python-pillow.org/ (image I/O for transforms)

## How it Works (Flow)

1. Frontend (Streamlit) captures webcam frames.
2. Backend (FastAPI) accepts images at /predict/.
3. OpenCV detects faces and crops them.
4. EmotionModel returns a label + score for each crop (dummy now).
5. Frontend will overlay results on the live video (coming next).

##  Example
**Input:** Webcam feed  
**Output:** Bounding box with label `"Happy"`

##  Screenshots
(Add video stills of detection)

##  Installation
```bash
git clone <your-repo-url> emotion-recognition
cd emotion-recognition
python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows
# venv\Scripts\activate

pip install -r requirements.txt
```
```bash
# Run the Backend
uvicorn backend.main:app --reload
```
```bash
# Test the Prediction API (dummy model)
curl -X POST "http://127.0.0.1:8000/predict/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/any_face_image.jpg"
```
```bash
# Example JSON response:

{
  "num_faces": 1,
  "predictions": [
    {
      "bbox": [234, 120, 96, 96],
      "label": "happy",
      "score": 0.41
    }
  ]
}
```
```bash
# Run the Frontend (Webcam preview)

streamlit run frontend/app.py
```

###  Swapping in a Real Model

Replace the **predict_logits()** method in backend/model.py with your trained model:
Load weights once in **__init__**.
Keep the preprocess_np_bgr() and predict_on_crop() interfaces intact.
Return the same **{"label", "score"}** dict.

### 📝 Notes
The current predictor is random and only for plumbing.
For production, consider OpenCV DNN or RetinaFace for detection and a trained CNN/ResNet/ViT head for emotions.
Enable stricter CORS before deployment.
