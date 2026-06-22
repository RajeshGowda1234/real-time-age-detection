---
title: Real-Time Age Detection
emoji: 🎯
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "5.33.0"
app_file: app.py
pinned: false
license: mit
python_version: "3.11"
short_description: Live webcam age detection using DeepFace + OpenCV
---

# 🎯 Real-Time Age Detection

A real-time age detection app that uses your webcam to detect faces and predict exact ages using deep learning.

## How It Works
1. Allow camera access in your browser
2. Your face is detected using OpenCV
3. DeepFace estimates your exact age using a pre-trained deep learning model
4. Your age is displayed in real-time on screen

## Technology Stack
- **Gradio** — web interface with live webcam streaming
- **DeepFace** — pre-trained deep learning model for age estimation
- **OpenCV** — face detection and image processing
- **Python 3.10**

## Run Locally
```bash
pip install -r requirements.txt
python app.py
```

## Project Structure
```
├── app.py              # Gradio web app (webcam streaming)
├── main.py             # Local OpenCV desktop app
├── train.py            # Model training script
├── src/
│   ├── face_detection.py
│   ├── age_model.py
│   └── utils.py
├── requirements.txt
└── README.md
```
