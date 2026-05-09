# Real-Time Human Age Detection Using Live Webcam Feed

This project implements a real-time system that detects human faces from a webcam and predicts age groups using a MobileNetV2 deep learning model.

## Technology Stack
- **Python 3.x**
- **TensorFlow / Keras** (MobileNetV2 Transfer Learning)
- **OpenCV** (Haar Cascades for face detection)

## Architecture Overview
1. **Face Detection**: Fast CPU-friendly detection using OpenCV's `haarcascade_frontalface_default.xml`.
2. **Age Prediction Model**: MobileNetV2 pre-trained on ImageNet, fine-tuned to classify 6 age groups (`0-10`, `11-20`, `21-30`, `31-40`, `41-50`, `51+`).
3. **Inference Pipeline**: Real-time webcam capture, bounding box calculation, frame resizing, skipping alternate frames, and exponential smoothing to reduce label flickering.

## Setup Instructions

1. **Install Requirements**
   Run the following command to install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Dataset Preparation (Optional, for training)**
   - Download the [UTKFace Dataset](https://susanqq.github.io/UTKFace/).
   - Extract the images into the `dataset/UTKFace` folder.

3. **Train the Model (Optional)**
   If you wish to train the model from scratch on the UTKFace dataset, run:
   ```bash
   python train.py --dataset dataset/UTKFace --epochs 10
   ```
   The best model will be saved to `models/best_age_model.h5`.

4. **Run Real-Time Detection**
   To start the webcam and view real-time predictions, run:
   ```bash
   python main.py
   ```
   *Note: If you trained the model, pass the model path as an argument:*
   ```bash
   python main.py --model models/best_age_model.h5
   ```
   If no model is provided, the script will use the initialized MobileNetV2 with random weights to demonstrate the pipeline speed and structure.

## Folder Structure
```
project/
├── dataset/             # Place UTKFace images here
├── models/              # Saved model weights
├── src/
│   ├── face_detection.py # OpenCV Haar Cascade wrapper
│   ├── age_model.py      # MobileNetV2 model architecture
│   ├── utils.py          # Smoothing and drawing utilities
├── train.py             # Training script
├── main.py              # Webcam inference pipeline
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
```
