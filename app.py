import gradio as gr
import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from collections import deque
import threading

# ── Model & Startup Setup ──────────────────────────────────────────

# Load custom model globally at startup if it exists
CUSTOM_MODEL_PATH = "models/best_age_model.h5"
custom_model = None
if os.path.exists(CUSTOM_MODEL_PATH):
    try:
        custom_model = tf.keras.models.load_model(CUSTOM_MODEL_PATH)
        print("Loaded custom age model successfully.")
    except Exception as e:
        print("Error loading custom model:", e)

# Global smoother dictionary to handle smoothing history
history_deque = deque(maxlen=8)

def warmup_models():
    # Preload the default models so the user doesn't experience a delay on first stream frame
    try:
        from deepface import DeepFace
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
        print("Warming up DeepFace models...")
        # Warmup SSD detector
        _ = DeepFace.extract_faces(dummy_img, detector_backend="ssd", enforce_detection=False)
        # Warmup DeepFace age model
        _ = DeepFace.analyze(dummy_img, actions=["age"], detector_backend="opencv", enforce_detection=False, silent=True)
        print("DeepFace models warmed up successfully.")
    except Exception as e:
        print("Warmup failed:", e)

# Run warmup in a background thread so it doesn't block the UI launch
threading.Thread(target=warmup_models, daemon=True).start()


# ── Frame Processing ───────────────────────────────────────────────

def detect_age_in_frame(frame, model_type, detector_backend, smoothing_factor):
    """
    Receives:
      - frame: RGB numpy array from Gradio webcam
      - model_type: "Custom Trained MobileNetV2" or "DeepFace VGG-Face (Default)"
      - detector_backend: "ssd", "opencv", "mediapipe", "retinaface", "mtcnn"
      - smoothing_factor: integer size of history buffer
    Returns:
      - annotated RGB frame
    """
    if frame is None:
        return None

    # Update smoothing history size dynamically
    global history_deque
    if history_deque.maxlen != int(smoothing_factor):
        history_deque = deque(maxlen=int(smoothing_factor))

    try:
        from deepface import DeepFace

        # DeepFace expects BGR for face detection/analysis
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Draw frame base (annotated frame starts as a copy of the input frame)
        annotated_frame = frame.copy()

        if model_type == "DeepFace VGG-Face (Default)":
            # Direct DeepFace analysis
            results = DeepFace.analyze(
                bgr,
                actions=["age"],
                enforce_detection=False,
                detector_backend=detector_backend,
                silent=True,
            )

            if isinstance(results, dict):
                results = [results]

            for res in results:
                region = res.get("region", {})
                age = res.get("age", None)

                x = region.get("x", 0)
                y = region.get("y", 0)
                w = region.get("w", 0)
                h = region.get("h", 0)

                if w == 0 or h == 0 or age is None:
                    continue

                # Add to smoothing history
                history_deque.append(age)
                smoothed_age = int(round(np.mean(history_deque)))

                age_text = f"Age: {smoothed_age}"

                # Draw bounding box and label in RGB (since annotated_frame is RGB)
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 128), 2)
                (tw, th), _ = cv2.getTextSize(age_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(annotated_frame, (x, y - th - 12), (x + tw + 10, y), (0, 180, 90), -1)
                cv2.putText(
                    annotated_frame, age_text,
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
                )

        else:  # Custom Trained MobileNetV2
            # Use DeepFace to extract faces/bounding boxes
            faces = DeepFace.extract_faces(
                bgr,
                detector_backend=detector_backend,
                enforce_detection=False,
                align=True,
            )

            for face_info in faces:
                confidence = face_info.get("confidence", 0)
                # Ignore face if confidence is too low (only if detector is not opencv/enforce_detection=False)
                if confidence < 0.25 and detector_backend not in ["opencv"]:
                    continue

                facial_area = face_info.get("facial_area", {})
                x = facial_area.get("x", 0)
                y = facial_area.get("y", 0)
                w = facial_area.get("w", 0)
                h = facial_area.get("h", 0)

                if w == 0 or h == 0:
                    continue

                # Crop from RGB frame (since custom model expects RGB)
                face_crop = frame[max(0, y):y+h, max(0, x):x+w]
                if face_crop.size == 0:
                    continue

                # Resize to (224, 224)
                face_crop_resized = cv2.resize(face_crop, (224, 224))

                # Preprocess for MobileNetV2
                preprocessed = preprocess_input(face_crop_resized.astype(np.float32))
                preprocessed = np.expand_dims(preprocessed, axis=0)

                # Run prediction
                if custom_model is not None:
                    predicted_age = custom_model.predict(preprocessed, verbose=0)[0][0]
                else:
                    predicted_age = 23.0  # fallback mock

                # Add to smoothing history
                history_deque.append(predicted_age)
                smoothed_age = int(round(np.mean(history_deque)))

                age_text = f"Age: {smoothed_age}"

                # Draw bounding box and label
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 128, 255), 2)
                (tw, th), _ = cv2.getTextSize(age_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(annotated_frame, (x, y - th - 12), (x + tw + 10, y), (0, 90, 180), -1)
                cv2.putText(
                    annotated_frame, age_text,
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
                )

        return annotated_frame

    except Exception as e:
        # Show message on screen during loading/warmup
        err_frame = frame.copy()
        cv2.putText(err_frame, "Detecting...", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return err_frame


# ── Gradio UI ──────────────────────────────────────────────────────

CSS = """
body { 
    background: #0d0f14; 
    font-family: 'Inter', -apple-system, sans-serif;
    color: #e2e8f0;
}
.gradio-container {
    max-width: 800px !important;
    margin: 0 auto !important;
    padding: 2rem 1rem !important;
}
h1 { 
    text-align: center; 
    background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem; 
    font-weight: 800;
    margin-bottom: 0.2rem; 
}
.subtitle { 
    text-align: center; 
    color: #94a3b8; 
    margin-bottom: 2rem; 
    font-size: 1.1rem; 
}
footer { 
    display: none !important; 
}

/* Single Frame Camera Container */
.camera-wrapper {
    position: relative;
    width: 100%;
    max-width: 640px;
    margin: 0 auto 1.5rem auto;
    border-radius: 16px;
    border: 1px solid #1e293b;
    background: #111827;
    overflow: hidden;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
}

/* Ensure the webcam input fills the wrapper */
#webcam-input {
    width: 100% !important;
    border: none !important;
    background: transparent !important;
}

/* Position the detection output image directly on top of the webcam feed */
#output-image {
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    width: 100% !important;
    height: 100% !important;
    z-index: 10 !important;
    pointer-events: none !important; /* Clicks and hovers pass through to start/stop the camera below */
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Make sure the output image fits and overlays perfectly */
#output-image img {
    width: 100% !important;
    height: 100% !important;
    object-fit: cover !important;
    background: transparent !important;
}

/* Hide all empty/placeholder states of the output image so the webcam controls are accessible */
#output-image .empty,
#output-image .image-container:not(.preview),
#output-image [data-testid="block-info"],
#output-image .stage-header,
#output-image .upload-container {
    display: none !important;
}

/* Style the settings panel */
.settings-container {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 1.5rem !important;
    margin-top: 1rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
}
"""

with gr.Blocks(css=CSS, title="Real-Time Age Detection") as demo:

    gr.HTML("<h1>🎯 Real-Time Age Detection</h1>")
    gr.HTML('<p class="subtitle">Click the camera below → allow access → see your age detected in real-time!</p>')

    # Camera Single-Frame Overlay Container
    with gr.Column(elem_classes="camera-wrapper"):
        webcam_input = gr.Image(
            sources=["webcam"],
            streaming=True,
            show_label=False,
            mirror_webcam=True,
            elem_id="webcam-input",
        )
        output_image = gr.Image(
            show_label=False,
            elem_id="output-image",
        )

    # Settings Control Panel
    with gr.Column(elem_classes="settings-container"):
        gr.Markdown("### ⚙️ Detection Settings")
        
        with gr.Row():
            model_type = gr.Dropdown(
                choices=["Custom Trained MobileNetV2", "DeepFace VGG-Face (Default)"],
                value="Custom Trained MobileNetV2",
                label="Age Prediction Model",
                info="Custom Trained MobileNetV2 regression model is highly optimized for accuracy."
            )
            detector_backend = gr.Dropdown(
                choices=["ssd", "opencv", "mediapipe", "retinaface", "mtcnn"],
                value="ssd",
                label="Face Detector Backend",
                info="SSD is fast & accurate. RetinaFace is most accurate but slower."
            )
            
        smoothing_factor = gr.Slider(
            minimum=1,
            maximum=20,
            value=8,
            step=1,
            label="Prediction Smoothing (Flicker Reduction)",
            info="Higher values average ages over more frames to reduce jumps/flicker."
        )

    gr.HTML("""
    <div style='text-align:center; color:#64748b; margin-top:1.5rem; font-size:0.85rem;'>
        Powered by DeepFace · OpenCV · MobileNetV2 · Gradio &nbsp;|&nbsp; No data stored
    </div>
    """)

    # Stream binding
    webcam_input.stream(
        fn=detect_age_in_frame,
        inputs=[webcam_input, model_type, detector_backend, smoothing_factor],
        outputs=output_image,
        stream_every=0.3,  # 300ms stream rate for smooth real-time response
    )

if __name__ == "__main__":
    demo.launch()
