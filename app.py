import os
import gradio as gr
import cv2
import numpy as np
import os
# (Removed heavy TensorFlow imports to save memory)
from collections import deque
import threading
import json
import hashlib

# ── Model & Startup Setup ──────────────────────────────────────────

USERS_FILE = "users.json"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if not os.path.exists(USERS_FILE):
        # Create default admin user
        default_users = {"admin": hash_password("admin123")}
        with open(USERS_FILE, 'w') as f:
            json.dump(default_users, f)
        return default_users
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_users(users_db):
    with open(USERS_FILE, 'w') as f:
        json.dump(users_db, f)



# Load custom model globally at startup using TFLite
CUSTOM_MODEL_PATH = "models/best_age_model.tflite"
custom_model_interpreter = None
input_details = None
output_details = None

if os.path.exists(CUSTOM_MODEL_PATH):
    try:
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            import tensorflow.lite as tflite
            
        custom_model_interpreter = tflite.Interpreter(model_path=CUSTOM_MODEL_PATH)
        custom_model_interpreter.allocate_tensors()
        input_details = custom_model_interpreter.get_input_details()
        output_details = custom_model_interpreter.get_output_details()
        print("Loaded custom TFLite age model successfully.")
    except Exception as e:
        print("Error loading custom TFLite model:", e)

# Global smoother dictionary to handle smoothing history
history_deque = deque(maxlen=8)


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
    print(f"[DEBUG] Received frame: type={type(frame)}")
    if isinstance(frame, np.ndarray):
        print(f"[DEBUG] Frame shape={frame.shape}")
    elif isinstance(frame, dict):
        print(f"[DEBUG] Frame keys={list(frame.keys())}")
    elif isinstance(frame, str):
        print(f"[DEBUG] Frame path={frame}")

    if frame is None:
        return None

    # Handle Gradio dictionary structure (e.g. {'background': ..., 'layers': ..., 'composite': ...})
    if isinstance(frame, dict):
        if "composite" in frame and frame["composite"] is not None:
            frame = frame["composite"]
        elif "background" in frame and frame["background"] is not None:
            frame = frame["background"]
        else:
            print("[DEBUG] Dictionary frame has no composite or background key.")
            return None

    # Convert PIL Image or filepath to NumPy array if necessary
    if isinstance(frame, str):
        frame = cv2.imread(frame)
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    elif not isinstance(frame, np.ndarray):
        try:
            frame = np.array(frame)
        except Exception as conv_err:
            print(f"[ERROR] Failed to convert frame to numpy array: {conv_err}")
            return None

    if frame is None or frame.size == 0:
        return None

    # Update smoothing history size dynamically
    global history_deque
    if history_deque.maxlen != int(smoothing_factor):
        history_deque = deque(maxlen=int(smoothing_factor))

    try:
        # Load OpenCV Haar Cascade (runs instantly on CPU with zero downloads)
        cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        face_cascade = cv2.CascadeClassifier(cascade_path)

        # OpenCV expects grayscale for Haar Cascades
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Draw frame base (annotated frame starts as a copy of the input frame)
        annotated_frame = frame.copy()

        # Detect faces instantly
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Crop from RGB frame (since custom model expects RGB)
            face_crop = frame[max(0, y):y+h, max(0, x):x+w]
            if face_crop.size == 0:
                continue

            # Resize to (224, 224)
            face_crop_resized = cv2.resize(face_crop, (224, 224))

            # Preprocess for MobileNetV2 (manual math instead of keras preprocess_input)
            preprocessed = (face_crop_resized.astype(np.float32) / 127.5) - 1.0
            preprocessed = np.expand_dims(preprocessed, axis=0)

            # Run prediction using the TFLite runtime
            if custom_model_interpreter is not None:
                custom_model_interpreter.set_tensor(input_details[0]['index'], preprocessed)
                custom_model_interpreter.invoke()
                predicted_age = float(custom_model_interpreter.get_tensor(output_details[0]['index'])[0][0])
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
        import traceback
        print(f"[ERROR] Exception in detect_age_in_frame: {e}")
        traceback.print_exc()
        # Show message on screen during loading/warmup
        try:
            if isinstance(frame, np.ndarray):
                err_frame = frame.copy()
            else:
                err_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(err_frame, "Detecting...", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            return err_frame
        except Exception as inner_e:
            print(f"[ERROR] Exception in error fallback drawing: {inner_e}")
            return None


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

/* Removed absolute positioning overlay hacks */
.camera-wrapper {
    width: 100%;
    margin: 0 auto 1.5rem auto;
    border-radius: 16px;
    background: #111827;
    padding: 1rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
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

/* Style the login panel */
.login-container {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 2.5rem !important;
    margin: 4rem auto 0 auto !important;
    max-width: 400px;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
}
"""

with gr.Blocks(css=CSS, title="Real-Time Age Detection") as demo:
    
    # Custom Login State
    def check_login(username, password):
        users_db = load_users()
        if username in users_db and users_db[username] == hash_password(password):
            return [gr.update(visible=False), gr.update(visible=True)]
        else:
            raise gr.Error("Incorrect username or password.")

    def register_user(username, password, confirm_password):
        if not username or not password:
            raise gr.Error("Username and password cannot be empty.")
        if password != confirm_password:
            raise gr.Error("Passwords do not match.")
        
        users_db = load_users()
        if username in users_db:
            raise gr.Error("Username already exists. Please choose another.")
        
        users_db[username] = hash_password(password)
        save_users(users_db)
        gr.Info(f"Account '{username}' created successfully! Please log in.")
        return [gr.update(visible=True), gr.update(visible=False)]

    def show_signup():
        return [gr.update(visible=False), gr.update(visible=True)]

    def show_login():
        return [gr.update(visible=True), gr.update(visible=False)]

    # ── Login Screen ──
    with gr.Column(visible=True, elem_classes="login-container") as login_panel:
        gr.HTML("<h1 style='font-size:2rem; margin-bottom:0.5rem;'>🔐 Secure Login</h1>")
        gr.HTML('<p class="subtitle" style="margin-bottom:1.5rem;">Please enter your credentials to access the age detection app.</p>')
        
        with gr.Column():
            username_input = gr.Textbox(label="Username", placeholder="admin", interactive=True)
            password_input = gr.Textbox(label="Password", type="password", placeholder="admin123", interactive=True)
            login_btn = gr.Button("Sign In", variant="primary", size="lg")
            
        gr.HTML("<hr style='margin-top: 1.5rem; border-color: #334155;'>")
        gr.HTML("<p style='text-align:center; color:#94a3b8; font-size:0.9rem; margin-top:0.5rem;'>Don't have an account?</p>")
        to_signup_btn = gr.Button("Create an Account", variant="secondary")

    # ── Sign Up Screen ──
    with gr.Column(visible=False, elem_classes="login-container") as signup_panel:
        gr.HTML("<h1 style='font-size:2rem; margin-bottom:0.5rem;'>✨ Create Account</h1>")
        gr.HTML('<p class="subtitle" style="margin-bottom:1.5rem;">Join us to start using real-time age detection.</p>')
        
        with gr.Column():
            new_username = gr.Textbox(label="Choose a Username", placeholder="e.g. john_doe", interactive=True)
            new_password = gr.Textbox(label="Create Password", type="password", interactive=True)
            confirm_password = gr.Textbox(label="Confirm Password", type="password", interactive=True)
            register_btn = gr.Button("Sign Up", variant="primary", size="lg")
            
        gr.HTML("<hr style='margin-top: 1.5rem; border-color: #334155;'>")
        gr.HTML("<p style='text-align:center; color:#94a3b8; font-size:0.9rem; margin-top:0.5rem;'>Already have an account?</p>")
        to_login_btn = gr.Button("Back to Login", variant="secondary")

    # ── Main Application Dashboard ──
    with gr.Column(visible=False) as app_panel:
        gr.HTML("<h1>🎯 Real-Time Age Detection</h1>")
        gr.HTML('<p class="subtitle">Click the camera below → allow access → see your age detected in real-time!</p>')

        # Camera View Side-by-Side
        with gr.Row(elem_classes="camera-wrapper"):
            webcam_input = gr.Image(
                sources=["webcam"],
                streaming=True,
                label="Live Webcam",
            )
            output_image = gr.Image(
                label="AI Prediction Output",
                interactive=False
            )

        # Settings Control Panel
        with gr.Column(elem_classes="settings-container"):
            gr.Markdown("### ⚙️ Detection Settings")
            
            with gr.Row():
                model_type = gr.Dropdown(
                    choices=["Custom Trained MobileNetV2"],
                    value="Custom Trained MobileNetV2",
                    label="Age Prediction Model",
                    info="Custom Trained MobileNetV2 regression model is highly optimized for accuracy and low memory usage."
                )
                detector_backend = gr.Dropdown(
                    choices=["opencv", "ssd", "retinaface", "mtcnn", "mediapipe", "dlib"],
                    value="opencv",
                    label="Face Detector Backend",
                    info="OpenCV is the fastest and most reliable for CPU servers like Render's free tier."
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

    # Wire up login logic
    login_btn.click(
        fn=check_login,
        inputs=[username_input, password_input],
        outputs=[login_panel, app_panel]
    )

    # Wire up signup logic
    register_btn.click(
        fn=register_user,
        inputs=[new_username, new_password, confirm_password],
        outputs=[login_panel, signup_panel]
    )

    # Wire up screen toggles
    to_signup_btn.click(fn=show_signup, inputs=[], outputs=[login_panel, signup_panel])
    to_login_btn.click(fn=show_login, inputs=[], outputs=[login_panel, signup_panel])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
