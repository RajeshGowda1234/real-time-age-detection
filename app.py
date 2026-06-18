import gradio as gr
import cv2
import numpy as np
from deepface import DeepFace

# ── Color palette ──────────────────────────────────────────────────
BOX_COLOR   = (0, 255, 128)   # bright green
TEXT_COLOR  = (255, 255, 255) # white
BG_COLOR    = (0, 180, 90)    # label background

def detect_age_in_frame(frame):
    """
    Receives an RGB numpy array from the Gradio webcam,
    runs face detection + age estimation, draws overlays,
    and returns the annotated RGB frame.
    """
    if frame is None:
        return None

    # DeepFace expects BGR
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    try:
        results = DeepFace.analyze(
            bgr,
            actions=["age"],
            enforce_detection=False,   # don't crash if no face found
            detector_backend="opencv", # fast CPU detector
            silent=True,
        )

        # results can be a list (multiple faces) or a single dict
        if isinstance(results, dict):
            results = [results]

        for res in results:
            region = res.get("region", {})
            age    = res.get("age", None)

            x = region.get("x", 0)
            y = region.get("y", 0)
            w = region.get("w", 0)
            h = region.get("h", 0)

            if w == 0 or h == 0 or age is None:
                continue

            age_text = f"Age: {int(age)}"

            # Bounding box
            cv2.rectangle(bgr, (x, y), (x + w, y + h), BOX_COLOR, 2)

            # Label background
            (tw, th), _ = cv2.getTextSize(age_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(bgr, (x, y - th - 12), (x + tw + 10, y), BG_COLOR, -1)

            # Label text
            cv2.putText(
                bgr, age_text,
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, TEXT_COLOR, 2, cv2.LINE_AA,
            )

    except Exception:
        pass   # no face detected — return original frame

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ── Gradio UI ──────────────────────────────────────────────────────
CSS = """
body { background: #0f1117; font-family: 'Inter', sans-serif; }
h1   { text-align: center; color: #00ff88; font-size: 2rem; margin-bottom: 0.2rem; }
.subtitle { text-align: center; color: #aaa; margin-bottom: 1.5rem; font-size: 1rem; }
.gr-button-primary { background: #00ff88 !important; color: #000 !important;
                     font-weight: bold; border-radius: 8px; }
footer { display: none !important; }
"""

with gr.Blocks(css=CSS, title="Real-Time Age Detection") as demo:

    gr.HTML("<h1>🎯 Real-Time Age Detection</h1>")
    gr.HTML('<p class="subtitle">Allow camera access → your age appears instantly on screen</p>')

    with gr.Row(equal_height=True):
        with gr.Column():
            gr.Markdown("### 📷 Live Camera")
            webcam_input = gr.Image(
                sources=["webcam"],
                streaming=True,
                label="Camera Feed",
                mirror_webcam=True,
            )
        with gr.Column():
            gr.Markdown("### 🔍 Detection Output")
            output_image = gr.Image(
                label="Age Detection",
                streaming=True,
            )

    gr.HTML("""
    <div style='text-align:center; color:#666; margin-top:1rem; font-size:0.85rem;'>
        Powered by DeepFace · OpenCV · Gradio &nbsp;|&nbsp; No data is stored
    </div>
    """)

    # Wire webcam stream → process → output
    webcam_input.stream(
        fn=detect_age_in_frame,
        inputs=webcam_input,
        outputs=output_image,
        stream_every=0.1,   # ~10 fps processing
    )

if __name__ == "__main__":
    demo.launch()
