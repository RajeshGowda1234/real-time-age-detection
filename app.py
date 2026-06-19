import os
import gradio as gr
import cv2
import numpy as np
from deepface import DeepFace

# ── Color palette ──────────────────────────────────────────────────
BOX_COLOR = (0, 255, 128)    # bright green
TEXT_COLOR = (255, 255, 255)  # white
BG_COLOR = (0, 180, 90)       # label background


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
            enforce_detection=False,  # don't crash if no face found
            detector_backend="opencv",  # fast CPU detector
            silent=True,
        )

        # results can be a list (multiple faces) or a single dict
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
        pass  # no face detected — return original frame

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ── Gradio UI ────────────────────────────────────────
