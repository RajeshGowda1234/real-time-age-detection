import cv2
import numpy as np
from collections import deque

class PredictionSmoother:
    def __init__(self, history_size=5):
        # We store history of predictions to avoid flickering
        self.history = deque(maxlen=history_size)
    
    def add_prediction(self, probs):
        self.history.append(probs)
        
    def get_smoothed_prediction(self):
        if not self.history:
            return None
        # Average probabilities across history
        avg_probs = np.mean(self.history, axis=0)
        return avg_probs

def draw_label(img, text, pos, bg_color=(0, 255, 0), text_color=(255, 255, 255)):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    color = text_color
    thickness = cv2.FILLED
    margin = 2

    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)

def draw_bounding_box(frame, bbox, label, color=(0, 255, 0)):
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    draw_label(frame, label, (x, y - 5), bg_color=color)
