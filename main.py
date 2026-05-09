import cv2
import time
import argparse
import os
from src.face_detection import FaceDetector
from src.age_model import AgePredictor
from src.utils import PredictionSmoother, draw_bounding_box

def main():
    parser = argparse.ArgumentParser(description="Real-Time Human Age Detection")
    parser.add_layer = parser.add_argument('--model', type=str, default=None, help='Path to trained h5 model')
    args = parser.parse_args()

    # Initialize modules
    face_detector = FaceDetector()
    
    model_path = args.model if args.model and os.path.exists(args.model) else None
    if not model_path:
        print("Warning: No trained model provided. The model will output random predictions.")
    
    age_predictor = AgePredictor(model_path=model_path)
    
    # Dictionary to hold smoothers for different faces
    # For a robust multi-face tracker, we'd need a tracking algorithm (e.g., SORT)
    # Here we'll simplify by matching boxes based on IOU or simple distance 
    # to maintain smoothness across frames.
    
    # Start video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Frame processing optimizations
    process_this_frame = True
    prev_time = 0
    fps = 0
    
    print("Starting webcam feed... Press 'q' to quit.")

    # Setup Full Screen Window
    cv2.namedWindow("Real-Time Age Detection", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Real-Time Age Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # A simple way to keep smoothers. Just clear if face count changes drastically.
    smoothers = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Optimization: Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        if process_this_frame:
            faces = face_detector.detect_faces(small_frame)
            
            # Adjust face count for smoothers
            if len(faces) != len(smoothers):
                # Increased history size to 60 for better smoothing at high FPS
                smoothers = [PredictionSmoother(history_size=60) for _ in range(len(faces))]

            predictions = []
            for i, face_rect in enumerate(faces):
                x, y, w, h = face_rect
                
                # Scale back up to original frame size
                x *= 2
                y *= 2
                w *= 2
                h *= 2
                bbox = (x, y, w, h)
                
                # Crop and predict
                face_crop = face_detector.crop_face(frame, bbox)
                if face_crop.size > 0:
                    probs = age_predictor.predict(face_crop)
                    if i < len(smoothers):
                        smoothers[i].add_prediction(probs)
                        smoothed_probs = smoothers[i].get_smoothed_prediction()
                        label = age_predictor.get_label(smoothed_probs)
                        predictions.append((bbox, label))

        # Draw predictions on frame
        for bbox, label in predictions:
            draw_bounding_box(frame, bbox, label)

        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time
        
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Skip alternate frames for face detection to save CPU
        process_this_frame = not process_this_frame

        cv2.imshow("Real-Time Age Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 27 is the ESC key
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
