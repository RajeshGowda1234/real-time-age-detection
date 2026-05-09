import cv2

class FaceDetector:
    def __init__(self):
        # Load the pre-trained Haar Cascade classifier for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def detect_faces(self, frame, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        """
        Detects faces in a given BGR frame.
        Returns a list of bounding boxes (x, y, w, h)
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray_frame, 
            scaleFactor=scale_factor, 
            minNeighbors=min_neighbors, 
            minSize=min_size, 
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces

    def crop_face(self, frame, bbox, margin=20):
        """
        Crops the face from the frame with a given margin.
        """
        x, y, w, h = bbox
        
        # Add margin
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)
        
        return frame[y1:y2, x1:x2]
