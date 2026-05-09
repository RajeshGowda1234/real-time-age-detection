import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import numpy as np
import cv2

def build_model(input_shape=(224, 224, 3)):
    """
    Builds a MobileNetV2-based model for exact age prediction (regression).
    """
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
        
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    # Single output neuron with linear activation for regression
    predictions = Dense(1, activation='linear')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

class AgePredictor:
    def __init__(self, model_path=None):
        self.input_shape = (224, 224)
        if model_path:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Loaded model from {model_path}")
        else:
            self.model = build_model()
            print("Initialized untrained model.")
            
    def preprocess_image(self, face_img):
        """
        Preprocesses the cropped face image for MobileNetV2 input.
        """
        img = cv2.resize(face_img, self.input_shape)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        img = np.expand_dims(img, axis=0)
        return img
        
    def predict(self, face_img):
        """
        Predicts the exact age for a preprocessed face image.
        Returns a single scalar value.
        """
        processed_img = self.preprocess_image(face_img)
        # Model returns shape (1, 1), we extract the scalar
        age = self.model.predict(processed_img, verbose=0)[0][0]
        return age
        
    def get_label(self, age):
        """
        Returns the formatted label for the predicted age.
        """
        return f"Age: {int(round(age))}"
