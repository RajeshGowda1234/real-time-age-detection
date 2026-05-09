import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from src.age_model import build_model
import glob

# Custom data generator for UTKFace dataset
# UTKFace format: [age]_[gender]_[race]_[date&time].jpg
class UTKFaceDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, batch_size=32, target_size=(224, 224), shuffle=True):
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_paths = [self.image_paths[k] for k in indexes]
        X, y = self.__data_generation(batch_paths)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_paths):
        X = np.empty((self.batch_size, *self.target_size, 3))
        y = np.empty((self.batch_size), dtype=int)

        for i, path in enumerate(batch_paths):
            # Read and preprocess image
            img = tf.keras.preprocessing.image.load_img(path, target_size=self.target_size)
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
            
            # Extract age from filename
            filename = os.path.basename(path)
            age = int(filename.split('_')[0])
            
            X[i,] = img
            y[i] = age

        return X, y

def train(dataset_dir, epochs=10, batch_size=32):
    image_paths = glob.glob(os.path.join(dataset_dir, '*.jpg'))
    if not image_paths:
        print(f"No images found in {dataset_dir}. Please download the UTKFace dataset.")
        return

    np.random.shuffle(image_paths)
    split = int(0.8 * len(image_paths))
    train_paths = image_paths[:split]
    val_paths = image_paths[split:]

    train_gen = UTKFaceDataGenerator(train_paths, batch_size=batch_size)
    val_gen = UTKFaceDataGenerator(val_paths, batch_size=batch_size)

    model = build_model()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_absolute_error',
        metrics=['mae']
    )

    os.makedirs('models', exist_ok=True)
    
    callbacks = [
        ModelCheckpoint('models/best_age_model.h5', monitor='val_mae', mode='min', save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ]

    print("Starting training...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks
    )
    print("Training complete. Model saved to models/best_age_model.h5")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dataset/UTKFace', help='Path to UTKFace dataset folder')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    args = parser.parse_args()
    
    train(args.dataset, args.epochs)
