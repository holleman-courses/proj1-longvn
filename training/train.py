import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Define image size and dataset paths
IMG_SIZE = 96
DATASET_PATH = os.path.join(os.path.dirname(__file__), "../data/images/")
LABELS_FILE = os.path.join(os.path.dirname(__file__), "../data/labels.csv")
MODEL_PATH = "models/trained_model.h5"

def load_and_preprocess_data():
    labels_df = pd.read_csv(LABELS_FILE)
    
    images = []
    labels = []

    for _, row in labels_df.iterrows():
        img_name, label = row["filename"], row["label"]
        img_path = os.path.join(DATASET_PATH, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not load {img_name}")
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype("float32") / 255.0 

        images.append(img)
        labels.append(label)

    images = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    labels = np.array(labels)

    return images, labels

X, y = load_and_preprocess_data()

# Split data into training and testing sets (80% train, 20% test)
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.fit(X_train, y_train, epochs=10, batch_size=2, validation_data=(X_test, y_test))

os.makedirs("models", exist_ok=True)
model.save(MODEL_PATH)

print(f"Model saved to {MODEL_PATH}")
