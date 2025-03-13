import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Define paths
IMG_SIZE = 96
DATASET_PATH = os.path.join(os.path.dirname(__file__), "../data/images/")
LABELS_FILE = os.path.join(os.path.dirname(__file__), "../data/labels.csv")
MODEL_PATH = "models/trained_model.h5"

def load_test_data():
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

    # Split into test set (80% 20%)
    split_idx = int(0.8 * len(images))
    X_test, y_test = images[split_idx:], labels[split_idx:]

    return X_test, y_test

X_test, y_test = load_test_data()

model = tf.keras.models.load_model(MODEL_PATH)

y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred, zero_division=1)

print(f"Model Evaluation Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
