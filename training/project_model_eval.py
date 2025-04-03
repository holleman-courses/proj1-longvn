import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns

train_path = "../data_set/train"
val_path = "../data_set/val"

train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_path,
    target_size=(96, 96),
    color_mode='grayscale',
    class_mode='binary',
    batch_size=32
)

val_data = val_gen.flow_from_directory(
    val_path,
    target_size=(96, 96),
    color_mode='grayscale',
    class_mode='binary',
    batch_size=32
)

def build_model():
    model = Sequential([
        Conv2D(4, (3,3), activation='relu', input_shape=(96,96,1)),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Conv2D(8, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Conv2D(16, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_model()
model.fit(train_data, validation_data=val_data, epochs=12)

model.save("trained_model.h5")

# Evaluate
val_data.reset()
y_true = val_data.classes
y_pred = (model.predict(val_data) > 0.5).astype(int).flatten()
cm = confusion_matrix(y_true, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Not Screwdriver", "Screwdriver"],
            yticklabels=["Not Screwdriver", "Screwdriver"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
