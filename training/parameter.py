import tensorflow as tf

MODEL_PATH = "models/trained_model.h5"

model = tf.keras.models.load_model(MODEL_PATH)

model.summary()
