import tensorflow as tf
import numpy as np
import subprocess

def representative_data_gen():
    for _ in range(100):
        dummy = np.random.rand(1, 96, 96, 1).astype(np.float32)
        yield [dummy]

model = tf.keras.models.load_model("trained_model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
with open("trained_model.tflite", "wb") as f:
    f.write(tflite_model)
