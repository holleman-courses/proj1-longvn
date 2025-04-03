#include <Wire.h>
#include <Arduino.h>
#include <TinyMLShield.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "trained_model.h"

#define LED_PIN 13
#define CAMERA_WIDTH 176
#define CAMERA_HEIGHT 144
#define RESIZED_WIDTH 96
#define RESIZED_HEIGHT 96
#define DETECTION_THRESHOLD 0.49f

constexpr int kTensorArenaSize = 100 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

int8_t image_data[RESIZED_WIDTH * RESIZED_HEIGHT];

tflite::MicroErrorReporter micro_error_reporter;
tflite::AllOpsResolver resolver;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

float input_scale = 0.0f;
int input_zero_point = 0;
float output_scale = 0.0f;
int output_zero_point = 0;

void setup() {
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  Serial.begin(115200);
  while (!Serial);
  Serial.println("Booting up...");
  Serial.println("Trying to init camera...");

  if (!Camera.begin(QCIF, GRAYSCALE, 1, OV7675)) {
    Serial.println("Failed to init camera");
    while (1);
  }
  Serial.println("Camera initialized");

  model = tflite::GetModel((const void*)trained_model);
  interpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter);

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Tensor allocation failed");
    while (1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  input_scale = input->params.scale;
  input_zero_point = input->params.zero_point;
  output_scale = output->params.scale;
  output_zero_point = output->params.zero_point;

  Serial.println("System Ready - Running continuous inference");
}

void loop() {
  captureAndInfer();
  delay(2000); // Every 2 seconds
}

void captureAndInfer() {
  byte camera_data[CAMERA_WIDTH * CAMERA_HEIGHT];
  Camera.readFrame(camera_data);

  int index = 0;
  int start_x = (CAMERA_WIDTH - RESIZED_WIDTH) / 2;
  int start_y = (CAMERA_HEIGHT - RESIZED_HEIGHT) / 2;

  for (int y = start_y; y < start_y + RESIZED_HEIGHT; y++) {
    for (int x = start_x; x < start_x + RESIZED_WIDTH; x++) {
      image_data[index++] = (int8_t)(camera_data[y * CAMERA_WIDTH + x] - 128);
    }
  }

  for (int i = 0; i < RESIZED_WIDTH * RESIZED_HEIGHT; i++) {
    input->data.int8[i] = image_data[i];
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Inference failed");
    return;
  }

  float prediction = (output->data.int8[0] - output_zero_point) * output_scale;

  Serial.print("Screwdriver score: ");
  Serial.println(prediction);
  Serial.print("Not screwdriver score: ");
  Serial.println(1.0 - prediction);

  if (prediction > DETECTION_THRESHOLD) {
    Serial.println("Prediction: Screwdriver");
    digitalWrite(LED_PIN, HIGH);
  } else {
    Serial.println("Prediction: Not Screwdriver");
    digitalWrite(LED_PIN, LOW);
  }

  Serial.println("-----------------------------");
}