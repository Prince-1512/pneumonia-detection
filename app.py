"""
app.py  –  Flask web app for Pneumonia Detection with Grad-CAM
Run: python app.py
Open: http://localhost:5000
"""

import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow import keras
import tensorflow as tf
from PIL import Image
import base64
import io
import cv2

app = Flask(__name__)

MODEL_PATH = "models/best_model.h5"
model = None


def load_model():
    global model
    if model is None:
        model = keras.models.load_model(MODEL_PATH)
    return model


def preprocess_image(image_bytes, img_size=224):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((img_size, img_size))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def compute_gradcam(model, img_array, layer_name):
    with tf.GradientTape() as tape:
        x = tf.cast(img_array, tf.float32)
        current = x
        conv_output = None
        for layer in model.layers:
            current = layer(current)
            if layer.name == layer_name:
                conv_output = current
                tape.watch(conv_output)
        predictions = current

    grads = tape.gradient(predictions, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def generate_gradcam_image(image_bytes, img_size=224):
    m = load_model()

    # Find last conv layer
    layer_name = None
    for layer in reversed(m.layers):
        if isinstance(layer, keras.layers.Conv2D):
            layer_name = layer.name
            break

    # Preprocess
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_resized = img.resize((img_size, img_size))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array_batch = np.expand_dims(img_array, axis=0)

    # Predict
    prob = float(m.predict(img_array_batch, verbose=0)[0][0])
    label = "PNEUMONIA" if prob >= 0.5 else "NORMAL"
    confidence = prob if prob >= 0.5 else 1 - prob

    # Grad-CAM
    heatmap = compute_gradcam(m, img_array_batch, layer_name)

    # Overlay
    img_cv = np.array(img_resized)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    heatmap_resized = cv2.resize(heatmap, (img_size, img_size))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed = (0.4 * heatmap_colored + 0.6 * img_cv).astype(np.uint8)
    superimposed = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)

    # Encode gradcam to base64
    gradcam_pil = Image.fromarray(superimposed)
    buf = io.BytesIO()
    gradcam_pil.save(buf, format="JPEG")
    gradcam_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return label, round(confidence * 100, 2), gradcam_b64


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        img_bytes = file.read()

        # Original image base64
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        # Grad-CAM
        label, confidence, gradcam_b64 = generate_gradcam_image(img_bytes)

        return jsonify({
            "label": label,
            "confidence": confidence,
            "image": img_b64,
            "gradcam": gradcam_b64
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Loading model...")
    load_model()
    print("Model loaded. Starting server...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)
