"""
gradcam.py  –  Grad-CAM heatmap visualizer (Keras 3 compatible)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import os


def compute_gradcam(model, img_array, layer_name):
    """Grad-CAM using tf.GradientTape — Keras 3 compatible."""

    # Build a sub-model using functional approach
    conv_layer = model.get_layer(layer_name)

    # Get intermediate output by running a forward pass with tape
    with tf.GradientTape() as tape:
        # Convert to tensor and watch it
        x = tf.cast(img_array, tf.float32)

        # We need to get conv layer output manually
        # Run model layer by layer
        current = x
        conv_output = None
        for layer in model.layers:
            current = layer(current)
            if layer.name == layer_name:
                conv_output = current
                tape.watch(conv_output)

        predictions = current  # final output

    # Compute gradients of prediction w.r.t conv output
    grads = tape.gradient(predictions, conv_output)

    # Pool gradients over spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight conv output channels by pooled grads
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def overlay_heatmap(original_img_path, heatmap, alpha=0.4):
    img = cv2.imread(original_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    superimposed = (alpha * heatmap_colored + (1 - alpha) * img).astype(np.uint8)
    return img, heatmap_resized, superimposed


def visualize(model_path, image_path, img_size=224, save_path="results/gradcam.png"):
    os.makedirs("results", exist_ok=True)

    model = keras.models.load_model(model_path)

    # Find last conv layer
    layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            layer_name = layer.name
            break
    print(f"Using layer: {layer_name}")

    # Preprocess image
    img = Image.open(image_path).convert("RGB").resize((img_size, img_size))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prob = float(model.predict(img_array, verbose=0)[0][0])
    label = "PNEUMONIA" if prob >= 0.5 else "NORMAL"
    conf  = prob if prob >= 0.5 else 1 - prob
    print(f"Prediction: {label} ({conf*100:.1f}% confidence)")

    # Grad-CAM
    heatmap = compute_gradcam(model, img_array, layer_name)
    original, hm_raw, superimposed = overlay_heatmap(image_path, heatmap)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(original)
    axes[0].set_title("Original X-ray")
    axes[0].axis("off")

    axes[1].imshow(hm_raw, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(superimposed)
    axes[2].set_title(f"Overlay\n{label} ({conf*100:.1f}%)")
    axes[2].axis("off")

    fig.suptitle(f"Grad-CAM Visualization  |  Prediction: {label}", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Grad-CAM saved → {save_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",    required=True)
    parser.add_argument("--model",    default="models/best_model.h5")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--output",   default="results/gradcam.png")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualize(args.model, args.image, args.img_size, args.output)