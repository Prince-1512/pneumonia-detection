"""
predict.py  –  Run inference on a single chest X-ray image

Usage:
    python predict.py --image path/to/xray.jpg --model models/best_model.h5
"""

import argparse
import numpy as np
from tensorflow import keras
from PIL import Image


LABELS = {0: "NORMAL", 1: "PNEUMONIA"}
COLORS = {0: "\033[92m", 1: "\033[91m"}   # green / red for terminal
RESET  = "\033[0m"


def load_and_preprocess(image_path: str, img_size: int = 224) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((img_size, img_size))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # shape: (1, H, W, 3)


def predict(model_path: str, image_path: str, img_size: int = 224):
    model = keras.models.load_model(model_path)

    x = load_and_preprocess(image_path, img_size)
    prob = float(model.predict(x, verbose=0)[0][0])
    label_idx = int(prob >= 0.5)

    label      = LABELS[label_idx]
    confidence = prob if label_idx == 1 else 1.0 - prob
    color      = COLORS[label_idx]

    print(f"\nImage  : {image_path}")
    print(f"Result : {color}{label}{RESET}")
    print(f"Confidence : {confidence * 100:.2f}%")
    print(f"Raw probability (Pneumonia) : {prob:.4f}\n")

    return label, confidence


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",    required=True,  help="Path to chest X-ray image")
    parser.add_argument("--model",    default="models/best_model.h5")
    parser.add_argument("--img_size", type=int, default=224)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(args.model, args.image, args.img_size)
