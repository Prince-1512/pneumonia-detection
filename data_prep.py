"""
data_prep.py  –  Dataset exploration and class-imbalance analysis

Run this before training to understand your data.

Usage:
    python data_prep.py --data_dir ./data
"""

import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def count_images(data_dir):
    """Print class distribution per split."""
    splits = ["train", "val", "test"]
    classes = ["NORMAL", "PNEUMONIA"]

    print(f"\n{'Split':<10} {'NORMAL':>10} {'PNEUMONIA':>12} {'Total':>8}")
    print("-" * 44)

    totals = {"NORMAL": 0, "PNEUMONIA": 0}
    for split in splits:
        counts = {}
        for cls in classes:
            path = os.path.join(data_dir, split, cls)
            if os.path.exists(path):
                n = len([f for f in os.listdir(path)
                         if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            else:
                n = 0
            counts[cls] = n
            totals[cls] += n
        total = sum(counts.values())
        print(f"{split:<10} {counts['NORMAL']:>10} {counts['PNEUMONIA']:>12} {total:>8}")

    total_all = sum(totals.values())
    print("-" * 44)
    print(f"{'ALL':<10} {totals['NORMAL']:>10} {totals['PNEUMONIA']:>12} {total_all:>8}")
    print(f"\nClass ratio  NORMAL : PNEUMONIA = "
          f"1 : {totals['PNEUMONIA'] / max(totals['NORMAL'], 1):.2f}")


def sample_images(data_dir, n=3, save_path="results/sample_images.png"):
    """Save a grid of sample X-rays from each class."""
    os.makedirs("results", exist_ok=True)
    classes = ["NORMAL", "PNEUMONIA"]
    fig, axes = plt.subplots(2, n, figsize=(n * 4, 8))

    for row, cls in enumerate(classes):
        folder = os.path.join(data_dir, "train", cls)
        if not os.path.exists(folder):
            print(f"  Folder not found: {folder}")
            continue
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        chosen = random.sample(files, min(n, len(files)))
        for col, fname in enumerate(chosen):
            img = Image.open(os.path.join(folder, fname)).convert("RGB")
            axes[row, col].imshow(img, cmap="gray")
            axes[row, col].set_title(cls, fontsize=11, color="green" if cls == "NORMAL" else "red")
            axes[row, col].axis("off")

    fig.suptitle("Sample Chest X-rays: Normal vs Pneumonia", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Sample images saved → {save_path}")


def compute_mean_std(data_dir, sample_size=500):
    """Estimate per-channel mean and std from training images."""
    folder_normal    = os.path.join(data_dir, "train", "NORMAL")
    folder_pneumonia = os.path.join(data_dir, "train", "PNEUMONIA")

    all_files = []
    for folder in [folder_normal, folder_pneumonia]:
        if os.path.exists(folder):
            all_files += [os.path.join(folder, f) for f in os.listdir(folder)
                          if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not all_files:
        print("No images found for mean/std computation.")
        return

    sampled = random.sample(all_files, min(sample_size, len(all_files)))
    pixels = []
    for path in sampled:
        arr = np.array(Image.open(path).convert("RGB").resize((224, 224)),
                       dtype=np.float32) / 255.0
        pixels.append(arr.reshape(-1, 3))

    pixels = np.vstack(pixels)
    print(f"\nDataset stats (from {len(sampled)} images):")
    print(f"  Mean : R={pixels[:,0].mean():.4f}  G={pixels[:,1].mean():.4f}  B={pixels[:,2].mean():.4f}")
    print(f"  Std  : R={pixels[:,0].std():.4f}  G={pixels[:,1].std():.4f}  B={pixels[:,2].std():.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    count_images(args.data_dir)
    sample_images(args.data_dir)
    compute_mean_std(args.data_dir)
