"""
train.py  –  Entry point to train the Pneumonia Detection CNN
Usage:
    python train.py --data_dir ./data --epochs 30 --batch_size 32
"""

import argparse
import os
from src.model import build_cnn, get_data_generators, train_model, \
                      evaluate_model, plot_training_history


def parse_args():
    parser = argparse.ArgumentParser(description="Train Pneumonia Detection CNN")
    parser.add_argument("--data_dir",   type=str, default="data",
                        help="Path to dataset root (must contain train/val/test)")
    parser.add_argument("--img_size",   type=int, default=224,
                        help="Image size (square)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--dropout",    type=float, default=0.5)
    parser.add_argument("--save_dir",   type=str, default="models",
                        help="Folder to save the best model")
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, "best_model.h5")

    print("\n[1/4] Loading data generators ...")
    train_gen, val_gen, test_gen = get_data_generators(
        data_dir=args.data_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
    )

    print("\n[2/4] Building model ...")
    model = build_cnn(
        input_shape=(args.img_size, args.img_size, 3),
        dropout_rate=args.dropout,
    )
    model.summary()

    print("\n[3/4] Training ...")
    history = train_model(model, train_gen, val_gen,
                          epochs=args.epochs, save_path=model_path)

    print("\n[4/4] Evaluating on test set ...")
    evaluate_model(model, test_gen)
    plot_training_history(history)

    print("\n✓ Done. Model saved at:", model_path)


if __name__ == "__main__":
    main()
