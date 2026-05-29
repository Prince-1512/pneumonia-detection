"""
transfer_learning.py  –  Fine-tune VGG16 for Pneumonia Detection

Use this if you want better accuracy than the custom CNN.
ImageNet pretrained weights give a head start on medical images.

Usage:
    python transfer_learning.py --data_dir ./data --epochs 20
"""

import argparse
import os
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from src.model import get_data_generators, evaluate_model, plot_training_history


def build_vgg_transfer(img_size=224, dropout=0.5):
    """
    VGG16 base (frozen) + custom classification head.
    After initial training, we unfreeze last few blocks for fine-tuning.
    """
    base = keras.applications.VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(img_size, img_size, 3),
    )
    # Freeze pretrained layers initially
    base.trainable = False

    model = keras.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(dropout),
        layers.Dense(128, activation="relu"),
        layers.Dropout(dropout / 2),
        layers.Dense(1, activation="sigmoid"),
    ], name="VGG16_Pneumonia")

    return model, base


def fine_tune(model, base_model, unfreeze_from_layer=15):
    """Unfreeze some VGG16 layers for fine-tuning with a lower LR."""
    base_model.trainable = True
    for layer in base_model.layers[:unfreeze_from_layer]:
        layer.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")],
    )
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="data")
    parser.add_argument("--img_size",   type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs",     type=int, default=20)
    parser.add_argument("--finetune_epochs", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs("models", exist_ok=True)

    train_gen, val_gen, test_gen = get_data_generators(
        args.data_dir, (args.img_size, args.img_size), args.batch_size
    )

    # Phase 1: Train with frozen base
    print("\n[Phase 1] Training head with frozen VGG16 base ...")
    model, base_model = build_vgg_transfer(args.img_size)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")],
    )
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3),
        ModelCheckpoint("models/vgg16_phase1.h5", save_best_only=True,
                        monitor="val_accuracy", verbose=1),
    ]
    history1 = model.fit(train_gen, validation_data=val_gen,
                         epochs=args.epochs, callbacks=callbacks)

    # Phase 2: Fine-tune
    print("\n[Phase 2] Fine-tuning VGG16 last blocks ...")
    model = fine_tune(model, base_model)
    callbacks[2] = ModelCheckpoint("models/vgg16_finetuned.h5",
                                   save_best_only=True,
                                   monitor="val_accuracy", verbose=1)
    history2 = model.fit(train_gen, validation_data=val_gen,
                         epochs=args.finetune_epochs, callbacks=callbacks)

    print("\n[Evaluation]")
    evaluate_model(model, test_gen, results_dir="results/vgg16")
    plot_training_history(history2, results_dir="results/vgg16")

    print("\n✓ VGG16 fine-tuning complete.")


if __name__ == "__main__":
    main()
