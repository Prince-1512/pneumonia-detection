"""
Pneumonia Detection using CNN
Author: Prince Ranjan
Description: Custom CNN architecture for binary classification
             of chest X-ray images (Normal vs Pneumonia)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)


# ──────────────────────────────────────────────
# 1.  Build Model
# ──────────────────────────────────────────────

def build_cnn(input_shape=(224, 224, 3), dropout_rate=0.5):
    """
    Custom CNN with 4 convolutional blocks + dense head.
    Kept relatively small so it trains well even without a GPU.
    """
    model = keras.Sequential(name="PneumoniaCNN")

    # Block 1 – learn low-level edges / textures
    model.add(layers.Conv2D(32, (3, 3), padding="same",
                            activation="relu",
                            input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 2
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 3 – mid-level features
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 4
    model.add(layers.Conv2D(256, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Classifier head
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation="relu",
                           kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(dropout_rate / 2))
    model.add(layers.Dense(1, activation="sigmoid"))   # binary output

    return model


# ──────────────────────────────────────────────
# 2.  Data Loaders
# ──────────────────────────────────────────────

def get_data_generators(data_dir, img_size=(224, 224), batch_size=32):
    """
    Returns train / val / test generators with augmentation on train only.
    Expects folder layout:
        data_dir/
            train/  NORMAL/  PNEUMONIA/
            val/    NORMAL/  PNEUMONIA/
            test/   NORMAL/  PNEUMONIA/
    """
    train_aug = keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    val_aug = keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_aug.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True,
    )
    val_gen = val_aug.flow_from_directory(
        os.path.join(data_dir, "val"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )
    test_gen = val_aug.flow_from_directory(
        os.path.join(data_dir, "test"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )

    return train_gen, val_gen, test_gen


# ──────────────────────────────────────────────
# 3.  Training
# ──────────────────────────────────────────────

def train_model(model, train_gen, val_gen, epochs=30, save_path="models/best_model.h5"):
    """Compile and train; returns history object."""

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall"),
                 keras.metrics.AUC(name="auc")],
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=7,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                          patience=3, min_lr=1e-7, verbose=1),
        ModelCheckpoint(save_path, monitor="val_accuracy",
                        save_best_only=True, verbose=1),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
    )
    return history


# ──────────────────────────────────────────────
# 4.  Evaluation helpers
# ──────────────────────────────────────────────

def evaluate_model(model, test_gen, results_dir="results"):
    """Print metrics and save confusion matrix + ROC plots."""
    os.makedirs(results_dir, exist_ok=True)

    y_true = test_gen.classes
    y_prob = model.predict(test_gen, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    labels = list(test_gen.class_indices.keys())   # ['NORMAL', 'PNEUMONIA']

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=labels))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved → {results_dir}/confusion_matrix.png")


def plot_training_history(history, results_dir="results"):
    """Save accuracy and loss curves."""
    os.makedirs(results_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    axes[0].plot(history.history["accuracy"], label="Train Acc")
    axes[0].plot(history.history["val_accuracy"], label="Val Acc")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    # Loss
    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "training_curves.png"), dpi=150)
    plt.close(fig)
    print(f"Training curves saved → {results_dir}/training_curves.png")
