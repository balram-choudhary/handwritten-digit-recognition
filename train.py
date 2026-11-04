import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "mnist_cnn.keras"
HISTORY_PATH = ROOT / "training_history.png"


def build_model():
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(28, 28, 1)),
            keras.layers.Conv2D(32, 3, activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 3, activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.25),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.25),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser(description="Train a CNN on the Keras MNIST dataset.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5).")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32")[..., np.newaxis] / 255.0
    x_test = x_test.astype("float32")[..., np.newaxis] / 255.0

    model = build_model()
    history = model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=128,
        validation_split=0.1,
        verbose=2,
    )
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    model.save(MODEL_PATH)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.2%}")
    print(f"Saved model: {MODEL_PATH}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["accuracy"], label="Training")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history.history["loss"], label="Training")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(HISTORY_PATH, dpi=150)
    print(f"Saved training curves: {HISTORY_PATH}")


if __name__ == "__main__":
    main()
