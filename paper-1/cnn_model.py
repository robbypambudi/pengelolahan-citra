"""Customised CNN of the paper (Table 2, Fig. 4)."""
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
from keras import layers


def build_cnn(width=96, height=96, depth=3, classes=7) -> keras.Model:
    return keras.Sequential([
        keras.Input(shape=(height, width, depth)),

        # CONV(32) -> RELU -> BN -> POOL(3x3): 96x96 -> 32x32
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(3, 3)),
        layers.Dropout(0.25),

        # (CONV(64) -> RELU -> BN) * 2 -> POOL(2x2)
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # (CONV(128) -> RELU -> BN) * 2 -> POOL(2x2)
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # FC(1024) -> RELU -> BN -> DROPOUT -> softmax
        layers.Flatten(),
        layers.Dense(1024, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(classes, activation="softmax"),
    ], name="customised_cnn")


if __name__ == "__main__":
    build_cnn().summary()
