import tensorflow as tf
import keras
import numpy as np
import sys

import util_functions as uf
from util_variables import *

epoch_num = sys.argv[1] if len(sys.argv) >= 1 else 5
model_checkpoint = sys.argv[2] if len(sys.argv) >= 3 else None
fitting_directory = sys.argv[3] if len(sys.argv) >= 4 else "fitting_data"

if model_checkpoint is None:
    # Defining the U-net model
    segmentation_model = keras.models.Sequential([

        # input and rescaling
        keras.layers.Input(shape=(side_size, side_size, 3)),
        keras.layers.Rescaling(1. / 255),

        # convolution layers
        keras.layers.Conv2D(64, 3, padding="same", activation="relu", strides=2),
        keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        keras.layers.Conv2D(128, 3, padding="same", activation="relu", strides=2),
        keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
        keras.layers.Dropout(0.01),

        # transpose layers
        keras.layers.Conv2DTranspose(128, 3, padding="same", activation="relu"),
        keras.layers.Conv2DTranspose(128, 3, padding="same", activation="relu", strides=2),
        keras.layers.Conv2DTranspose(64, 3, padding="same", activation="relu"),
        keras.layers.Conv2DTranspose(64, 3, padding="same", activation="relu", strides=2),
        keras.layers.Dropout(0.01),

        # output layer
        keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid")
    ])

    # compile the model
    segmentation_model.compile(
        optimizer="rmsprop",
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryIoU(),
            keras.metrics.BinaryAccuracy()
        ]
    )
else:
    segmentation_model = keras.models.load_model(model_checkpoint)

print(segmentation_model.summary())

x_train_numpy_final = np.load(f"{fitting_directory}/x_train_numpy_final.npy")
y_train_numpy_final = np.load(f"{fitting_directory}/y_train_numpy_final.npy")

x_test_numpy = np.load(f"{fitting_directory}/x_test_numpy.npy")
y_test_numpy = np.load(f"{fitting_directory}/y_test_numpy.npy")

callbacks = [
    keras.callbacks.ModelCheckpoint("models/best_model.keras", save_best_only=True)
]

history = segmentation_model.fit(x_train_numpy_final, y_train_numpy_final, epochs=3, batch_size=128, validation_data=(x_test_numpy, y_test_numpy), callbacks=callbacks)

segmentation_model.save("models/segmentation_model.keras")
