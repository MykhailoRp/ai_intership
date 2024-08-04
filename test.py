import keras
import numpy as np
import sys
import cv2

import util_functions as uf

image_show = sys.argv[1] if len(sys.argv) >= 2 else 5
model_checkpoint = sys.argv[2] if len(sys.argv) >= 3 else "models/best_model.keras"
fitting_directory = sys.argv[3] if len(sys.argv) >= 4 else "fitting_data"

segmentation_model = keras.models.load_model(model_checkpoint)

test_image = 0

x_test_numpy = np.load(f"{fitting_directory}/x_test_numpy.npy")
y_test_numpy = np.load(f"{fitting_directory}/y_test_numpy.npy")

uf.visualize(img = x_test_numpy[test_image], mask = y_test_numpy[test_image], predicted = segmentation_model.predict(x_test_numpy[test_image].reshape(1, 128, 128, 3))[0])

eval_res = segmentation_model.evaluate(x_test_numpy, y_test_numpy)

print(f"Eval results:\n{eval_res}")

all_prediction = segmentation_model.predict(x_test_numpy)

# see overal model results

uf.visualize_random_rows(
    image_show,
    x_test_numpy,
    y_test_numpy,
    all_prediction
)

# test to see if horisontal and vertical flipping impacts the prediction

slice_num = 20

uf.visualize(
    norm = segmentation_model.predict(x_test_numpy[slice_num].reshape(1,128,128,3))[0],
    fliped_ver = cv2.flip(segmentation_model.predict(cv2.flip(x_test_numpy[slice_num], 1).reshape(1,128,128,3))[0], 1),
    fliped_hor = cv2.flip(segmentation_model.predict(cv2.flip(x_test_numpy[slice_num], 0).reshape(1,128,128,3))[0], 0),
    fliped_hor_ver = cv2.flip(segmentation_model.predict(cv2.flip(x_test_numpy[slice_num], -1).reshape(1,128,128,3))[0], -1),
)
