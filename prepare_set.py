import numpy as np
import pandas as pd
import os
import cv2

import util_functions as uf
from util_variables import *

masks_df = pd.read_csv('datasets/airbus-ship-detection/train_ship_segmentations_v2.csv')

def resize_mask(row):  # decodes and resizes masks to side_size by side_size

    if row["EncodedPixels"] is np.NaN:
        return np.zeros((side_size, side_size))

    mask = row["EncodedPixels"]

    np_seg = uf.rle_decode(mask)

    return cv2.resize(np_seg, dsize=(side_size, side_size), interpolation=cv2.INTER_CUBIC)


def load_resize_image(row):  # loads images into the dataframe
    image = cv2.imread(os.path.join(x_train_dir, row["ImageId"]))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dsize=(side_size, side_size), interpolation=cv2.INTER_CUBIC)

    return image


masks_df_slice = pd.concat([masks_df.dropna(), masks_df[masks_df.isna().any(axis=1)][:10000]])

masks_df_slice["decoded_mask"] = masks_df_slice.swifter.apply(resize_mask, axis = 1)

grouped = masks_df_slice.drop("EncodedPixels", axis = 1).groupby("ImageId").sum().reset_index()

grouped["Area"] = grouped["decoded_mask"].apply(lambda x: x.sum())
grouped_slice = grouped[(grouped["Area"] > 6) | (grouped["Area"] == 0)]

grouped_slice[("gr_image")] = grouped_slice.swifter.apply(load_resize_image, axis = 1)

uf.visualize(img = grouped_slice.loc[745, "gr_image"], mask = grouped_slice.loc[745, "decoded_mask"])  # demonstraiting a sample from dataset



# splitting data into train and test and saving it as a numpy array


xy_train = grouped_slice[["gr_image", "decoded_mask"]].sample(frac=0.8,random_state=200) # split dataset into train and test sets, split is 80/20
xy_test = grouped_slice[["gr_image", "decoded_mask"]].drop(xy_train.index) # remove train part from test partition

x_train = xy_train.drop("decoded_mask", axis = 1)
x_test = xy_test.drop("decoded_mask", axis = 1)

y_train = xy_train[["decoded_mask"]]
y_test = xy_test[["decoded_mask"]]


y_train_numpy = np.stack(y_train[["decoded_mask"]].values.flatten()) # transform dataframe of numpy arrays into 3d numpy array
y_train_numpy = y_train_numpy.reshape(*y_train_numpy.shape, 1) # transform it into 4d array for model compatability
y_train_numpy[y_train_numpy > 1] = 1 # due to the procces of merging all of the masks, some overlap, creating values above 1

x_train_numpy = np.stack(x_train[["gr_image"]].values.flatten()) # transform dataframe of numpy arrays into 4d numpy array

y_test_numpy = np.stack(y_test[["decoded_mask"]].values.flatten()) # transform dataframe of numpy arrays into 3d numpy array
y_test_numpy = y_test_numpy.reshape(*y_test_numpy.shape, 1) # transform it into 4d array for model compatability
y_test_numpy[y_test_numpy > 1] = 1 # due to the procces of merging all of the masks, some overlap, creating values above 1

x_test_numpy = np.stack(x_test[["gr_image"]].values.flatten()) # transform dataframe of numpy arrays into 4d numpy array

# Augmenting the images

import imgaug as ia
import imgaug.augmenters as iaa

aug_seq = iaa.Sequential([
    iaa.imgcorruptlike.Fog(severity=(1,3))
], random_order=False)

x_train_numpy_aug = aug_seq(images = x_train_numpy) # adding fog to images to imitate clouds

# fliping images vertically and horisontaly

x_train_numpy_vr = x_train_numpy[:, :, ::-1, :]
y_train_numpy_vr = y_train_numpy[:, :, ::-1, :]
x_train_numpy_hr = x_train_numpy[:, ::-1, :, :]
y_train_numpy_hr = y_train_numpy[:, ::-1, :, :]
x_train_numpy_hr_vr = x_train_numpy[:, ::-1, ::-1, :]
y_train_numpy_hr_vr = y_train_numpy[:, ::-1, ::-1, :]



# slicing the datasets
# the split is:
# 100% of original images                  ~ 50% of resulting training set
# 33% of augmented                         ~ 15% of resulting training set
# 25% of verticaly flipped                 ~ 12% of resulting training set
# 25% of horisontaly flipped               ~ 12% of resulting training set
# 25% of verticaly and horisontaly flipped ~ 12% of resulting training set

x_train_numpy_final = np.concatenate((
    x_train_numpy,
    x_train_numpy_aug[2::3],
    x_train_numpy_vr[2::4],
    x_train_numpy_hr[3::4],
    x_train_numpy_hr_vr[1::4]
))
y_train_numpy_final = np.concatenate((
    y_train_numpy,
    y_train_numpy[2::3],
    y_train_numpy_vr[2::4],
    y_train_numpy_hr[3::4],
    y_train_numpy_hr_vr[1::4]
))

try:
    os.mkdir("fitting_data")
except:
    pass

np.save("fitting_data/x_train_numpy_final", x_train_numpy_final)
np.save("fitting_data/y_train_numpy_final", y_train_numpy_final)
np.save("fitting_data/x_test_numpy", x_test_numpy)
np.save("fitting_data/y_test_numpy", y_test_numpy)

print(f"Done creating datasets, resulting split:\nTrain size: {len(x_train_numpy_final)}\nTest size: {len(x_test_numpy)}")
