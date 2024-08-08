import math
import pickle
import torch
import numpy as np
import tifffile as tiff
from torch.utils.data.dataset import random_split
from device import get_default_device


def save_dataset_in_pickle(path, image_datasets):
    with open(path + "image_datasets.pkl", "wb") as handle:
        pickle.dump(image_datasets, handle)


def load_dataset_from_pickle(path):
    with open(path + "images_dataset.pkl", "rb") as handle:
        return pickle.load(handle)


# split the dataset to train, test and validations sets
def split_dataset(image_datasets):
    # first take out the 20% of the dataset for validation
    lengths = [
        math.floor(len(image_datasets) * 0.8),
        math.ceil(len(image_datasets) * 0.2),
    ]
    train_data, val_data = random_split(image_datasets, [21000, 6000])

    # first take out the 20% of the train dataset for validation
    lengths = [math.floor(len(train_data) * 0.8), math.ceil(len(train_data) * 0.2)]
    train_data, test_data = random_split(train_data, [15000, 6000])

    print("Train Length = " + str(len(train_data)))
    print("Validation Length = " + str(len(val_data)))
    print("Test Length = " + str(len(test_data)))

    return train_data, val_data, test_data


def tiff_loader(filename):
  img = tiff.imread(filename)
  img = torch.tensor(img.astype(np.float32), device=get_default_device())
  return img