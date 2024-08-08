
import io
import zipfile
import requests
import numpy as np
from PIL import Image
from keras.utils import np_utils
from sklearn.model_selection import train_test_split


def load_dataset(url):
    # download zip
    request = requests.get(url)
    zipFile = zipfile.ZipFile(io.BytesIO(request.content))

    # get file names
    files = [file for file in zipFile.namelist()]

    # keep only those containing ".jpg"
    files = [f for f in files if ".jpg" in f]

    # read images to numpy array
    images = np.zeros([len(files), 64, 64, 3])

    for (idx, image) in enumerate(files):
        images[idx] = np.asarray(Image.open(zipFile.open(image))).astype('uint8')/255
   
    # clear memory
    del zipFile
    
    labels = np.empty(len(files), dtype = 'S20')
    
    for (idx,label) in enumerate(files):
        labels[idx] = label.split('/')[1]
    
    _, y_labels = np.unique(labels, return_inverse=True)
    
    return images, y_labels


def split_dataset(images,labels,num_classes):
    X_train, X_test, y_train, y_test = train_test_split(images, labels, stratify = labels, train_size = 0.5, random_state=42)
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)
    return X_train, X_test, y_train, y_test
