import numpy as np
import os
import tensorflow as tf
import shutil

from settings import config
from imutils import paths
from multiprocessing.pool import Pool
from sklearn.preprocessing import normalize

def build_dataset():
    for split in (config.TRAIN, config.TEST, config.VAL):
        print("[INFO] processing '{} split'...".format(split))
        p = os.path.sep.join([config.ORIG_INPUT_DATASET, split])
        imagePaths = list(paths.list_images(p))

        for imagePath in imagePaths:
            filename = imagePath.split(os.path.sep)[-1]
            label = imagePath[(imagePath.rfind("_") + 1): imagePath.rfind(".")]

            dirPath = os.path.sep.join([config.BASE_PATH, split, label])

            if not os.path.exists(dirPath):
                    os.makedirs(dirPath)

            p = os.path.sep.join([dirPath, filename])
            shutil.copy2(imagePath, p)

def loadImage(path):
    image = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))
    x = tf.keras.preprocessing.image.img_to_array(image)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    return x

def loadBatch(img_paths):
    with Pool(processes=8) as pool:
        imgs = pool.map(loadImage, img_paths)
        return np.vstack(imgs)

def get_features(model, batch):
    features =  model.predict(batch)
    features = features.reshape((features.shape[0], 7 * 7 * 2048)) # Resnet 7 * 7 * 2048 | VGG16 7 * 7 * 512
    return normalize(features, axis=1, norm='l2')

def batchGenerator(img_paths, labels, batch_size):
    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i:(i + batch_size)]
        batch_labels = labels[i:(i + batch_size)]
        batch_images = loadBatch(batch_paths)
        yield batch_images, batch_labels
