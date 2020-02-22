import numpy as np
import os,sys, joblib, pickle
import tensorflow as tf
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from settings import config
from os.path import join, isdir, dirname, join
from os import listdir
from preproccessing.imageLoader import loadBatch, get_features, loadImage

file_dir = dirname(__file__)
intermediatePath = join(file_dir, "./preproccessing/models")
imagePath = join(file_dir, "./examples")
featurePath = join(intermediatePath, "features.npy")
labelPath = join(intermediatePath, "labels.npy")
svmPath = join(intermediatePath, "lg.pkl")

if __name__ == '__main__':
    le = pickle.loads(open(config.LE_PATH, "rb").read())

    if isdir(imagePath):
        imagePaths = [join(imagePath, f) for f in listdir(imagePath) if f.endswith(('.png', '.jpg'))]
    else:
        imagePaths = [imagePath,]
        
    print("[INFO] Loading images...")
    Images = loadBatch(imagePaths)
    print("[INFO] Loading Logistic Regression model...")
    clf = joblib.load(svmPath)
        
    print("[INFO] Extracting features, this may take a while for large collections of images...")
    extractor = tf.keras.applications.ResNet50(weights="imagenet", include_top=False)
    features  = get_features(extractor, Images)

    classes = clf.best_estimator_.classes_ if hasattr(clf, "best_estimator_") else clf.classes_
    print("[INFO] Predicting the Hieroglyph type...")
    result = []
    prob = np.array(clf.predict(features))

    for i in prob:
        result.append(le.classes_[i])

    fig = plt.figure(figsize=(40, 40))
    df = pd.read_csv("./dataset/gardiner_sign_list.csv", sep=';')

    for i, path in enumerate(imagePaths):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(5,5,i+1)
        row = df.loc[df['Gardiner'] == result[i]]
        description = row['Description'].values
        plt.imshow(img)
        plt.title('{}->{}'.format(result[i], description))
        plt.xticks([])
        plt.yticks([])
    plt.show()