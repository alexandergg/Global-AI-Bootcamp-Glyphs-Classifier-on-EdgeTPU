import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

from settings import config
from sklearn.externals import joblib
from os.path import join, isdir, isfile, dirname, join
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from PIL import Image

file_dir = dirname(__file__)
intermediatePath = join(file_dir, "./preproccessing/models")
imagePath = join(file_dir, "./examples")
featurePath = join(intermediatePath, "features.npy")
labelPath = join(intermediatePath, "labels.npy")
lgPath = join(intermediatePath, "lg.pkl")

if __name__ == '__main__':
	features = np.load(featurePath)
	labels = np.load(labelPath)

	tobeDeleted = np.nonzero(labels == "UNKNOWN")
	features = np.delete(features,tobeDeleted, 0)
	labels = np.delete(labels,tobeDeleted, 0)

	X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, random_state=42)

	print("[INFO] Training Logistic Regression...")
	model = LogisticRegression(solver="lbfgs", multi_class="auto", C=10000)
	model.fit(X_train, y_train)

	print("[INFO] Evaluating...")
	preds = model.predict(X_test)

	print(classification_report(y_test, preds))

	accuracy = np.sum(y_test == preds) / float(len(preds))
	for idx, pred in enumerate(preds):
	    print("%-5s --> %s" % (y_test[idx], pred))
	print("accuracy = {}%".format(accuracy*100))

	print("[INFO] Finished training! saving...")
	joblib.dump(model, lgPath, compress=1)