import numpy as np
import pickle, itertools, os
import tensorflow as tf

from imutils import paths
from settings import config
from os.path import join, dirname, join
from sklearn.preprocessing import LabelEncoder
from preproccessing.imageLoader import loadImage, get_features

file_dir = dirname(__file__)
intermediatePath = join(file_dir, "./models")
featurePath = join(intermediatePath, "features.npy")
labelPath = join(intermediatePath, "labels.npy")

class FeatureExtractor():
	def __init__(self):
		print("loading DeepNet (Inception-V3) ...")
		self._model = tf.keras.applications.Resnet50(weights="imagenet", include_top=False)
		self._le = None
		self._data = []
		self._labels_list = []
		self._labelsencoder = []

	def get_labels(self, shuffleimages, p):
		self._labelsencoder.append([p.split(os.path.sep)[-1] for p in shuffleimages])
		labels = [p.split(os.path.sep)[-1] for p in shuffleimages]
		return labels
	
	def create_LabelEncoder(self):
		flatten = list(itertools.chain(*self._labelsencoder))
		if self._le is None:
			self._le = LabelEncoder()
			self._le.fit(flatten)
			
		f = open(config.LE_PATH, "wb")
		f.write(pickle.dumps(self._le))
		f.close()

	def create_batch(self, imagePaths, labels):
		for (b, i) in enumerate(range(0, len(imagePaths), config.BATCH_SIZE)):
			print("[INFO] processing batch {}/{}".format(b + 1,
				int(np.ceil(len(imagePaths) / float(config.BATCH_SIZE)))))
			batchPaths = imagePaths[i:i + config.BATCH_SIZE]
			batchLabels = self._le.transform(labels[i:i + config.BATCH_SIZE])

			for label in batchLabels:
				self._labels_list.append(label)
			
			batchImages = []

			for imagePath in batchPaths:
				image = loadImage(imagePath)
				batchImages.append(image)

			batchImages = np.vstack(batchImages)
			features_norm = get_features(self._model, batchImages)
			self._data.append(features_norm)

		lb = np.asarray(self._labels_list)
		ftr = np.vstack(self._data)

		np.save(featurePath, ftr)
		np.save(labelPath, lb)
