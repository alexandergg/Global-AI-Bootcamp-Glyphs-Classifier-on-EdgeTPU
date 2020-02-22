import os
import random

from settings import config
from imutils import paths
from preproccessing.extract_features import FeatureExtractor

if __name__ == '__main__':
    extractor = FeatureExtractor()
    p = os.path.sep.join([config.BASE_PATH, config.TRAIN])
    imagePaths = list(paths.list_images(p))
    random.shuffle(imagePaths)
    labels = extractor.get_labels(imagePaths, p)
    extractor.create_LabelEncoder()
    extractor.create_batch(imagePaths, labels)