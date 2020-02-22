import os

ORIG_INPUT_DATASET = "GlyphsDataset"

BASE_PATH = "dataset"

TRAIN = "training"
TEST = "evaluation"
VAL = "validation"

BATCH_SIZE = 32

LE_PATH = os.path.sep.join(["output", "le.cpickle"])
BASE_CSV_PATH = "output"