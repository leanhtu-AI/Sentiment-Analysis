MAX_SEQUENCE_LENGTH = 256

BATCH_SIZE = 5

# you shoud set batch = 20 if you have gpu or training on cloud by colab, kaggle, etc. else batch = 5

EPOCHS = 1

MODEL_PATH = 'model'

TRAIN_PATH_PHONE = "./data/phone/train.csv"
VAL_PATH_PHONE = "./data/phone/val.csv"
TEST_PATH_PHONE = "./data/phone/test.csv"

# TRAIN_PATH_RES = "./data/res/train.csv"
# VAL_PATH_RES= "./data/res/val.csv"
# TEST_PATH_RES = "./data/res/test.csv"

# For testing traning with small data when you don't have gpu or training on cloud
# TRAIN_PATH_PHONE = "./data/phone/train_small.csv"
# VAL_PATH_PHONE = "./data/phone/val_small.csv"
# TEST_PATH_PHONE = "./data/phone/test_small.csv"

TRAIN_PATH_RES = "./data/res/train_small.csv"
VAL_PATH_RES= "./data/res/val_small.csv"
TEST_PATH_RES = "./data/res/test_small.csv"

DROP_OUT = 0.2

LEARNING_RATE=1e-4

WEIGHT_DECAY =5e-3