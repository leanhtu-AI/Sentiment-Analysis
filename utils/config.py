MAX_SEQUENCE_LENGTH = 256

BATCH_SIZE = 20

# you shoud set batch = 20 if you have gpu or training on cloud by colab, kaggle, etc. else batch = 5

EPOCHS = 5

MODEL_PATH = 'model'
# # With OTHER
# TRAIN_PATH = "./data/Train.csv"
# VAL_PATH = "./data/Val.csv"
# TEST_PATH = "./data/Test.csv"

# Without OTHERS
TRAIN_PATH = "./data/train.csv"
VAL_PATH = "./data/val.csv"
TEST_PATH = "./data/test.csv"

# # # For testing traning with small data when you don't have gpu or training on cloud
# TRAIN_PATH = "./data/train_small.csv"
# VAL_PATH = "./data/val_small.csv"
# TEST_PATH = "./data/test_small.csv"

DROP_OUT = 0.2

LEARNING_RATE=1e-4

WEIGHT_DECAY =5e-3