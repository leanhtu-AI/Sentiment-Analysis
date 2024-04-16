MAX_SEQUENCE_LENGTH = 256

BATCH_SIZE = 20

# you shoud set batch = 20 if you have gpu or training on cloud by colab, kaggle, etc.

EPOCHS = 5

MODEL_PATH = 'model'
# # With OTHER
# TRAIN_PATH = "./data/Train.csv"
# VAL_PATH = "./data/Val.csv"
# TEST_PATH = "./data/Test.csv"

# Without OTHERS
TRAIN_PATH = "./data/Train_without_OTHER.csv"
VAL_PATH = "./data/Val_without_OTHER.csv"
TEST_PATH = "./data/Test_without_OTHER.csv"

# # For testing traning with small data when you don't have gpu or training on cloud
# TRAIN_PATH = "./data/train.csv"
# VAL_PATH = "./data/val.csv"
# TEST_PATH = "./data/test.csv"

DROP_OUT = 0.2

LEARNING_RATE=1e-4

WEIGHT_DECAY =5e-3