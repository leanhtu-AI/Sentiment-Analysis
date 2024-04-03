from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.train import CheckpointOptions
import os

MODEL_PATH = './model'

MAX_SEQUENCE_LENGTH = 256

BATCH_SIZE = 21

EPOCHS = 10

