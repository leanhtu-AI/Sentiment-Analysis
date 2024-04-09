from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.train import CheckpointOptions
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

MODEL_PATH = 'model'

MAX_SEQUENCE_LENGTH = 256

BATCH_SIZE = 20

EPOCHS = 5
