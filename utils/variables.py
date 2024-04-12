from datasets import load_dataset
from utils.tokenizer import call_tokenizer, tokenize_function,read_csv, make_outputs
from utils.tf_format import to_tensorflow_format, preprocess_tokenized_dataset
from utils.config import BATCH_SIZE, MODEL_PATH, TRAIN_PATH, TEST_PATH, VAL_PATH
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.train import CheckpointOptions
import pandas as pd

raw_datasets = load_dataset('csv', data_files={'train': TRAIN_PATH, 'val': VAL_PATH, 'test': TEST_PATH})

STEPS_PER_EPOCH = len(raw_datasets['train']) // BATCH_SIZE

VALIDATION_STEPS = len(raw_datasets['val']) // BATCH_SIZE

df_train = read_csv(TRAIN_PATH)
df_val = read_csv(VAL_PATH)
df_test = read_csv(TEST_PATH)

y_train = make_outputs(df_train)
y_val = make_outputs(df_val) 
y_test = make_outputs(df_test)

tokenizer = call_tokenizer()

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

early_stopping = EarlyStopping(monitor='val_loss',patience=1, verbose=1)
checkpoint_filepath = MODEL_PATH + '/best.h5'

# Define the ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',  # Monitor validation loss
    save_weights_only=True,
    save_best_only=True,  # Save only the best model
    mode='min',  # Mode for the monitored quantity: minimize validation loss
    verbose=1
)