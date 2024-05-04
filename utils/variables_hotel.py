from config import BATCH_SIZE, TRAIN_PATH_HOTEL, TEST_PATH_HOTEL, VAL_PATH_HOTEL, MODEL_PATH
from datasets import load_dataset
from tokenizer import call_tokenizer, tokenize_function_hotel,read_csv, make_outputs
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.train import CheckpointOptions
import pandas as pd
from tf_format import preprocess_tokenized_dataset

raw_datasets_hotel = load_dataset('csv', data_files={'train': TRAIN_PATH_HOTEL, 'val': VAL_PATH_HOTEL, 'test': TEST_PATH_HOTEL})

STEPS_PER_EPOCH_HOTEL = len(raw_datasets_hotel['train']) // BATCH_SIZE

VALIDATION_STEPS_HOTEL = len(raw_datasets_hotel['val']) // BATCH_SIZE

# You should plus 1 if have residual after // steps

df_train_hotel = read_csv(TRAIN_PATH_HOTEL)
df_val_hotel = read_csv(VAL_PATH_HOTEL)
df_test_hotel = read_csv(TEST_PATH_HOTEL)

y_train_hotel = make_outputs(df_train_hotel)
y_val_hotel = make_outputs(df_val_hotel) 
y_test_hotel = make_outputs(df_test_hotel)

tokenizer = call_tokenizer()

tokenized_datasets = raw_datasets_hotel.map(tokenize_function_hotel, batched=True)

train_hotel_dataset = preprocess_tokenized_dataset(tokenized_datasets['train'], tokenizer, y_train_hotel, BATCH_SIZE, shuffle=True)
val_hotel_dataset = preprocess_tokenized_dataset(tokenized_datasets['val'], tokenizer, y_val_hotel, BATCH_SIZE)
test_hotel_dataset = preprocess_tokenized_dataset(tokenized_datasets['test'],  tokenizer, y_test_hotel, BATCH_SIZE)

early_stopping = EarlyStopping(monitor='val_loss',patience=1, verbose=1)
checkpoint_filepath = MODEL_PATH + '/hotel.h5'

# Define the ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',  # Monitor validation loss
    save_weights_only=True,
    save_best_only=True,  # Save only the best model
    mode='min',  # Mode for the monitored quantity: minimize validation loss
    verbose=1
)