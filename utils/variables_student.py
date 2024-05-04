from datasets import load_dataset
from tokenizer import call_tokenizer, tokenize_function,read_csv, make_outputs
from config import BATCH_SIZE, TRAIN_PATH_STU, TEST_PATH_STU, VAL_PATH_STU, MODEL_PATH
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.train import CheckpointOptions
import pandas as pd
from tf_format import preprocess_tokenized_dataset

raw_datasets_stu = load_dataset('csv', data_files={'train': TRAIN_PATH_STU, 'val': VAL_PATH_STU, 'test': TEST_PATH_STU})

STEPS_PER_EPOCH_STU = len(raw_datasets_stu['train']) // BATCH_SIZE

VALIDATION_STEPS_STU = len(raw_datasets_stu['val']) // BATCH_SIZE

# You should plus 1 if have residual after // steps

df_train_stu = read_csv(TRAIN_PATH_STU)
df_val_stu = read_csv(VAL_PATH_STU)
df_test_stu = read_csv(TEST_PATH_STU)

y_train_stu = make_outputs(df_train_stu)
y_val_stu = make_outputs(df_val_stu) 
y_test_stu = make_outputs(df_test_stu)

tokenizer = call_tokenizer()

tokenized_datasets = raw_datasets_stu.map(tokenize_function, batched=True)

train_stu_dataset = preprocess_tokenized_dataset(tokenized_datasets['train'], tokenizer, y_train_stu, BATCH_SIZE, shuffle=True)
val_stu_dataset = preprocess_tokenized_dataset(tokenized_datasets['val'], tokenizer, y_val_stu, BATCH_SIZE)
test_stu_dataset = preprocess_tokenized_dataset(tokenized_datasets['test'],  tokenizer, y_test_stu, BATCH_SIZE)

early_stopping = EarlyStopping(monitor='val_loss',patience=1, verbose=1)
checkpoint_filepath = MODEL_PATH + '/best_v2.h5'

# Define the ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',  # Monitor validation loss
    save_weights_only=True,
    save_best_only=True,  # Save only the best model
    mode='min',  # Mode for the monitored quantity: minimize validation loss
    verbose=1
)