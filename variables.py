from datasets import load_dataset
from utils.tokenizer import call_tokenizer, tokenize_function,read_csv, make_outputs
from utils.tf_format import to_tensorflow_format, preprocess_tokenized_dataset
from config import BATCH_SIZE, MODEL_PATH

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.train import CheckpointOptions

TRAIN_PATH = "./data/Train_v2.csv"
VAL_PATH = "./data/Val_v2.csv"
TEST_PATH = "./data/Test_v2.csv"

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

train_tf_dataset = preprocess_tokenized_dataset(tokenized_datasets['train'], tokenizer, y_train, BATCH_SIZE, shuffle=True)
val_tf_dataset = preprocess_tokenized_dataset(tokenized_datasets['val'], tokenizer, y_val, BATCH_SIZE)
test_tf_dataset = preprocess_tokenized_dataset(tokenized_datasets['test'],  tokenizer, y_test, BATCH_SIZE)
print(train_tf_dataset)

# Stop if no improvement after 5 epochs
early_stop_callback = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
checkpoint_path = MODEL_PATH + '/checkpoints/cp-{epoch:03d}.keras'
checkpoint_dir = os.path.dirname(checkpoint_path)
# Create a callback that saves the model's weights every 10 epochs
checkpoint_callback = ModelCheckpoint(
    filepath = checkpoint_path,
    save_best_only = True,
    save_weights_only = False,
    save_freq = 10 * STEPS_PER_EPOCH,
    verbose = 1
)