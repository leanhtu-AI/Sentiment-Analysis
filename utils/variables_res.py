# Standard library imports
import os

# Local imports
from config import BATCH_SIZE, MODEL_PATH, TEST_PATH_RES, TRAIN_PATH_RES, VAL_PATH_RES

# Third-party imports
from datasets import load_dataset
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tf_format import preprocess_tokenized_dataset
from tokenizer import call_tokenizer, make_outputs, read_csv, tokenize_function

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

raw_datasets_res = load_dataset('csv', data_files={'train': TRAIN_PATH_RES, 'val': VAL_PATH_RES, 'test': TEST_PATH_RES})

STEPS_PER_EPOCH_RES = len(raw_datasets_res['train']) // BATCH_SIZE

VALIDATION_STEPS_RES = len(raw_datasets_res['val']) // BATCH_SIZE

# You should plus 1 if have residual after // steps

df_train_res = read_csv(TRAIN_PATH_RES)
df_val_res = read_csv(VAL_PATH_RES)
df_test_res = read_csv(TEST_PATH_RES)

y_train_res = make_outputs(df_train_res)
y_val_res = make_outputs(df_val_res) 
y_test_res= make_outputs(df_test_res)

tokenizer = call_tokenizer()

tokenized_datasets = raw_datasets_res.map(tokenize_function, batched=True)

train_res_dataset = preprocess_tokenized_dataset(tokenized_datasets['train'], tokenizer, y_train_res, BATCH_SIZE, shuffle=True)
val_res_dataset = preprocess_tokenized_dataset(tokenized_datasets['val'], tokenizer, y_val_res, BATCH_SIZE)
test_res_dataset = preprocess_tokenized_dataset(tokenized_datasets['test'],  tokenizer, y_test_res, BATCH_SIZE)

early_stopping = EarlyStopping(monitor='val_loss',patience=1, verbose=1)
checkpoint_filepath = MODEL_PATH + '/res.h5'

# Define the ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',  # Monitor validation loss
    save_weights_only=True,
    save_best_only=True,  # Save only the best model
    mode='min',  # Mode for the monitored quantity: minimize validation loss
    verbose=1
)