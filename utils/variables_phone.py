# Standard library imports
import os

# Local imports
from config import BATCH_SIZE, MODEL_PATH, TEST_PATH_PHONE, TRAIN_PATH_PHONE, VAL_PATH_PHONE

# Third-party imports
from datasets import load_dataset
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tf_format import preprocess_tokenized_dataset
from tokenizer import call_tokenizer, make_outputs, read_csv, tokenize_function

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

raw_datasets_phone = load_dataset('csv', data_files={'train': TRAIN_PATH_PHONE, 'val': VAL_PATH_PHONE, 'test': TEST_PATH_PHONE})

STEPS_PER_EPOCH_PHONE = len(raw_datasets_phone['train']) // BATCH_SIZE

VALIDATION_STEPS_PHONE = len(raw_datasets_phone['val']) // BATCH_SIZE

# You should plus 1 if have residual after // steps

df_train_phone = read_csv(TRAIN_PATH_PHONE)
df_val_phone = read_csv(VAL_PATH_PHONE)
df_test_phone = read_csv(TEST_PATH_PHONE)

y_train_phone = make_outputs(df_train_phone)
y_val_phone = make_outputs(df_val_phone) 
y_test_phone = make_outputs(df_test_phone)

tokenizer = call_tokenizer()

tokenized_datasets = raw_datasets_phone.map(tokenize_function, batched=True)

train_phone_dataset = preprocess_tokenized_dataset(tokenized_datasets['train'], tokenizer, y_train_phone, BATCH_SIZE, shuffle=True)
val_phone_dataset = preprocess_tokenized_dataset(tokenized_datasets['val'], tokenizer, y_val_phone, BATCH_SIZE)
test_phone_dataset = preprocess_tokenized_dataset(tokenized_datasets['test'],  tokenizer, y_test_phone, BATCH_SIZE)

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