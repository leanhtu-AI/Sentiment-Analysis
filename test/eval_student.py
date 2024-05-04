import sys
# adding Folder_2 to the system path
sys.path.insert(0, 'utils/')
from tensorflow.keras.models import load_model
from create_model import create_model_stu
from transformers import TFAutoModel
from tokenizer import PRETRAINED_MODEL
from variables_student import df_test_stu, test_stu_dataset
from config import MODEL_PATH, BATCH_SIZE
import numpy as np

replacements = {0: None, 3: 'positive', 1: 'negative', 2: 'neutral'}
categories = df_test_stu.columns[1:]

# Predict on test dataset
def print_acsa_pred_test(replacements, categories, sentence_pred):
    sentiments = map(lambda x: replacements[x], sentence_pred)
    for category, sentiment in zip(categories, sentiments):
        if sentiment: print(f'=> {category},{sentiment}')
def predict_test(model, inputs, batch_size=1, verbose=0):
    y_pred = model.predict(inputs, batch_size=batch_size, verbose=verbose)
    y_pred = y_pred.reshape(len(y_pred), -1, 4)
    return np.argmax(y_pred, axis=-1) # sentiment values (position that have max value)

pretrained_bert = TFAutoModel.from_pretrained(PRETRAINED_MODEL, output_hidden_states=True)
reloaded_model = create_model_stu(pretrained_bert)
reloaded_model.load_weights(f'{MODEL_PATH}/student.h5')
y_pred = predict_test(reloaded_model, test_stu_dataset, BATCH_SIZE, verbose=1)
reloaded_model.evaluate(test_stu_dataset  , batch_size=BATCH_SIZE, verbose=1)

print_acsa_pred_test(replacements, categories, y_pred[0])


