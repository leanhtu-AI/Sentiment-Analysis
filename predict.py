from tensorflow.train import latest_checkpoint
from tensorflow.keras.models import load_model
from create_model import create_model
from utils.config import MODEL_PATH
from transformers import TFAutoModel
from utils.tokenizer import PRETRAINED_MODEL
from utils.variables import df_test, tokenizer
from utils.preprocess_text import preprocess
import numpy as np
import pandas as pd
from tensorflow.data import Dataset

pretrained_bert = TFAutoModel.from_pretrained(PRETRAINED_MODEL, output_hidden_states=True)
reloaded_model = create_model(pretrained_bert)
reloaded_model.load_weights(f"{MODEL_PATH}/anhtu1.h5")

replacements = {0: None, 3: 'positive', 1: 'negative', 2: 'neutral'}
categories = df_test.columns[1:]

def print_acsa_pred(replacements, categories, sentence_pred):
    sentiments = map(lambda x: replacements[x], sentence_pred)
    results = []
    for category, sentiment in zip(categories, sentiments):
        if sentiment: 
            results.append(f'{category},{sentiment}')
    return results
def predict_text(model, inputs, batch_size=1, verbose=0):
    y_pred = model.predict(inputs, batch_size=batch_size, verbose=verbose)
    y_pred = y_pred.reshape(len(y_pred), -1, 4)
    return np.argmax(y_pred, axis=-1) # sentiment values (position that have max value)

def show_predict_text(text):
    tokenized_input = tokenizer(text, padding='max_length', truncation=True)
    features = {x: [[tokenized_input[x]]] for x in tokenizer.model_input_names}
    pred = predict_text(reloaded_model, Dataset.from_tensor_slices(features))
    results = print_acsa_pred(replacements, categories, pred[0])
    return results


# # Đọc dữ liệu từ file CSV đã được xử lý
# csv_file_path = "data_user/cleandata.csv"  # Thay đổi đường dẫn đến file CSV của bạn
# df_input = pd.read_csv(csv_file_path)
# # print(df_input.iloc[:,0])
# # Tiền xử lý dữ liệu đầu vào
# input_sentences = df_input.iloc[:,0].tolist()
# tokenized_inputs = tokenizer(input_sentences, padding='max_length', truncation=True)
# features = {x: [tokenized_inputs[x]] for x in tokenizer.model_input_names}

# # Dự đoán cảm xúc cho dữ liệu đầu vào
# pred = predict(reloaded_model, Dataset.from_tensor_slices(features))

# for i in range(len(pred)):
#     print_acsa_pred(replacements,categories, pred[i])
# # In DataFrame sau khi ghép kết quả dự đoán
# # print(df_input)
