import pandas as pd
from utils.preprocess_text import preprocess
import os
TF_ENABLE_ONEDNN_OPTS='0'
import tensorflow
from tensorflow.data import Dataset
from utils.tokenizer import call_tokenizer
def auto_detect_filter_data(input_path, output_path):
    # Đọc dữ liệu từ file vào một DataFrame
    df = pd.read_csv(input_path)
    
    # Giả định cột đánh giá là cột văn bản có độ dài trung bình cao nhất
    text_length = df.applymap(lambda x: len(str(x))).mean()
    review_column = text_length.idxmax()

    # Lọc và lưu cột đánh giá
    filtered_df = df[[review_column]]
    filtered_df = filtered_df.dropna()

    filtered_df.to_csv(output_path, index=False)

def preprocess_data(df):
    for column in df.columns:
        df[column] = df[column].apply(preprocess)
    return df

DATA_PATH = "data_user/cleandata.csv"

def convert_to_token(DATA_PATH):
    read = pd.read_csv(DATA_PATH, index_col=None)
    print(read.head())
    tokenizer = call_tokenizer()
    for column in read.columns:
        for row in read[column]:
            tokenized_inputs = tokenizer(row, max_length=256, truncation=True, padding='max_length', return_tensors="tf")

# convert_to_token(DATA_PATH)

def print_acsa_pred(replacements, categories, sentence_pred):
    sentiments = map(lambda x: replacements[x], sentence_pred)
    for category, sentiment in zip(categories, sentiments):
        if sentiment: print(f'=> {category},{sentiment}')
        
def predict(model, inputs, batch_size=1, verbose=0):
    y_pred = model.predict(inputs, batch_size=batch_size, verbose=verbose)
    y_pred = y_pred.reshape(len(y_pred), -1, 4)
    return np.argmax(y_pred, axis=-1) # sentiment values (position that have max value)
