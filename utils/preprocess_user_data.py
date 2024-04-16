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
    text_length = df.map(lambda x: len(str(x))).mean()
    review_column = text_length.idxmax()

    # Lọc và lưu cột đánh giá
    filtered_df = df[[review_column]]
    filtered_df = filtered_df.dropna()

    filtered_df.to_csv(output_path, index=False)

def preprocess_data(df):
    for column in df.columns:
        df[column] = df[column].apply(preprocess)
    return df

