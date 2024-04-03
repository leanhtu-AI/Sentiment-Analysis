from transformers import AutoTokenizer
from utils.preprocess_text import preprocess
import pandas as pd
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

PRETRAINED_MODEL = 'vinai/phobert-base'

def read_csv(file_path):
    dataset = pd.read_csv(file_path)
    return dataset

def make_outputs(df):
    outputs = []
    for row in range(len(df)):
        row_one_hot = []
        for col in range(1, len(df.columns)):
            sentiment = df.iloc[row, col]
            if   sentiment == 0: one_hot = [1, 0, 0, 0]
            elif sentiment == 1: one_hot = [0, 1, 0, 0]
            elif sentiment == 2: one_hot = [0, 0, 1, 0]
            elif sentiment == 3: one_hot = [0, 0, 0, 1]
            row_one_hot.append(one_hot)
        outputs.append(row_one_hot)
    return np.array(outputs, dtype='uint8')    

def call_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    return tokenizer

def tokenize_function(examples):
    tokenizer = call_tokenizer()
    clean_texts = [preprocess(comment) for comment in examples['comment']]
    tokenized_inputs = tokenizer(clean_texts, max_length=256, truncation=True, padding='max_length', return_tensors="tf")

    return tokenized_inputs