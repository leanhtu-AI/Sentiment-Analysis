from tensorflow.train import latest_checkpoint
from tensorflow.keras.models import load_model
from create_model import create_model
from transformers import TFAutoModel
from utils.tokenizer import PRETRAINED_MODEL
from utils.variables import df_test, tokenizer
from utils.preprocess_text import preprocess
import numpy as np
from tensorflow.data import Dataset
from utils.preprocess_user_data import preprocess_data
import pandas as pd
import urllib.request
import os
import streamlit as st
@st.cache_resource
def load_model():
    if not os.path.isfile('model.h5'):
        urllib.request.urlretrieve('https://github.com/leanhtu-AI/Sentiment-Analysis/raw/main/model/best.h5', 'model/model.h5')
        pretrained_bert = TFAutoModel.from_pretrained(PRETRAINED_MODEL, output_hidden_states=True)
        reloaded_model = create_model(pretrained_bert)
        reloaded_model.load_weights('model/model.h5')
        return reloaded_model

reloaded_model = load_model()
reloaded_model.summary()

replacements = {0: None, 3: 'positive', 1: 'negative', 2: 'neutral'}
categories = df_test.columns[1:]

def print_acsa_pred(replacements, categories, sentence_pred, confidence_scores):
    sentiments = map(lambda x: replacements[x], sentence_pred)
    results = []
    for category, sentiment, confidence in zip(categories, sentiments, confidence_scores):
        if sentiment: 
            results.append(f'{category},{sentiment},{confidence:.2f}')
    return results if results else None  

def predict_text(model, inputs, batch_size=1, verbose=0):
    y_pred = model.predict(inputs, batch_size=batch_size, verbose=verbose)
    y_pred = y_pred.reshape(len(y_pred), -1, 4)
    return np.argmax(y_pred, axis=-1), np.max(y_pred, axis=-1)  

def show_predict_text(text):
    text = preprocess(text)
    tokenized_input = tokenizer(text, padding='max_length', truncation=True)
    features = {x: [[tokenized_input[x]]] for x in tokenizer.model_input_names}
    pred, confidences = predict_text(reloaded_model, Dataset.from_tensor_slices(features))
    results = []
    for i in range(len(pred)):
        absa_pred = print_acsa_pred(replacements, categories, pred[i], confidences[i])
        if(absa_pred != None):
            for i in range(len(absa_pred)):
                parts = absa_pred[i].split(',')
                positive_value = parts[1]
                confidences_value = float(parts[2])
                if (positive_value == 'positive' and confidences_value >= 0.6):
                    parts.append('⭐️⭐️⭐️⭐️⭐️')
                if (positive_value == 'positive' and  confidences_value < 0.6):
                    parts.append('⭐️⭐️⭐️⭐️')
                elif positive_value == 'neutral':
                    parts.append('⭐️⭐️⭐️')
                elif (positive_value == 'negative'and confidences_value < 0.6):
                    parts.append('⭐️⭐️')
                elif (positive_value == 'negative'and confidences_value >= 0.6):
                    parts.append('⭐️')
                absa_pred[i] = ','.join(parts)
            results.append(absa_pred)
        else:
            results.append(absa_pred)
    return results

def predict_csv(model, df):
    input_sentences = df.iloc[:, 0].tolist()
    tokenized_inputs = tokenizer(input_sentences, padding='max_length', truncation=True)
    features = {x: [tokenized_inputs[x]] for x in tokenizer.model_input_names}
    pred, confidences = predict_text(model, Dataset.from_tensor_slices(features))
    return pred, confidences

def process_predict_csv(df_clean, output_csv_path):
    pred, confidences = predict_csv(reloaded_model, df_clean)
    results = []
    for i in range(len(pred)):
        absa_pred = print_acsa_pred(replacements, categories, pred[i], confidences[i])
        if(absa_pred != None):
            for i in range(len(absa_pred)):
                parts = absa_pred[i].split(',')
                positive_value = parts[1]
                confidences_value = float(parts[2])
                if (positive_value == 'positive' and confidences_value >= 0.6):
                    parts.append('5')
                if (positive_value == 'positive' and  confidences_value < 0.6):
                    parts.append('4')
                elif positive_value == 'neutral':
                    parts.append('3')
                elif (positive_value == 'negative'and confidences_value < 0.6):
                    parts.append('2')
                elif (positive_value == 'negative'and confidences_value >= 0.6):
                    parts.append('1')
                absa_pred[i] = ','.join(parts)
            results.append(absa_pred)
        else:
            results.append(absa_pred)

    df_source = pd.read_csv("data_user/raw.csv")
    df_input_with_pred = df_source.copy()
    df_input_with_pred['label'] = results
    df_input_with_pred.to_csv(output_csv_path, index=False)
    
# final_output_with star file csv
def extract_first_last(text):
    parts = text.split(',')
    first_value = parts[0].split("'")[1]
    last_value = parts[-1].split("'")[0]
    return first_value, last_value

# Function to add icon to average star values
def add_icon_to_avg_star(row):
    avg_star = row['star']
    icon = '⭐'  # Star icon
    return f"{avg_star}{icon}"

def show_predict_csv():
    df = pd.read_csv("./data_user/data_with_label.csv")
    df = df.dropna()
    df[['aspect', 'star']] = df['label'].apply(lambda x: pd.Series(extract_first_last(x)))
    
    drop_col = df.columns[0:2]
    df.drop(columns=drop_col, axis=1)
    
    df['star'] = df['star'].astype('int') 
    avg_by_aspect = df.groupby('aspect')['star'].mean().round(1)
    result_df = avg_by_aspect.reset_index(name='star')
    result_df['star'] = result_df.apply(add_icon_to_avg_star, axis=1)
    return result_df