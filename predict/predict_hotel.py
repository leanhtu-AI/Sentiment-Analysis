from tensorflow.train import latest_checkpoint
from tensorflow.keras.models import load_model
from utils.create_model import create_model_hotel
from transformers import TFAutoModel
from utils.tokenizer import PRETRAINED_MODEL
from utils.variables_hotel import df_test_hotel, tokenizer
from utils.preprocess_text import preprocess
import numpy as np
from tensorflow.data import Dataset
from utils.preprocess_user_data import preprocess_data
import pandas as pd
import urllib.request
import os
import streamlit as st

# web services
# @st.cache_resource
# def load_model():
#     if not os.path.isfile('model.h5'):
#         urllib.request.urlretrieve('https://github.com/NguyenHuyHoangCome/steamlit/raw/main/model/best4_16_4_4.h5','model.h5')
#         pretrained_bert = TFAutoModel.from_pretrained(PRETRAINED_MODEL, output_hidden_states=True)
#         reloaded_model = create_model(pretrained_bert)
#         reloaded_model.load_weights('model.h5')
#         return reloaded_model

# reloaded_model = load_model()

# local -
pretrained_bert = TFAutoModel.from_pretrained(PRETRAINED_MODEL, output_hidden_states=True)
reloaded_model = create_model_hotel(pretrained_bert)
reloaded_model.load_weights('model/hotel.h5')

replacements = {0: None, 1: 'positive', 2: 'negative', 3: 'neutral'}
categories = df_test_hotel.columns[1:]

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
    tokenized_input = tokenizer(text, max_length=256,padding='max_length', truncation=True)
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
    tokenized_inputs = tokenizer(input_sentences,max_length=256, padding='max_length', truncation=True)
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
    
def add_icon_to_avg_star(row):
    avg_star = row['star']
    if avg_star == 5:
        return str(avg_star) + '⭐⭐⭐⭐⭐'
    elif avg_star >= 4:
        return str(avg_star) + '⭐⭐⭐⭐'
    elif avg_star >=3:
        return str(avg_star) + '⭐⭐⭐'
    elif avg_star >=2:
        return str(avg_star) + '⭐⭐'
    else:
        return str(avg_star) + '⭐'
    
def extract_first_last(text):
    # Split the text and strip unwanted spaces or characters from each part
    parts = [part.strip() for part in text.split(',')]
    first_value = parts[0]  # Get the first element
    last_value = parts[-1]  # Get the last element
    return first_value, last_value

def process_cell(cell):
    # Remove the square brackets and extra quotes, then strip any additional unwanted whitespace or characters
    cleaned = cell.strip("[]' ").replace("'", "")  # Also remove single quotes globally
    # Handle escape sequences like tabs '\t'
    cleaned = cleaned.replace("\t", " ").strip()  # Replace tabs with spaces and strip again
    # Split the cleaned cell into entries
    entries = cleaned.split(", ")
    results = [extract_first_last(entry) for entry in entries]
    return results

def show_predict_csv():
    # Load the data
    df = pd.read_csv("data_user/data_with_label.csv")
    df = df.dropna()

    # Apply the process_cell function to each cell in the 'label' column and create new DataFrame columns
    df['extracted'] = df['label'].apply(process_cell)

    # Expand the 'extracted' column into separate rows
    rows = []
    for index, row in df.iterrows():
        for aspect_star in row['extracted']:
            # Append a new row with duplicated information from the original row and the extracted aspect and star
            new_row = row.to_dict()
            new_row['aspect'], new_row['star'] = aspect_star
            rows.append(new_row)

    # Create a new DataFrame from the rows with expanded data
    df = pd.DataFrame(rows)
    # Drop the original 'label' and 'extracted' columns if no longer needed
    drop_col = df.columns[0:2]
    df.drop(columns=drop_col, axis=1)
    df['star'] = df['star'].astype('int') 
    avg_by_aspect = df.groupby('aspect')['star'].mean().round(1)
    result_df = avg_by_aspect.reset_index(name='star')
    result_df['star'] = result_df.apply(add_icon_to_avg_star, axis=1)
    return result_df
