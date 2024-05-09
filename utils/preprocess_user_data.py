import ast
import re

import pandas as pd

from utils.preprocess_text import preprocess


def auto_detect_filter_data(input_path, output_path):
    """
    Lọc và lưu trữ cột đánh giá từ một tệp CSV vào một tệp mới.
    
    Parameters:
    - input_path (str): Đường dẫn đến tệp CSV đầu vào.
    - output_path (str): Đường dẫn đến tệp CSV đầu ra sau khi lọc.
    
    Returns:
    None
    """
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
    """
    Tiền xử lý dữ liệu trong DataFrame bằng cách áp dụng hàm preprocess() cho từng cột.

    Parameters:
    - df (DataFrame): DataFrame chứa dữ liệu cần tiền xử lý.

    Returns:
    DataFrame: DataFrame sau khi đã được tiền xử lý.
    """
    for column in df.columns:
        df[column] = df[column].apply(preprocess)
    return df

def take_info(df):
    """
    Đếm số lần xuất hiện của các aspect đã được chỉ định từ cột "label" trong DataFrame.

    Parameters:
    - df (DataFrame): DataFrame chứa cột "label" cần kiểm tra.

    Returns:
    DataFrame: DataFrame chứa hai cột "aspect" và "số lần xuất hiện".
    """
    regex = r'(\w+),\w+,\d+\.\d+,\d+'

    aspect_list = ['CAMERA','SCREEN','GENERAL','STORAGE','PERFORMANCE','SERACC','BATTERY','PRICE','DESIGN','FEATURES']

    aspect_counts = {aspect: 0 for aspect in aspect_list}

    for label in df['label']:
        # Kiểm tra nếu giá trị là chuỗi
        if isinstance(label, str):
            matches = re.findall(regex, label)
            for aspect in matches:
                if aspect in aspect_list:
                    aspect_counts[aspect] += 1

    # Tạo DataFrame từ dict aspect_counts
    aspect_df = pd.DataFrame(aspect_counts.items(), columns=['Aspect', 'Frequency'])
    return aspect_df

def sentiments_frequency(df):
    # Define a regular expression pattern to extract sentiment values
    pattern = r"'[^']*?,([^']*?),"

    # Initialize an empty list to store extracted sentiment values
    sentiments = []

    # Iterate over each row in the 'label' column and apply the regex pattern
    for row in df['label']:
        matches = re.findall(pattern, str(row))
        for match in matches:
            sentiment = match.strip()
            sentiments.append(sentiment)

    # Count frequency of each sentiment
    sentiment_counts = {}
    for sentiment in sentiments:
        if sentiment in sentiment_counts:
            sentiment_counts[sentiment] += 1
        else:
            sentiment_counts[sentiment] = 1

    # Create DataFrame from sentiment_counts dictionary
    sentiments_df = pd.DataFrame(sentiment_counts.items(), columns=['sentiments', 'frequency'])

    return sentiments_df

def sentiment_bar(df):
    aspect_sentiment_counts = {}

    # Iterate through each row of the DataFrame
    for index, row in df.iterrows():
        # Check if the row contains NaN values
        if row.isnull().values.any():
            continue

        # Parse the sentiment list from the string representation
        sentiment_list = ast.literal_eval(row['label'])
        
        # Iterate through each sentiment entry in the list
        for sentiment_entry in sentiment_list:
            aspect, sentiment, score, _ = sentiment_entry.split(",")
            # Increment the corresponding sentiment count for the aspect and sentiment
            if aspect not in aspect_sentiment_counts:
                aspect_sentiment_counts[aspect] = {}
            if sentiment not in aspect_sentiment_counts[aspect]:
                aspect_sentiment_counts[aspect][sentiment] = 0
            aspect_sentiment_counts[aspect][sentiment] += 1

    # Create a DataFrame from the aspect sentiment counts
    aspect_sentiment_df = pd.DataFrame(aspect_sentiment_counts).transpose().reset_index()

    # Rename columns to ensure consistency
    aspect_sentiment_df = aspect_sentiment_df.rename(columns={'index': 'aspect'})

    # Ensure columns for all sentiments ('positive', 'neutral', 'negative') exist
    for sentiment in ['positive', 'neutral', 'negative']:
        if sentiment not in aspect_sentiment_df.columns:
            aspect_sentiment_df[sentiment] = 0

    # Fill NaN values with 0
    aspect_sentiment_df.fillna(0, inplace=True)

    # Convert sentiment counts to integers
    aspect_sentiment_df[['positive', 'neutral', 'negative']] = aspect_sentiment_df[['positive', 'neutral', 'negative']].astype(int)

    # Reorder columns
    aspect_sentiment_df = aspect_sentiment_df[['aspect', 'neutral', 'negative','positive']]

    # Calculate total sentiment count for each aspect
    aspect_sentiment_df['total'] = aspect_sentiment_df['negative'] + aspect_sentiment_df['neutral'] + aspect_sentiment_df['positive']
    aspect_sentiment_df = aspect_sentiment_df.sort_values(by='total', ascending=False)

    # Print or return the resulting DataFrame
    return aspect_sentiment_df

    
