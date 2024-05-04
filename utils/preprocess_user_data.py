import pandas as pd
import re
from preprocess_text import preprocess
import matplotlib.pyplot as plt
import ast
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

def take_info_phone(df):
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


def take_info_hotel(df):
    regex = r'^([^,]+)'
    aspect_list = ['FACILITIES#CLEANLINESS', 'FACILITIES#COMFORT', 'FACILITIES#DESIGN&FEATURES', 'FACILITIES#GENERAL', 'FACILITIES#MISCELLANEOUS', 'FACILITIES#PRICES', 'FACILITIES#QUALITY', 'FOOD&DRINKS#MISCELLANEOUS', 'FOOD&DRINKS#PRICES', 'FOOD&DRINKS#QUALITY', 'FOOD&DRINKS#STYLE&OPTIONS', 'HOTEL#CLEANLINESS', 'HOTEL#COMFORT', 'HOTEL#DESIGN&FEATURES', 'HOTEL#GENERAL', 'HOTEL#MISCELLANEOUS', 'HOTEL#PRICES', 'HOTEL#QUALITY', 'LOCATION#GENERAL', 'ROOM_AMENITIES#CLEANLINESS', 'ROOM_AMENITIES#COMFORT', 'ROOM_AMENITIES#DESIGN&FEATURES', 'ROOM_AMENITIES#GENERAL', 'ROOM_AMENITIES#MISCELLANEOUS', 'ROOM_AMENITIES#PRICES', 'ROOM_AMENITIES#QUALITY', 'ROOMS#CLEANLINESS', 'ROOMS#COMFORT', 'ROOMS#DESIGN&FEATURES', 'ROOMS#GENERAL', 'ROOMS#MISCELLANEOUS', 'ROOMS#PRICES', 'ROOMS#QUALITY', 'SERVICE#GENERAL']

    aspect_counts = {aspect: 0 for aspect in aspect_list}

    for label_data in df['label']:
        # Handle cases where label_data is a string representation of a list
        if isinstance(label_data, str):
            try:
                labels = ast.literal_eval(label_data)
            except ValueError:
                continue  # Skip rows where the data can't be evaluated to a list
        else:
            labels = label_data  # Assuming labels is already a list (if not, handle other cases)

        for label in labels:
            match = re.match(regex, label)
            if match:
                aspect = match.group(1)
                if aspect in aspect_list:
                    aspect_counts[aspect] += 1

    return pd.DataFrame(list(aspect_counts.items()), columns=['Aspect', 'Frequency'])

def take_info_res(df):
    regex = r'^([^,]+)'
    aspect_list = ['FOOD#PRICES', 'DRINKS#STYLE&OPTIONS', 'FOOD#STYLE&OPTIONS', 'RESTAURANT#MISCELLANEOUS', 'SERVICE#GENERAL', 'DRINKS#QUALITY', 'RESTAURANT#PRICES', 'LOCATION#GENERAL', 'DRINKS#PRICES', 'RESTAURANT#GENERAL', 'AMBIENCE#GENERAL', 'FOOD#QUALITY']
    aspect_counts = {aspect: 0 for aspect in aspect_list}

    for label_data in df['label']:
        # Handle cases where label_data is a string representation of a list
        if isinstance(label_data, str):
            try:
                labels = ast.literal_eval(label_data)
            except ValueError:
                continue  # Skip rows where the data can't be evaluated to a list
        else:
            labels = label_data  # Assuming labels is already a list (if not, handle other cases)

        for label in labels:
            match = re.match(regex, label)
            if match:
                aspect = match.group(1)
                if aspect in aspect_list:
                    aspect_counts[aspect] += 1

    return pd.DataFrame(list(aspect_counts.items()), columns=['Aspect', 'Frequency'])

def take_info_stu(df):
    """
    Đếm số lần xuất hiện của các aspect đã được chỉ định từ cột "label" trong DataFrame.

    Parameters:
    - df (DataFrame): DataFrame chứa cột "label" cần kiểm tra.

    Returns:
    DataFrame: DataFrame chứa hai cột "aspect" và "số lần xuất hiện".
    """
    regex = r'(\w+),\w+,\d+\.\d+,\d+'

    aspect_list = ['facility','lecturer','others','training_program']

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
