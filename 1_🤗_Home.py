import streamlit as st
import json
import requests
from streamlit_lottie import st_lottie
import time
import pandas as pd
import os
from utils.preprocess_user_data import auto_detect_filter_data
from utils.preprocess_user_data import preprocess_data
from utils.tokenizer import tokenize_function, call_tokenizer
from datasets import load_dataset

st.set_page_config(
    page_title="ML APP",
    page_icon="👋",
)

# Initialize session state for file upload status
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False


def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
    
lottie_ai = load_lottiefile("lottiefiles/logo.json")
lottie_robot = load_lottiefile("lottiefiles/robot_orange.json")

# sidebar decoration
with st.sidebar:
    st_lottie(lottie_ai, speed=1, loop=True, quality="low")
    st.info("Select a choice below.")
    choice = st.radio('Navigation',['Home','Upload','Preprocess','Apply ABSA','About us'])
lottie_ai = load_lottiefile("lottiefiles/logo.json")

# hanlde choice
if choice == 'Home':
    st.title("こんにちは! Welcome to our ABSA web app😊")
    st_lottie(lottie_robot, speed=1, loop=True, quality="low")
    # snowfall
    if st.button(":pink[きれいなゆき・Bông tuyết trong sạch]🤡"):
        st.snow()

elif choice == 'Upload':
    st.title("Upload your data here")
    file = st.file_uploader("We accept various types of data. So don't worry, just go ahead!")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('C:/Users/FPTSHOP/OneDrive/Documents/SAV/data_user/source.csv', index=None)
        st.dataframe(df)
        st.success("Yahoo! Your data has been uploaded successfully. Now move to the next step for preprocessing🎉",)
        st.session_state.file_uploaded = True
# elif choice in ['Preprocess', 'Tokenization', 'Apply ABSA']:
#     if not st.session_state.file_uploaded:
#         st.warning("Please upload a file first before proceeding to this step.")
#     else:       
elif choice == 'Preprocess':
    st.title("Preprocessing")
    input_path = "C:/Users/FPTSHOP/OneDrive/Documents/SAV/data_user/source.csv"
    output_path = "C:/Users/FPTSHOP/OneDrive/Documents/SAV/data_user/raw.csv"
    auto_detect_filter_data(input_path, output_path)
    df_detect = pd.read_csv(output_path, index_col=None)
    df_clean = preprocess_data(df_detect)
    df_clean.to_csv("C:/Users/FPTSHOP/OneDrive/Documents/SAV/data_user/cleandata.csv", index = False)
    st.dataframe(df_clean)
    st.success("Yahoo! Your data has been preprocessing successfully. Now you can try our ABSA model🎉",)

elif choice == "Apply ABSA":
    DATA_PATH = "C:/Users/FPTSHOP/OneDrive/Documents/SAV/data_user/cleandata.csv"
    read = pd.read_csv(DATA_PATH,index_col=None)
    tokenizer = call_tokenizer()
    for column in read.columns:
        for row in read[column]:
            tokenized_inputs = tokenizer(row, max_length=256, truncation=True, padding='max_length', return_tensors="tf")
    # with st.spinner("Wait for seconds..."):
    #     time.sleep(1)                
    
    

    