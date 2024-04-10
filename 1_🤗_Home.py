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
from utils.preprocess_text import preprocess
from predict import show_predict_text

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
        df.to_csv('data_user/source.csv', index=None)
        st.dataframe(df,use_container_width=True)
        st.success("Yahoo! Your data has been uploaded successfully. Now move to the next step for preprocessing🎉",)
        st.session_state.file_uploaded = True
# elif choice in ['Preprocess', 'Apply ABSA']:
#     if not st.session_state.file_uploaded:
#         st.warning("Please upload a file first before proceeding to this step.")
#     else:       
elif choice == 'Preprocess':
    st.title("Preprocessing")
    input_path = "data_user/source.csv"
    output_path = "data_user/raw.csv"
    auto_detect_filter_data(input_path, output_path)
    df_detect = pd.read_csv(output_path, index_col=None)
    df_clean = preprocess_data(df_detect)
    df_clean.to_csv("data_user/cleandata.csv", index = False)
    st.dataframe(df_clean)
    st.success("Yahoo! Your data has been preprocessing successfully. Now you can try our ABSA model🎉",)

elif choice == "Apply ABSA":
    if 'ready_to_input' not in st.session_state:
        st.session_state['ready_to_input'] = False

    # Nút để bắt đầu nhập liệu
    if st.button('Start typing'):
        st.session_state['ready_to_input'] = True  # Đặt trạng thái sẵn sàng nhập

    # Nếu trạng thái sẵn sàng nhập là True, hiển thị ô nhập văn bản
    if st.session_state['ready_to_input']:
        user_input = st.text_input("Enter some text 👇", key='user_input',placeholder="This is a placeholder...")

        # Nếu người dùng nhấn Enter trong ô nhập liệu (text_input luôn trả về giá trị, kể cả chuỗi rỗng)
        if 'user_input' in st.session_state and st.session_state.user_input != '':
            text = st.session_state.user_input
            results = show_predict_text(text)
            if results is not None:  # Kiểm tra xem results có phải là None hay không
                for result in results:
                    st.write(f'=>{result}\n')     
            else:
                st.write("Sorry, I don't recognize any aspect of smartphone in your review")   
        elif 'user_input' in st.session_state and st.session_state.user_input == '':
            st.warning('Please ensure to fill some text before hitting enter.')  # Cảnh báo nếu không nhập gì
                
    
          
    

    

    