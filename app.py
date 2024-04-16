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
from utils.preprocess_text import preprocess
from predict import show_predict_text,process_predict_csv, show_predict_csv
from annotated_text import annotated_text

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
    choice = st.radio('Navigation',['Home','Upload','Apply ABSA','About us'])

# hanlde choice
if choice == 'Home':
    st.title("こんにちは! Welcome to our ABSA web app😊")
    st_lottie(lottie_robot, speed=1, loop=True, quality="low")
    # snowfall
    if st.button("きれいなゆき・Bông tuyết trong sạch🤡"):
        st.snow()

elif choice == 'Upload':
    if 'ready_to_input' not in st.session_state:
        st.session_state['ready_to_input'] = False

    # Nút để bắt đầu nhập liệu
    if st.button('🤖📢Before upload, please press me if you want to know what we will do👌'):
        st.session_state['ready_to_input'] = True  # Đặt trạng thái sẵn sàng nhập

    # Nếu trạng thái sẵn sàng nhập là True, hiển thị ô nhập văn bản
    if st.session_state['ready_to_input']:
        user_input = st.text_input("Enter some review about your smartphone 👇", key='user_input',placeholder="This is a placeholder...")

        # Nếu người dùng nhấn Enter trong ô nhập liệu (text_input luôn trả về giá trị, kể cả chuỗi rỗng)
        if 'user_input' in st.session_state and st.session_state.user_input != '':
            text = st.session_state.user_input
            results = show_predict_text(text)
            if results is not None:  # Kiểm tra xem results có phải là None hay không
                for result in results:
                    st.write(f'=>{result}\n')     
            elif results == None:
                st.write("Sorry, I don't recognize any aspect of smartphone in your review")   
        elif 'user_input' in st.session_state and st.session_state.user_input == '':
            st.warning('Please ensure to fill some text before hitting enter.')  # Cảnh báo nếu không nhập gì
    st.title("Upload your data here")
    file = st.file_uploader("We accept various types of data. So don't worry, just go ahead!")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('data_user/source.csv', index=None)
        st.dataframe(df,use_container_width=True)
        st.success("Yahoo! Your data has been uploaded successfully. Now move to the next step for preprocessing🎉",)
        st.session_state.file_uploaded = True
elif choice in ['Apply ABSA']:
    if not st.session_state.file_uploaded:
        st.warning("Please upload a file first before proceeding to this step.")
    else:       
        if choice == "Apply ABSA":
            lottie_data_to_ai = load_lottiefile("lottiefiles/data_to_ai.json")
            st_lottie(lottie_data_to_ai, speed=1, loop=True, quality="low")    
            input_path = "data_user/source.csv"
            output_path = "data_user/raw.csv"
            auto_detect_filter_data(input_path, output_path)
            df_detect = pd.read_csv(output_path, index_col=None)
            df_clean = preprocess_data(df_detect)
            output_csv_path = "data_user/data_with_label.csv"  # Specify output CSV file path
            process_predict_csv(df_clean, output_csv_path)
            df = pd.read_csv(output_csv_path)
            show = show_predict_csv()
            st.dataframe(show)
            if st.button('Click here if you want know more details🫶'):
                st.dataframe(df)
elif choice == 'About us':
    st.markdown("<h1 style='text-align: center; color: black;'>About Us</h1>", unsafe_allow_html=True)
    url_company = "https://jvb-corp.com/vi/"
    url_git = "https://github.com/leanhtu-AI/Sentiment-Analysis.git"
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("<h2 style='color: black;'>🤝Our Organization🤝</h2>", unsafe_allow_html=True)
        annotated_text(
            "Hi, I'm",
            ("Junior-VB", "", "#faa"),
            "🤖"
        )
        st.markdown("""- I was created by a team of AI interns from JVB Vietnam company.\n - Beside ABSA model, we also provide other technology solutions.\n - Check out this [link](%s) for more information about our group""" % url_company)
        st.markdown("")
        st.markdown("")
        st.markdown("<h2 style='color: black;'>🐙Github Repository😺</h2>", unsafe_allow_html=True)
        st.markdown("""- Want to deep understand how I work? Please visit this [repo](%s).\n - Every usage and contribute to the code are welcome!""" % url_git)
        annotated_text(
            ("Transformers🤖", "", "#fea"),
            ("Underthesea🌊", "", "#8ef"),
            ("PhoBert💕", "", "#ff80ed"),     
            ("Tensorflow🌞", "", "#afa"),
            ("Hugging Face🤗", "", "#faa"),
        )
        st.markdown("")
        st.markdown("")
        url_facebook = 'https://www.facebook.com/lnht1808.secsip'
        url_github = 'https://github.com/leanhtu-AI'
        url_gmail = 'https://mail.google.com/mail/u/3/#inbox'
        st.markdown("<h2 style='color: black;'>📞Contact🫶</h2>", unsafe_allow_html=True)
        st.markdown("""- [Facebook](%s)\n- [Gmail](%s)\n- [Github](%s)""" % (url_facebook, url_gmail, url_github))

    with col2:
        lottie_col1 = load_lottiefile("lottiefiles/hello.json")
        st_lottie(lottie_col1, speed=1, loop=True, quality="low")
        lottie_col1 = load_lottiefile("lottiefiles/github.json")
        st_lottie(lottie_col1, speed=1, loop=True, quality="low")
        
    st.markdown("<h4 style='text-align: center; color: black; opacity: 0.5;'>ありがとう ございます。</h4>", unsafe_allow_html=True)