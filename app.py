import os
import time
import streamlit as st
import json
import requests
from streamlit_lottie import st_lottie
import pandas as pd
from utils.preprocess_user_data import auto_detect_filter_data, take_info, sentiments_frequency, sentiment_bar
from utils.preprocess_user_data import preprocess_data
from utils.tokenizer import tokenize_function, call_tokenizer
from utils.preprocess_text import preprocess
from predict import show_predict_text,process_predict_csv, show_predict_csv
import matplotlib.pyplot as plt
import seaborn as sns
from annotated_text import annotated_text

# Initialize session state for file upload status
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False


def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
    
def plot_aspect_frequency(aspect_df):
    ## Tạo biểu đồ cột
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(aspect_df['Aspect'], aspect_df['Frequency'], color='skyblue')
    
    # Thêm tiêu đề và nhãn trục
    ax.set_title('Frequency of Aspects')
    ax.set_xlabel('Aspect')
    ax.set_ylabel('Frequency')
    
    # Xoay nhãn trục x
    plt.xticks(rotation=45)
    
    # Thay đổi màu của các thanh cột
    for bar in bars:
        bar.set_color('skyblue')
    
    # Hiển thị giá trị trên mỗi cột
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # Hiển thị biểu đồ
    st.pyplot(fig)
    
def plot_sentiment_frequencies(sentiment_df):
    # Set seaborn color palette
    sns.set_palette("Set3")

    # Plot pie chart using matplotlib
    fig, ax = plt.subplots(figsize=(6, 6))  # smaller size
    ax.pie(sentiment_df['frequency'], labels=sentiment_df['sentiments'], autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    # Add legend
    ax.set_title('Percentage of Sentiments')
    ax.legend(sentiment_df['sentiments'], loc="upper left", bbox_to_anchor=(1, 0, 0.5, 1))
    # Display pie chart using Streamlit
    st.pyplot(fig)

lottie_ai = load_lottiefile("lottiefiles/logo.json")
lottie_robot = load_lottiefile("lottiefiles/robot_orange.json")

def plot_sentiment_bar(aspect_sentiment_df):
    # Plot the horizontal bar chart with stacked bars
    sns.set_palette("Set3")

    fig, ax = plt.subplots(figsize=(10, 8))
    aspect_sentiment_df[['positive', 'negative', 'neutral']].plot(kind='barh', stacked=True, ax=ax)
    
    # Add labels and title
    ax.set_xlabel('Sentiment Count')
    ax.set_ylabel('Aspect')
    ax.set_title('Sentiment Analysis by Aspect')
    ax.legend(title='Sentiment')

    # Set y-ticks to display aspect names
    ax.set_yticks(range(len(aspect_sentiment_df.index)))
    ax.set_yticklabels(aspect_sentiment_df['aspect'])
    st.pyplot(plt)

# sidebar decoration
with st.sidebar:
    st_lottie(lottie_ai, speed=1, loop=True, quality="low")
    st.info("Select a choice below.")
    choice = st.radio('Navigation',['Home','Upload','Apply ABSA','More information','About us'])
if 'absa_applied' not in st.session_state:
    st.session_state.absa_applied = False  # Initialize the flag if it doesn't exist in session state
# hanlde choice
if choice == 'Home':
    st.title("こんにちは! Welcome to our ABSA web app😊")
    st_lottie(lottie_robot, speed=1, loop=True, quality="low")
    # snowfall
    if st.button("❄️降雪❄️"):
        st.snow()

elif choice == 'Upload':
    # Initialize session state variable if not already present
    st.subheader("🤖📢 Before upload, test our model if you want to know what we will do 👌")

    # Text input for user review about smartphone
    user_input = st.text_input("Enter some review about your smartphone 👇", 
                          placeholder="This is a placeholder...")

    # Display results when user inputs text
    if user_input:
        results = show_predict_text(user_input)
        if results:
            for result in results:
                st.write(f'=> {result}')
        else:
            st.write("Sorry, I don't recognize any aspect of smartphone in your review")
    st.warning('Please ensure to fill some text before hitting enter.')  # Warning if no text is entered
    st.title("Upload your data here")
    file = st.file_uploader("We accept various types of data. So don't worry, just go ahead!")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("data_user/source.csv", index = False)
        st.dataframe(df, use_container_width=True)
        st.session_state.file_uploaded = True
        st.success("Yahoo! Your data has been uploaded successfully. Now move to the next step for preprocessing🎉")

    elif not st.session_state.file_uploaded:
        df_demo = pd.read_csv("data/RawData/tikiData/tikiData_small.csv")
        df_demo.to_csv("data_user/source.csv", index = False)
        st.dataframe(df_demo, use_container_width=True)
        st.success("Demo file⭐")
        st.session_state.file_uploaded = True

    elif st.session_state.file_uploaded and file is None:
        df_demo = pd.read_csv("data/RawData/tikiData/tikiData_small.csv")
        df_demo.to_csv("data_user/source.csv", index = False)
        st.dataframe(df_demo, use_container_width=True)
        st.success("Demo file⭐")
        st.session_state.file_uploaded = True

    else:
        st.info("Demo data is hidden because you uploaded your own data.")
# Nếu người dùng đã tải lên một tệp, ẩn dữ liệu mẫu

if choice in ['Apply ABSA']:
    if not st.session_state.file_uploaded:
        st.warning("Please upload a file first before proceeding to this step.")
    else:   
        lottie_data_to_ai = load_lottiefile("lottiefiles/data_to_ai.json")
        st_lottie(lottie_data_to_ai, speed=1, loop=True, quality="low")  
        progress_bar = st.progress(0) 
                     
        progress_bar.progress(25)
        time.sleep(2)
        input_path = "data_user/source.csv"
        output_path = "data_user/raw.csv"
        auto_detect_filter_data(input_path, output_path)
        df_detect = pd.read_csv(output_path, index_col=None)
        
        progress_bar.progress(50)
        time.sleep(1)
        df_clean = preprocess_data(df_detect)
        output_csv_path = "data_user/data_with_label.csv"  # Specify output CSV file path
        
        progress_bar.progress(75)
        process_predict_csv(df_clean, output_csv_path)
        st.session_state.absa_applied = True  # Set flag to True indicating ABSA has been applied
        progress_bar.progress(100)
            
        show = show_predict_csv()
        st.dataframe(show)
        
elif choice == "More information":
    if not st.session_state.absa_applied:
        st.warning("Please apply ABSA first!")
    else:
        st.header('Want to Deeper Understand? Ok!👌', divider='rainbow')
        show = show_predict_csv()
        st.dataframe(show)
        
        st.subheader('The dataframe below contain predict and confidence score', divider='blue')     
        df = pd.read_csv("data_user/data_with_label.csv")
        st.dataframe(df)
        len_df = len(df)
        nan_rows = df[df.isna().any(axis=1)]
        num_predictors = len(nan_rows)
        st.info(f"We have successfully predicted for {len_df - num_predictors}/{len_df} reviews.", icon="⭐")
        if st.button("Click here if you want to insight the data which are not yet predicted"):
            st.dataframe(nan_rows)
        st.divider()
        st.subheader("Let's Explore Your Data")
        aspect_df = take_info(df)
        # Example list of sorted top aspect names
        top_aspect_names = aspect_df.nlargest(3, 'Frequency')['Aspect'].tolist()
        sorted_top_aspect_names = aspect_df[aspect_df['Aspect'].isin(top_aspect_names)].sort_values(by='Frequency', ascending=False)['Aspect'].tolist()
        # Create the HTML string with the sorted aspect names
        html_str = f"<p style='color: black;'>🐙Top 3 aspects that customers are concerned about: "
        for aspect in sorted_top_aspect_names:
            html_str += f" {aspect},"
        html_str = html_str[:-1]  # Remove the last comma
        html_str += "💕</p>"
        # Display the HTML string using st.markdown()
        st.markdown(html_str, unsafe_allow_html=True)        
        plot_aspect_frequency(aspect_df)
        st.divider()
        sentiment_df = sentiments_frequency(df)
        total_sentiment = sentiment_df.iloc[:, 1].sum()  # Access the values of the second column and calculate their sum
        html_str = f"<p style='color: black;'>👽We have calculated total "
        html_str += f"{total_sentiment} sentiment: "
        output = ", ".join([f"{frequency} {sentiment}s" for frequency, sentiment in zip(sentiment_df['frequency'], sentiment_df['sentiments'])])
        html_str += output
        html_str += "💗</p>"

        st.markdown(html_str, unsafe_allow_html=True)        
        plot_sentiment_frequencies(sentiment_df)
        
        st.divider()
        
        aspect_sentiment_df = sentiment_bar(df)
        plot_sentiment_bar(aspect_sentiment_df)

        
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