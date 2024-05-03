import streamlit as st
from annotated_text import annotated_text
import json
from streamlit_lottie import st_lottie

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
    
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