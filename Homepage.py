import json

import streamlit as st
from streamlit_lottie import st_lottie


def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


lottie_robot = load_lottiefile("lottiefiles/logo.json")

lottie_code = load_lottiefile("lottiefiles/robot_orange.json")
st.title("こんにちは! Welcome to our ABSA web app😊")

# sidebar decoration
with st.sidebar:
    st_lottie(lottie_robot, speed=1, loop=True, quality="low")
    st.success("Select a page above.")
st.info("You can try our services in the navigation")
st_lottie(lottie_code, speed=1, loop=True, quality="low")

# snowfall
if st.button("❄️降雪❄️"):
        st.snow()