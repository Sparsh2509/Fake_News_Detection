import streamlit as st
import requests

st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered")
st.title("📰 Fake News Detection")

st.write("Enter the news article text below, and the model will predict whether it is Fake or Real.")

# Default example text
default_text = (
    "An anonymous insider revealed that Coca-Cola will halt production worldwide "
    "after their secret formula was leaked on the dark web. "
    "The company has not yet responded to the allegations."
)

# Initialize session state for the text box
if "news_text" not in st.session_state:
    st.session_state.news_text = default_text

# Text input area
news_text = st.text_area("News Article", value=st.session_state.news_text, height=200, key="news_text")

# API endpoint
API_URL = "http://fake-news-detection2-env.eba-gdmi2vhg.ap-south-1.elasticbeanstalk.com/predict"

col1, col2 = st.columns(2)

with col1:
    predict_clicked = st.button("🔍 Predict")

with col2:
    clear_clicked = st.button("🧹 Clear Text")

# Clear text logic
if clear_clicked:
    st.session_state.news_text = ""
    st.experimental_rerun()

# Prediction logic
if predict_clicked:
    if news_text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Prepare payload
        payload = {"text": news_text}
        try:
            response = requests.post(API_URL, json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                st.subheader("Prediction Result")
                st.write(f"**Prediction:** {result['prediction']}")
                st.write(f"**Confidence:** {result['confidence']}")
                st.write(f"**Message:** {result['message']}")
                st.info(result['Final_Suggestion']['Suggestion'])
            else:
                st.error(f"API Error: {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")