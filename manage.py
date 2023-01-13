import streamlit as st

from src.SenseExtractor import SenseExtractor

st.title("User Review Classificator")
st.subheader("Please Enter Review")

with st.form("form", clear_on_submit=True):
    text = st.text_area("Review")

    submit = st.form_submit_button("Submit")

if submit:
    sentiment, keywords = SenseExtractor.analyse_text(text)

    st.write(sentiment.capitalize())
    st.write(keywords)
