import streamlit as st
import pandas as pd
from preprocess import preprocess_input
from model_helpers import load_model, predict

st.set_page_config(page_title="Placement Predictor", layout="centered")

st.title("🎓 Student Placement Predictor")
st.write("Enter your academic details below to predict your placement chance.")

# Input fields
cgpa = st.slider("CGPA", 0.0, 10.0, 7.0, 0.1)
iq = st.slider("IQ", 70, 160, 110)
profile_score = st.slider("Profile Score", 0, 100, 50)

if st.button("Predict Placement"):
    processed_input = preprocess_input(cgpa, iq, profile_score)
    model = load_model()
    result = predict(model, processed_input)

    if result == 1:
        st.success("🎉 Congratulations! You are likely to be placed.")
    else:
        st.error("❌ Sorry, placement is unlikely. Keep improving!")
