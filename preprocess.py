import pandas as pd
import pickle

def preprocess_input(cgpa, iq, profile_score):
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    input_df = pd.DataFrame([[cgpa, iq, profile_score]], columns=["cgpa", "iq", "profile_score"])
    input_scaled = scaler.transform(input_df)

    return input_scaled
