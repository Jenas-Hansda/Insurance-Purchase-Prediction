# Project 3 – Streamlit Insurance Prediction App (Minimal)

import pandas as pd
import pickle
import streamlit as st
from sklearn.linear_model import LogisticRegression
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

st.title("🚑 Insurance Purchase Prediction")
st.write("Will someone buy insurance based on their age?")

# Load data
df = pd.read_csv("insurance.csv")

# Prepare features and target
x = df[["age"]]
y = df["bought_insurance"]

# Train logistic regression model
model = LogisticRegression()
model.fit(x, y)

# Save model
with open("insurance_model.pkl", "wb") as f:
    pickle.dump(model, f)

# User input for prediction
st.subheader("🧮 Make a Prediction")
age_input = st.number_input("Enter Age:", min_value=1, max_value=120, value=25)

if st.button("Predict"):
    result = model.predict([[age_input]])[0]
    if result == 1:
        st.success("✅ Likely to buy insurance.")
    else:
        st.error("❌ Unlikely to buy insurance.")
