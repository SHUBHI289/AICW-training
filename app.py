import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("wine_model.pkl")

st.title("🍷 Wine Quality Predictor")

st.write("Enter wine characteristics:")

# Input fields
fixed_acidity = st.slider("Fixed Acidity", 0.0, 15.0)
volatile_acidity = st.slider("Volatile Acidity", 0.0, 2.0)
citric_acid = st.slider("Citric Acid", 0.0, 1.0)
residual_sugar = st.slider("Residual Sugar", 0.0, 15.0)
chlorides = st.slider("Chlorides", 0.0, 1.0)
free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 0, 100)
total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 0, 300)
density = st.slider("Density", 0.990, 1.005)
pH = st.slider("pH", 2.5, 4.5)
sulphates = st.slider("Sulphates", 0.0, 2.0)
alcohol = st.slider("Alcohol", 8.0, 15.0)

# Prediction
if st.button("Predict Quality"):
    features = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                          residual_sugar, chlorides, free_sulfur_dioxide,
                          total_sulfur_dioxide, density, pH,
                          sulphates, alcohol]])

    prediction = model.predict(features)
    st.success(f"Predicted Wine Quality: {prediction[0]}")