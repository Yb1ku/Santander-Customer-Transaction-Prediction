import streamlit as st
import joblib
import numpy as np

model = joblib.load("models/xgb_model.joblib")
pipeline = joblib.load("models/pipeline.joblib")

st.title("Santander Transaction Predictor")

features = [st.slider(f"Feature {i}", 0.0, 1.0, 0.5) for i in range(200)]
X = np.array([features])
X_trans = pipeline.transform(X)
proba = model.predict_proba(X_trans)[0, 1]
pred = int(proba >= 0.6)

st.write(f"Prediction: {pred} | Probability: {proba:.4f}")
