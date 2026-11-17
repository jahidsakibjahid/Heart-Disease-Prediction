
import streamlit as st
import pandas as pd
import joblib

# Load saved files
model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
columns = joblib.load("heart_columns.pkl")

st.title("Heart Disease Prediction")
st.write("Enter patient information below")

# User input
age = st.number_input("Age", 1, 120, 30)
sex = st.selectbox("Sex", ["M", "F"])

chest_pain = st.selectbox(
    "Chest Pain Type", ["TA", "ATA", "NAP", "ASY"]
)

resting_bp = st.number_input("Resting Blood Pressure", 1, 250, 120)
cholesterol = st.number_input("Cholesterol", 1, 700, 200)

fasting_bs = st.selectbox("Fasting Blood Sugar", [0, 1])

resting_ecg = st.selectbox(
    "Resting ECG", ["Normal", "ST", "LVH"]
)

max_hr = st.number_input("Maximum Heart Rate", 1, 250, 150)

exercise_angina = st.selectbox("Exercise Angina", ["N", "Y"])

oldpeak = st.number_input("Oldpeak", value=0.0, step=0.1)

st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Prediction button
if st.button("Predict"):

    input_data = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope
    }])

    # Encode categorical data
    input_encoded = pd.get_dummies(
        input_data, drop_first=True, dtype=int
    )

    # Match training columns
    input_encoded = input_encoded.reindex(
        columns=columns, fill_value=0
    )

    # Scale input
    scaled_input = scaler.transform(input_encoded)

    # Predict
    prediction = model.predict(scaled_input)

    if prediction[0] == 1:
        st.error("Model predicts: Heart Disease")
    else:
        st.success("Model predicts: No Heart Disease")
