
import streamlit as st
import pandas as pd
import joblib

# Load saved files
model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
columns = joblib.load("heart_columns.pkl")

# Custom CSS
st.markdown("""
<style>

.main {
    background-color: #f4f7fb;
}

.hero {
    background: linear-gradient(120deg, #102a43, #176b87, #20a4a8);
    padding: 35px 30px;
    border-radius: 18px;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 8px 25px rgba(16,42,67,0.18);
}

.hero h1 {
    color: white;
    font-size: 36px;
    font-weight: 800;
    margin-bottom: 10px;
}

.hero p {
    color: #e0f7fa;
    font-size: 16px;
    margin-bottom: 0;
}

</style>
""", unsafe_allow_html=True)

# Custom Header
st.markdown("""
<div class="hero">
    <h1>❤️ Heart Disease Prediction</h1>
    <p>AI-powered Heart Health Risk Prediction System</p>
</div>
""", unsafe_allow_html=True)


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
