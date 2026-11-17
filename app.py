
import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_files():
    model = joblib.load("knn_heart_model.pkl")
    scaler = joblib.load("heart_scaler.pkl")
    columns = joblib.load("heart_columns.pkl")
    return model, scaler, columns

model, scaler, columns = load_files()

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.stApp {
    background: #f4f7fb;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #102a43, #164e63);
}

[data-testid="stSidebar"] * {
    color: white;
}

.hero {
    background: linear-gradient(120deg, #102a43, #176b87, #20a4a8);
    padding: 35px;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 25px;
    box-shadow: 0 8px 25px rgba(16,42,67,0.15);
}

.hero h1 {
    color: white;
    font-size: 36px;
    font-weight: 800;
    margin: 0;
}

.hero p {
    color: #e0f7fa;
    font-size: 16px;
    margin-top: 12px;
}

.section-title {
    color: #12344d;
    font-size: 23px;
    font-weight: 750;
    margin: 12px 0;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: white;
    border-radius: 15px;
    border: 1px solid #e1e8f0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.04);
    padding: 15px;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #087f8c, #16a6a1);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 13px;
    font-weight: 700;
    font-size: 17px;
    transition: 0.3s;
}

.stButton > button:hover {
    background: #075b68;
    color: white;
    border: none;
}

.result-card {
    padding: 25px;
    border-radius: 16px;
    text-align: center;
    margin-top: 20px;
    margin-bottom: 20px;
}

.positive {
    background: #fff1f0;
    border: 1px solid #ffccc7;
}

.negative {
    background: #e9f9ef;
    border: 1px solid #a5e3b8;
}

.result-card h2 {
    font-size: 27px;
    margin-bottom: 10px;
}

.result-card p {
    font-size: 16px;
}

.footer {
    background: #102a43;
    padding: 22px;
    border-radius: 15px;
    text-align: center;
    color: white;
    margin-top: 40px;
}

.footer p {
    color: #d8e6ee;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.markdown("# ❤️ Heart AI")

    st.markdown("---")

    st.markdown("### Navigation")
    st.markdown("🏠 Home")
    st.markdown("📊 Prediction")
    st.markdown("🧠 Machine Learning Model")

    st.markdown("---")

    st.markdown("### About Project")
    st.write(
        "An educational machine learning "
        "project for heart disease prediction."
    )

    st.markdown("---")

    st.markdown("### Technologies")
    st.write("🐍 Python")
    st.write("🤖 Scikit-learn")
    st.write("📊 Pandas")
    st.write("🎨 Streamlit")

    st.markdown("---")

    st.caption("Developed as an ML learning project.")

# ---------------- HERO HEADER ----------------

st.markdown("""
<div class="hero">
    <h1>❤️ Heart Disease Prediction</h1>
    <p>AI-Powered Heart Health Prediction Dashboard</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "Enter the required information below to get a model prediction."
)

# ---------------- INPUT FORM ----------------

with st.form("prediction_form"):

    st.markdown(
        '<div class="section-title">👤 Personal Information</div>',
        unsafe_allow_html=True
    )

    with st.container(border=True):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Age",
                min_value=1,
                max_value=120,
                value=30
            )

        with col2:
            sex = st.selectbox("Sex", ["M", "F"])

    st.markdown(
        '<div class="section-title">🫀 Medical Information</div>',
        unsafe_allow_html=True
    )

    with st.container(border=True):
        col1, col2 = st.columns(2)

        with col1:
            chest_pain = st.selectbox(
                "Chest Pain Type",
                ["TA", "ATA", "NAP", "ASY"]
            )

            resting_bp = st.number_input(
                "Resting Blood Pressure",
                min_value=1,
                max_value=250,
                value=120
            )

            cholesterol = st.number_input(
                "Cholesterol",
                min_value=1,
                max_value=700,
                value=200
            )

            fasting_bs = st.selectbox(
                "Fasting Blood Sugar",
                [0, 1]
            )

        with col2:
            resting_ecg = st.selectbox(
                "Resting ECG",
                ["Normal", "ST", "LVH"]
            )

            max_hr = st.number_input(
                "Maximum Heart Rate",
                min_value=1,
                max_value=250,
                value=150
            )

            exercise_angina = st.selectbox(
                "Exercise Angina",
                ["N", "Y"]
            )

            oldpeak = st.number_input(
                "Oldpeak",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=0.1
            )

        st_slope = st.selectbox(
            "ST Slope",
            ["Up", "Flat", "Down"]
        )

    st.markdown("---")

    submitted = st.form_submit_button(
        "🔍 Predict Heart Disease"
    )

# ---------------- PREDICTION ----------------

if submitted:

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

    input_encoded = pd.get_dummies(
        input_data,
        drop_first=True,
        dtype=int
    )

    input_encoded = input_encoded.reindex(
        columns=columns,
        fill_value=0
    )

    scaled_input = scaler.transform(input_encoded)

    prediction = model.predict(scaled_input)

    st.markdown("---")
    st.markdown(
        '<div class="section-title">📋 Prediction Result</div>',
        unsafe_allow_html=True
    )

    if prediction[0] == 1:

        st.markdown("""
        <div class="result-card positive">
            <h2>⚠️ Model Prediction: Heart Disease</h2>
            <p>The model predicts a positive result.</p>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="result-card negative">
            <h2>✅ Model Prediction: No Heart Disease</h2>
            <p>The model predicts a negative result.</p>
        </div>
        """, unsafe_allow_html=True)

    st.info(
        "This is an educational machine learning prediction, "
        "not a medical diagnosis. Consult a qualified healthcare "
        "professional for medical advice."
    )

# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">
    <h3>❤️ Heart Disease Prediction</h3>
    <p>Machine Learning | Python | Scikit-learn | Streamlit</p>
    <p>Educational Project • Built for Learning AI & ML</p>
    <p>⚕️ Not intended for medical diagnosis or treatment.</p>
</div>
""", unsafe_allow_html=True)
