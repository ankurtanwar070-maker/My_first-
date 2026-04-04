import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Marketing Response Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
}
.prediction-success {
    background-color: #d4edda;
    border-left: 5px solid #28a745;
    padding: 15px;
    border-radius: 5px;
}
.prediction-danger {
    background-color: #f8d7da;
    border-left: 5px solid #dc3545;
    padding: 15px;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Load Model (Cached)
# =========================
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(__file__)
    
    model = joblib.load(os.path.join(BASE_DIR, 'src/marketing_response_model.pkl'))
    scaler = joblib.load(os.path.join(BASE_DIR, 'src/scaler.pkl'))
    le_dict = joblib.load(os.path.join(BASE_DIR, 'src/label_encoder.pkl'))
    
    return model, scaler, le_dict

model, scaler, le_dict = load_model()

# =========================
# Header
# =========================
st.markdown("# 📊 Marketing Response Analyzer")
st.markdown("*Predict customer subscription likelihood using machine learning*")

# =========================
# Model Metrics
# =========================
st.markdown("### 📈 Model Performance")
col1, col2, col3 = st.columns(3)

col1.metric("Model Type", "Random Forest", "100 Trees")
col2.metric("Training Accuracy", "99.97%", "+0.97%")
col3.metric("Testing Accuracy", "79.13%", "↓ Realistic")

st.divider()

# =========================
# Input Section
# =========================
st.markdown("### 👤 Customer Information")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographics")
    age = st.number_input("Age", 18, 100, 30)
    job = st.selectbox("Job", ['admin.', 'technician', 'services', 'management', 'retired',
                              'blue-collar', 'unemployed', 'entrepreneur', 'housemaid',
                              'unknown', 'self-employed', 'student'])
    marital = st.selectbox("Marital Status", ['married', 'single', 'divorced'])
    education = st.selectbox("Education", ['unknown', 'secondary', 'primary', 'tertiary'])

with col2:
    st.subheader("Financial Status")
    balance = st.number_input("Balance (€)", -10000, 100000, 0)
    default = st.selectbox("Credit Default?", ['no', 'yes'])
    housing = st.selectbox("Housing Loan?", ['no', 'yes'])
    loan = st.selectbox("Personal Loan?", ['no', 'yes'])

col3, col4 = st.columns(2)

with col3:
    st.subheader("Contact Details")
    contact = st.selectbox("Contact Type", ['unknown', 'telephone', 'cellular'])
    day = st.number_input("Contact Day", 1, 31, 15)
    month = st.selectbox("Month", ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])

with col4:
    st.subheader("Campaign History")
    duration = st.number_input("Duration (sec)", 0)
    campaign = st.number_input("Campaign Contacts", 1)
    pdays = st.number_input("Days Since Last Contact", -1)
    previous = st.number_input("Previous Contacts", 0)

poutcome = st.selectbox("Previous Outcome", ['unknown', 'other', 'failure', 'success'])

st.divider()

# =========================
# Predict Button (Centered)
# =========================
colA, colB, colC = st.columns([1, 2, 1])
with colB:
    predict_button = st.button("🔮 Make Prediction", use_container_width=True)

# =========================
# Prediction Logic
# =========================
if predict_button:

    # Input DataFrame
    input_data = pd.DataFrame({
        'age': [age],
        'job': [job],
        'marital': [marital],
        'education': [education],
        'default': [default],
        'balance': [balance],
        'housing': [housing],
        'loan': [loan],
        'contact': [contact],
        'day': [day],
        'month': [month],
        'duration': [duration],
        'campaign': [campaign],
        'pdays': [pdays],
        'previous': [previous],
        'poutcome': [poutcome]
    })

    # Expected Feature Order
    expected_order = [
        'age','job','marital','education','default','balance',
        'housing','loan','contact','day','month','duration',
        'campaign','pdays','previous','poutcome'
    ]
    input_data = input_data[expected_order]

    # Encode Safely
    categorical_cols = ['job','marital','education','default','housing','loan','contact','month','poutcome']

    for col in categorical_cols:
        try:
            input_data[col] = le_dict[col].transform(input_data[col])
        except:
            st.error(f"⚠️ Unknown category in {col}")
            st.stop()

    # Scale
    numerical_cols = ['age','balance','day','duration','campaign','pdays','previous']
    input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

    # Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.markdown("### 🎯 Prediction Result")

    col1, col2 = st.columns(2)

    # Result Text
    with col1:
        if prediction == 1:
            st.markdown(
                '<div class="prediction-success"><h3>✅ Likely to Subscribe</h3></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="prediction-danger"><h3>❌ Unlikely to Subscribe</h3></div>',
                unsafe_allow_html=True
            )

    # Probability Chart
    with col2:
        fig, ax = plt.subplots()
        ax.pie(
            probability,
            labels=['No', 'Yes'],
            autopct='%1.1f%%',
            startangle=90
        )
        ax.set_title("Subscription Probability")
        st.pyplot(fig)
        plt.close(fig)

    # Probability Metrics
    st.markdown("### 📊 Probability Details")
    m1, m2 = st.columns(2)
    m1.metric("Will NOT Subscribe", f"{probability[0]:.2%}")
    m2.metric("Will Subscribe", f"{probability[1]:.2%}")

    # Progress Bar
    st.progress(float(probability[1]))

    # Summary
    st.markdown("### 📋 Customer Summary")
    summary = pd.DataFrame({
        'Feature': ['Age','Balance','Job','Education','Duration','Campaign'],
        'Value': [age, balance, job, education, duration, campaign]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

    # Download Option
    st.download_button(
        "📥 Download Result",
        data=input_data.to_csv(index=False),
        file_name="prediction.csv"
    )