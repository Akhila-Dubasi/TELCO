import streamlit as st
import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# THEME TOGGLE
# --------------------------------------------------
mode = st.sidebar.radio("🌗 Theme Mode", ["Light Mode", "Dark Mode"])

# --------------------------------------------------
# CSS THEMES
# --------------------------------------------------
if mode == "Dark Mode":
    bg = "#020617"
    card = "#020617"
    text = "#f8fafc"
    subtext = "#cbd5f5"
else:
    bg = "#f4f7fb"
    card = "#ffffff"
    text = "#0f172a"
    subtext = "#334155"

st.markdown(f"""
<style>
html, body, [data-testid="stAppViewContainer"] {{
    background: {bg};
    color: {text};
}}

section[data-testid="stSidebar"] {{
    width: 420px !important;
    background: linear-gradient(180deg, #1e3a8a, #1e40af);
}}

section[data-testid="stSidebar"] * {{
    color: white !important;
    font-size: 22px !important;
}}

.title {{
    text-align: center;
    font-size: 60px;
    font-weight: 800;
}}

.subtitle {{
    text-align: center;
    font-size: 28px;
    color: {subtext};
    margin-bottom: 30px;
}}

.card {{
    background: {card};
    padding: 40px;
    border-radius: 24px;
    box-shadow: 0px 12px 30px rgba(0,0,0,0.25);
    margin-bottom: 30px;
}}

.good {{
    font-size: 42px;
    font-weight: 900;
    color: #22c55e;
}}

.bad {{
    font-size: 42px;
    font-weight: 900;
    color: #ef4444;
}}

button {{
    font-size: 28px !important;
    padding: 16px 32px !important;
    border-radius: 16px !important;
}}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------
page = st.sidebar.radio("📌 Navigation", ["Predict Churn", "Model Analytics"])

# --------------------------------------------------
# LOAD & TRAIN MODEL
# --------------------------------------------------
@st.cache_data
def train_model():
    df = pd.read_csv(
        r"C:\Users\akhil\Downloads\WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    X = df[["tenure", "MonthlyCharges", "TotalCharges"]]
    y = df["Churn"].map({"Yes": 1, "No": 0})

    X = X[X["TotalCharges"] != " "]
    y = y[X.index]
    X["TotalCharges"] = X["TotalCharges"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    return model, scaler, acc, precision, recall, f1, roc_auc, cm

model, scaler, accuracy, precision, recall, f1, roc_auc, cm = train_model()

# --------------------------------------------------
# PAGE 1: PREDICTION
# --------------------------------------------------
if page == "Predict Churn":

    st.markdown("<div class='title'>📊 Customer Churn Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Predict customer retention using billing data</div>", unsafe_allow_html=True)

    st.sidebar.markdown("## 🧾 Customer Details")
    st.sidebar.markdown("---")

    tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
    monthly = st.sidebar.slider("Monthly Charges (₹)", 0.0, 150.0, 70.0)
    total = st.sidebar.slider("Total Charges (₹)", 0.0, 10000.0, 1000.0)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write(f"**Tenure:** {tenure} months")
    st.write(f"**Monthly Charges:** ₹ {monthly}")
    st.write(f"**Total Charges:** ₹ {total}")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔍 Predict Customer Churn", use_container_width=True):

        with st.spinner("🔄 Analyzing customer behaviour..."):
            time.sleep(1.5)

        input_data = np.array([[tenure, monthly, total]])
        input_scaled = scaler.transform(input_data)
        prob = model.predict_proba(input_scaled)[0][1]

        st.progress(int(prob * 100))
        st.write(f"**Churn Probability:** {prob:.2f}")

        if prob >= 0.5:
            st.markdown("<div class='bad'>🚨 Likely to Churn</div>", unsafe_allow_html=True)
            st.write("Immediate retention strategy recommended.")
        else:
            st.markdown("<div class='good'>✅ Likely to Stay</div>", unsafe_allow_html=True)
            st.write("Customer shows strong loyalty.")

# --------------------------------------------------
# PAGE 2: MODEL ANALYTICS
# --------------------------------------------------
if page == "Model Analytics":

    st.markdown("<div class='title'>📊 Model Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Performance Metrics & Confusion Matrix</div>", unsafe_allow_html=True)

    # Metrics
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("🎯 Accuracy", f"{accuracy * 100:.2f}%")
        st.metric("📌 Precision", f"{precision:.2f}")
        st.metric("🔁 Recall", f"{recall:.2f}")

    with col2:
        st.metric("📐 F1 Score", f"{f1:.2f}")
        st.metric("📈 ROC-AUC", f"{roc_auc:.2f}")

    st.markdown("</div>", unsafe_allow_html=True)

    # Confusion Matrix
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔎 Confusion Matrix")

    cm_df = pd.DataFrame(
        cm,
        columns=["Predicted Stay", "Predicted Churn"],
        index=["Actual Stay", "Actual Churn"]
    )

    st.dataframe(cm_df, use_container_width=True)

    st.markdown("""
    **Explanation:**
    - **True Positive:** Correctly predicted churn customers  
    - **True Negative:** Correctly predicted retained customers  
    - **False Positive:** Retained customers predicted as churn  
    - **False Negative:** Churn customers predicted as retained  
    """)

    st.markdown("</div>", unsafe_allow_html=True)
