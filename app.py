import streamlit as st
import joblib
import pandas as pd
from src.processor import download_data
from src.visualization import plot_shap_explanation

# 1. SET PAGE CONFIG
st.set_page_config(
    page_title="CreditRisk-XAI | Stability-Focused System",
    page_icon="🛡️",
    layout="wide"
)

# 2. LOAD CUSTOM CSS
try:
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.info("Style.css not found, using default Streamlit theme.")

# 3. LOAD ARTIFACTS
@st.cache_resource
def load_artifacts():
    model = joblib.load('models/credit_risk_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    features = joblib.load('models/features.pkl')
    explainer = joblib.load('models/shap_explainer.pkl')
    threshold = 0.05
    return model, scaler, features, threshold, explainer

model, scaler, feature_names, threshold, explainer = load_artifacts()

# 4. HEADER
st.title("🛡️ CreditRisk-XAI")
st.markdown("### **Stability-Focused Decision System**")
st.write("This system uses Advanced XGBoost and SHAP explainability to assess creditworthiness based on employment stability and financial profiles.")

st.divider()

# 5. INPUT FORM
st.header("📋 Applicant Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Personal Details")
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age (Years)", 18, 80, 30)
    children = st.number_input("Number of Children", 0, 10, 0)
    education = st.selectbox("Education Level", [
        "Secondary / secondary special",
        "Higher education",
        "Incomplete higher",
        "Lower secondary",
        "Academic degree"
    ])
    family_status = st.selectbox("Family Status", ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"])

with col2:
    st.subheader("💼 Employment")
    income_type = st.selectbox("Income Type", ["Working", "Commercial associate", "State servant", "Pensioner", "Student"])
    years_employed = st.slider("Years of Employment", 0, 50, 5)
    occupation = st.selectbox("Occupation", [
        "Laborers", "Core staff", "Sales staff", "Managers", "Drivers",
        "High skill tech staff", "Accountants", "Medicine staff", "IT staff", "Other"
    ])
    own_car = st.radio("Owns a Car?", ["Yes", "No"], horizontal=True)
    own_realty = st.radio("Owns Real Estate?", ["Yes", "No"], horizontal=True)

with col3:
    st.subheader("💰 Financials")
    total_income = st.number_input("Total Annual Income ($)", min_value=0, value=50000, step=1000)
    housing_type = st.selectbox("Housing Type", [
        "House / apartment", "Rented apartment", "With parents",
        "Municipal apartment", "Office apartment", "Co-op apartment"
    ])
    flag_phone = st.checkbox("Has a Work Phone?")
    flag_email = st.checkbox("Has an Email Address?")

# 6. DATA PRE-PROCESSING
if st.button("🔍 ANALYZE APPLICATION", use_container_width=True):

    raw_data = {
        'CODE_GENDER': 1 if gender == "Female" else 0,
        'FLAG_OWN_CAR': 1 if own_car == "Yes" else 0,
        'FLAG_OWN_REALTY': 1 if own_realty == "Yes" else 0,
        'CNT_CHILDREN': children,
        'AMT_INCOME_TOTAL': total_income,
        'AGE': age,
        'YEARS_EMPLOYED': years_employed,
        'FLAG_PHONE': 1 if flag_phone else 0,
        'FLAG_EMAIL': 1 if flag_email else 0,
        'work_life_ratio': years_employed / age if age > 0 else 0,
        'income_per_capita': total_income / (children + 1)
    }

    input_df = pd.DataFrame([raw_data])

    # One-hot encoding
    input_df[f"NAME_EDUCATION_TYPE_{education}"] = 1
    input_df[f"NAME_INCOME_TYPE_{income_type}"] = 1
    input_df[f"NAME_FAMILY_STATUS_{family_status}"] = 1
    input_df[f"NAME_HOUSING_TYPE_{housing_type}"] = 1
    if occupation != "Other":
        input_df[f"OCCUPATION_TYPE_{occupation}"] = 1

    # Align columns to training features
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    final_df = input_df[feature_names]

    # Scale
    scaled_data = scaler.transform(final_df)

    # Predict
    risk_proba = model.predict_proba(scaled_data)[:, 1][0]
    is_bad = 1 if risk_proba >= threshold else 0

    # 7. DISPLAY RESULTS
    st.divider()
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        if is_bad:
            st.error("### ❌ APPLICATION REJECTED")
            st.metric("Risk Score", f"{risk_proba:.2%}")
        else:
            st.success("### ✅ APPLICATION APPROVED")
            st.metric("Risk Score", f"{risk_proba:.2%}")

    with res_col2:
        st.write("**Model Reasoning:**")
        if is_bad:
            st.warning("This applicant shows characteristics often associated with high default rates in our historical data.")
        else:
            st.info("This applicant meets the stability and financial thresholds for credit approval.")

        st.caption(f"Decision Threshold: {threshold} | Analysis Date: 2026")

    with st.expander("🔍 Deep Dive: Why this decision?"):
        plot_shap_explanation(explainer, scaled_data, feature_names)

# 8. FOOTER
st.sidebar.info(f"Connected to Kaggle Source. Model: XGBoost v1.2")