import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(page_title="Loan Default Predictor", layout="centered")
st.title("🏦 Loan Default Predictor")
st.markdown("Fill in the applicant details below.")

col1, col2 = st.columns(2)

with col1:
    person_age = st.number_input("Age", 18, 100, 28)
    person_income = st.number_input("Annual Income (₹)", 0, 10000000, 45000)
    loan_amnt = st.number_input("Loan Amount (₹)", 0, 100000, 12000)
    loan_int_rate = st.number_input("Interest Rate (%)", 0.0, 30.0, 14.5)

    credit_score = st.number_input("Credit Score", 300, 850, 620)
    cb_person_cred_hist_length = st.number_input("Credit History (years)", 0, 50, 4)
    previous_loan_defaults_on_file = st.selectbox("Prior Default?", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])

with col2:
    person_gender = st.selectbox("Gender", [("Female", 0), ("Male", 1)], format_func=lambda x: x[0])
    person_education = st.selectbox("Education",
        [("High School", 0), ("Associate", 1), ("Bachelor", 2), ("Master", 3), ("Doctorate", 4)],
        format_func=lambda x: x[0])
    home_ownership = st.selectbox("Home Ownership", ["MORTGAGE", "OTHER", "OWN", "RENT"])
    loan_intent = st.selectbox("Loan Intent", ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"])

if st.button("Predict", type="primary"):
    loan_percent_income = round(loan_amnt / person_income, 4) if person_income > 0 else 0.0
    st.caption(f"📊 Loan % of Income (auto-calculated): **{loan_percent_income:.2%}**")

    payload = {
        "person_age": person_age,
        "person_gender": person_gender[1],
        "person_education": float(person_education[1]),
        "person_income": person_income,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "credit_score": credit_score,
        "previous_loan_defaults_on_file": previous_loan_defaults_on_file[1],
        "person_home_ownership_OTHER": int(home_ownership == "OTHER"),
        "person_home_ownership_OWN": int(home_ownership == "OWN"),
        "person_home_ownership_RENT": int(home_ownership == "RENT"),
        "loan_intent_EDUCATION": int(loan_intent == "EDUCATION"),
        "loan_intent_HOMEIMPROVEMENT": int(loan_intent == "HOMEIMPROVEMENT"),
        "loan_intent_MEDICAL": int(loan_intent == "MEDICAL"),
        "loan_intent_PERSONAL": int(loan_intent == "PERSONAL"),
        "loan_intent_VENTURE": int(loan_intent == "VENTURE"),
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        result = response.json()
    except Exception as e:
        st.error(f"Could not connect to API: {e}")
        st.stop()

    prob = result["default_probability"]
    label = result["label"]

    if label == "Default":
        st.error(f"⚠️ High Risk — {prob*100:.1f}% probability of default")
    else:
        st.success(f"✅ Low Risk — {prob*100:.1f}% probability of default")

    # --- SHAP Explanation Chart ---
    st.markdown("### 🔍 Why this prediction?")
    st.caption("Each bar shows how much a feature pushed the prediction toward Default (+) or Approved (−)")

    drivers = result.get("top_drivers", [])
    if drivers:
        features = [d["feature"].replace("_", " ").title() for d in drivers]
        values   = [d["shap"] for d in drivers]
        colors   = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]

        fig, ax = plt.subplots(figsize=(7, 3.5))
        bars = ax.barh(features[::-1], values[::-1], color=colors[::-1])
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("SHAP Value (impact on default probability)")
        ax.set_title("Top 5 Feature Drivers")

        red_patch  = mpatches.Patch(color="#e74c3c", label="Increases default risk")
        green_patch = mpatches.Patch(color="#2ecc71", label="Decreases default risk")
        ax.legend(handles=[red_patch, green_patch], fontsize=8)

        plt.tight_layout()
        st.pyplot(fig)

        with st.expander("All feature SHAP values"):
            shap_df = pd.DataFrame(
                result["shap_values"].items(), columns=["Feature", "SHAP Value"]
            ).sort_values("SHAP Value", key=abs, ascending=False)
            st.dataframe(shap_df, use_container_width=True)

# To run: streamlit run streamlit_app.py
