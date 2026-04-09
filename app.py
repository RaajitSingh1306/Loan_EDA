from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import shap

app = FastAPI()
model = joblib.load("model.pkl")

# Pre-load SHAP explainer once at startup (expensive to create each request)
explainer = shap.TreeExplainer(model)

FEATURE_COLUMNS = [
    "person_age", "person_gender", "person_education", "person_income",
    "loan_amnt", "loan_int_rate", "loan_percent_income",
    "cb_person_cred_hist_length", "credit_score",
    "previous_loan_defaults_on_file",
    "person_home_ownership_OTHER", "person_home_ownership_OWN", "person_home_ownership_RENT",
    "loan_intent_EDUCATION", "loan_intent_HOMEIMPROVEMENT",
    "loan_intent_MEDICAL", "loan_intent_PERSONAL", "loan_intent_VENTURE",
]

class LoanApplication(BaseModel):
    person_age: int
    person_gender: int
    person_education: float
    person_income: int
    loan_amnt: int
    loan_int_rate: float
    loan_percent_income: float
    cb_person_cred_hist_length: int
    credit_score: int
    previous_loan_defaults_on_file: int
    person_home_ownership_OTHER: int
    person_home_ownership_OWN: int
    person_home_ownership_RENT: int
    loan_intent_EDUCATION: int
    loan_intent_HOMEIMPROVEMENT: int
    loan_intent_MEDICAL: int
    loan_intent_PERSONAL: int
    loan_intent_VENTURE: int

@app.get("/")
def root():
    return {"status": "Loan Default Predictor API is running"}

@app.post("/predict")
def predict(data: LoanApplication):
    df = pd.DataFrame([data.model_dump()])[FEATURE_COLUMNS]

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    # SHAP explanation for this single applicant
    shap_vals = explainer.shap_values(df)
    # Handle both old (list) and new (array) shap API
    if isinstance(shap_vals, list):
        sv = shap_vals[1][0]      # older shap: list of [class0_array, class1_array]
    else:
        sv = shap_vals[0, :, 1]   # newer shap: (n_samples, n_features, n_classes)

    shap_dict = {col: round(float(val), 4) for col, val in zip(FEATURE_COLUMNS, sv)}
    top_drivers = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

    return {
        "prediction": int(prediction),
        "label": "Default" if prediction == 1 else "Approved",
        "default_probability": round(float(probability), 4),
        "shap_values": shap_dict,
        "top_drivers": [{"feature": k, "shap": v} for k, v in top_drivers],
    }

# To run: uvicorn app:app --reload
