# views/6_Churn_Prediction.py
# ============================================================
#  Churn Prediction — Single-customer churn risk scoring and
#  retention action recommendations using the saved pipeline.
# ============================================================

from __future__ import annotations

import os
import sys
import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.features import add_features

PIPELINE_PATH = os.path.join(PROJECT_ROOT, "models", "final_pipeline.pkl")

MODEL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Tenure_Group",
    "Monthly_Charge_Group",
    "Customer_Segment",
    "Contract_Level",
    "Unprotected_Services_Count",
    "has_internet",
    "Payment_Risk",
    "Customer_Value",
]


@st.cache_resource
def load_pipeline():
    if not os.path.isfile(PIPELINE_PATH):
        return None
    return joblib.load(PIPELINE_PATH)


def get_risk_level(prob_pct: float) -> str:
    if prob_pct < 25.0:
        return "Low"
    if prob_pct < 50.0:
        return "Medium"
    if prob_pct < 75.0:
        return "High"
    return "Critical"


st.title("Churn Prediction")
st.caption("Estimates churn risk for one customer based on account, service, and billing details.")

pipeline = load_pipeline()
if pipeline is None:
    st.error(f"Trained model pipeline not found at `{PIPELINE_PATH}`. Run `python -m utils.train_model` first.")
    st.stop()

with st.form("churn_prediction_form"):
    st.subheader("Customer Profile & Service Inputs")
    c1, c2, c3 = st.columns(3)

    with c1:
        gender = st.selectbox("Gender", options=["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen (0 = No, 1 = Yes)", options=[0, 1])
        partner = st.selectbox("Partner", options=["No", "Yes"])
        dependents = st.selectbox("Dependents", options=["No", "Yes"])
        tenure = st.number_input("Tenure (months, 0–72)", min_value=0, max_value=72, value=12, step=1)
        contract = st.selectbox("Contract", options=["Month-to-month", "One year", "Two year"])

    with c2:
        phone_service = st.selectbox("Phone Service", options=["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", options=["No", "Yes", "No Phone Service"])
        internet_service = st.selectbox("Internet Service", options=["Fiber optic", "DSL", "No"])
        online_security = st.selectbox("Online Security", options=["No", "Yes", "No Internet Service"])
        online_backup = st.selectbox("Online Backup", options=["No", "Yes", "No Internet Service"])
        device_protection = st.selectbox("Device Protection", options=["No", "Yes", "No Internet Service"])

    with c3:
        tech_support = st.selectbox("Tech Support", options=["No", "Yes", "No Internet Service"])
        streaming_tv = st.selectbox("Streaming TV", options=["No", "Yes", "No Internet Service"])
        streaming_movies = st.selectbox("Streaming Movies", options=["No", "Yes", "No Internet Service"])
        paperless_billing = st.selectbox("Paperless Billing", options=["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            options=[
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly_charges = st.number_input(
            "Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.5
        )

    submitted = st.form_submit_button("Predict Churn Risk")

if submitted:
    if not (0 <= tenure <= 72) or not (0.0 <= monthly_charges <= 200.0):
        st.error("Invalid input: tenure must be between 0 and 72 months, and MonthlyCharges must be between $0 and $200.")
    else:
        try:
            total_charges = round(float(tenure * monthly_charges), 2)

            raw_row = pd.DataFrame([{
                "gender": gender,
                "SeniorCitizen": int(senior_citizen),
                "Partner": partner,
                "Dependents": dependents,
                "tenure": int(tenure),
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": float(monthly_charges),
                "TotalCharges": total_charges,
            }])

            featured_row = add_features(raw_row)
            for col in featured_row.select_dtypes(include=["Int64"]).columns:
                featured_row[col] = featured_row[col].astype(float)

            X_input = featured_row[MODEL_FEATURES]
            prob = float(pipeline.predict_proba(X_input)[:, 1][0])
            prob_pct = prob * 100.0
            pred_label = "Churn" if prob >= 0.50 else "No churn"
            risk_level = get_risk_level(prob_pct)

            st.subheader("Prediction Result")
            m1, m2, m3 = st.columns(3)
            m1.metric("Churn Probability", f"{prob_pct:.1f}%")
            m2.metric("Predicted Result (Threshold 0.50)", pred_label)
            m3.metric("Risk Level", risk_level)

            # Main risk indicators present for this customer
            indicators: list[str] = []
            if contract == "Month-to-month":
                indicators.append("Month-to-month contract")
            if payment_method == "Electronic check":
                indicators.append("Electronic check payment")
            if internet_service != "No" and tech_support == "No":
                indicators.append("No tech support")
            if internet_service != "No" and online_security == "No":
                indicators.append("No online security")
            if internet_service == "Fiber optic":
                indicators.append("Fiber optic service")
            if tenure < 12:
                indicators.append("Tenure under 12 months")
            if monthly_charges > 70.0:
                indicators.append("Monthly charge above $70")

            st.markdown("#### Main Risk Indicators")
            if indicators:
                for item in indicators:
                    st.markdown(f"- {item}")
            else:
                st.write("No major churn risk indicators detected for this profile.")

            # Suggested retention actions based on risk level and indicators
            st.markdown("#### Suggested Retention Action")
            if risk_level == "Low":
                st.info("Low risk: Standard engagement only.")
            else:
                actions: list[str] = []
                if risk_level in ("High", "Critical") and contract == "Month-to-month":
                    actions.append("Offer a discounted annual contract.")
                if payment_method == "Electronic check":
                    actions.append("Suggest switching to automatic payment.")
                if internet_service != "No" and (tech_support == "No" or online_security == "No"):
                    actions.append("Offer a free trial of protection services (Tech Support / Online Security).")
                if not actions:
                    actions.append("Provide a proactive loyalty check-in and review current plan value.")

                for act in actions:
                    st.markdown(f"- {act}")

            st.caption("Estimates describe patterns in past data and are not guarantees about any individual customer.")

            result_df = pd.DataFrame([{
                **raw_row.iloc[0].to_dict(),
                "Churn_Probability_Pct": round(prob_pct, 1),
                "Predicted_Result_Threshold_0.50": pred_label,
                "Risk_Level": risk_level,
            }])
            st.download_button(
                label="Download Prediction Result (CSV)",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="churn_prediction_result.csv",
                mime="text/csv",
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")