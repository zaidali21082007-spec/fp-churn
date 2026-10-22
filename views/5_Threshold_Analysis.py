# views/5_Threshold_Analysis.py
from __future__ import annotations

import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.preprocessing import load_clean_data
from utils.features import add_features

PIPELINE_PATH = os.path.join(PROJECT_ROOT, "models", "final_pipeline.pkl")


@st.cache_resource
def load_pipeline():
    return joblib.load(PIPELINE_PATH)


@st.cache_data
def get_test_data() -> tuple[pd.DataFrame, pd.Series]:
    df_clean, _ = load_clean_data()
    df = add_features(df_clean)

    # Match dtype casting and feature selection from utils/train_model.py
    for col in df.select_dtypes(include=["Int64"]).columns:
        df[col] = df[col].astype(float)

    y = df["Churn_Binary"].astype(int)
    drop_cols = ["customerID", "Churn", "Churn_Binary"]
    if "TotalCharges" in df.columns and not pd.api.types.is_numeric_dtype(df["TotalCharges"]):
        drop_cols.append("TotalCharges")

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    return X_test, y_test


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    flagged = tp + fp

    # Safe division to handle 0 denominators at extreme thresholds
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "Threshold": round(float(threshold), 2),
        "Flagged High Risk": flagged,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }


st.title("Threshold Analysis")
st.caption(
    "Trade-off: Lowering the decision threshold catches more actual churners (higher recall) "
    "but also flags more customers who would have stayed on their own (more false positives)."
)

pipeline = load_pipeline()
X_test, y_test = get_test_data()
y_true = y_test.to_numpy()
y_prob = pipeline.predict_proba(X_test)[:, 1]

threshold = st.slider(
    "Classification Decision Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.01,
)

current = compute_metrics(y_true, y_prob, threshold)

st.subheader(f"Test Set Performance at Threshold = {threshold:.2f}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Precision", f"{current['Precision']:.2%}")
c2.metric("Recall", f"{current['Recall']:.2%}")
c3.metric("F1 Score", f"{current['F1']:.2%}")
c4.metric("Flagged High Risk", f"{current['Flagged High Risk']:,}")

c5, c6, c7, c8 = st.columns(4)
c5.metric("True Positives (TP)", f"{current['TP']:,}")
c6.metric("False Positives (FP)", f"{current['FP']:,}")
c7.metric("True Negatives (TN)", f"{current['TN']:,}")
c8.metric("False Negatives (FN)", f"{current['FN']:,}")

st.subheader("Threshold Comparison Table")
fixed_thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
table_df = pd.DataFrame([compute_metrics(y_true, y_prob, t) for t in fixed_thresholds])
st.dataframe(table_df, use_container_width=True, hide_index=True)