# views/1_Customer_Explorer.py
# ============================================================
#  Customer Explorer — Interactive filtering, preview, and CSV
#  export for the cleaned Telco customer churn dataset.
# ============================================================

from __future__ import annotations

import os
import sys
import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.preprocessing import load_clean_data
from utils.features import add_features

PREVIEW_COLS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "Contract",
    "InternetService",
    "MonthlyCharges",
    "TotalCharges",
    "PaymentMethod",
    "Churn",
    "Customer_Segment",
]


@st.cache_data
def get_explorer_data() -> pd.DataFrame:
    df_clean, report = load_clean_data()
    df = add_features(df_clean)
    # Restore customerID from preprocessing report for customer-level inspection
    df.insert(0, "customerID", report["customer_ids"].values)
    return df


df = get_explorer_data()

st.title("Customer Explorer")
st.caption("Filter, inspect, and export individual customer records across contract, service, demographic, and billing attributes.")

# ── Page-body filters ────────────────────────────────────────────────────────
f_col1, f_col2, f_col3, f_col4 = st.columns(4)

with f_col1:
    contract_opts = sorted(df["Contract"].dropna().unique().tolist())
    sel_contract = st.multiselect("Contract", options=contract_opts, default=contract_opts)

with f_col2:
    internet_opts = sorted(df["InternetService"].dropna().unique().tolist())
    sel_internet = st.multiselect("InternetService", options=internet_opts, default=internet_opts)

with f_col3:
    segment_opts = sorted(df["Customer_Segment"].dropna().unique().tolist())
    sel_segment = st.multiselect("Customer_Segment", options=segment_opts, default=segment_opts)

with f_col4:
    sel_churn = st.selectbox("Churn", options=["All", "Yes", "No"], index=0)

s_col1, s_col2 = st.columns(2)

with s_col1:
    t_min, t_max = int(df["tenure"].min()), int(df["tenure"].max())
    sel_tenure = st.slider("tenure (months)", min_value=t_min, max_value=t_max, value=(t_min, t_max))

with s_col2:
    mc_min, mc_max = float(df["MonthlyCharges"].min()), float(df["MonthlyCharges"].max())
    sel_mc = st.slider(
        "MonthlyCharges ($)",
        min_value=mc_min,
        max_value=mc_max,
        value=(mc_min, mc_max),
    )

# ── Apply all filters before computing any metrics or tables ─────────────────
filtered_df = df[
    df["Contract"].isin(sel_contract)
    & df["InternetService"].isin(sel_internet)
    & df["Customer_Segment"].isin(sel_segment)
    & df["tenure"].between(sel_tenure[0], sel_tenure[1])
    & df["MonthlyCharges"].between(sel_mc[0], sel_mc[1])
]
if sel_churn != "All":
    filtered_df = filtered_df[filtered_df["Churn"] == sel_churn]

filtered_df = filtered_df[PREVIEW_COLS].copy()

# ── Summary metrics (computed strictly from filtered_df) ─────────────────────
n_filtered = len(filtered_df)
if n_filtered > 0:
    churned_rows = int((filtered_df["Churn"] == "Yes").sum())
    filtered_churn_rate = (churned_rows / n_filtered) * 100.0
else:
    filtered_churn_rate = 0.0

m1, m2 = st.columns(2)
m1.metric("Filtered Customers", f"{n_filtered:,}")
m2.metric("Filtered Churn Rate", f"{filtered_churn_rate:.2f}%")

# ── Preview table & export ───────────────────────────────────────────────────
if n_filtered == 0:
    st.warning("No customers match the selected filter criteria. Adjust the filters above to view records.")
else:
    n_preview = st.selectbox("Rows to display", options=[10, 25, 50, 100], index=1)
    st.dataframe(filtered_df.head(n_preview), use_container_width="stretch", hide_index=True)

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Filtered Data (CSV)",
        data=csv_data,
        file_name="telco_filtered_customers.csv",
        mime="text/csv",
    )