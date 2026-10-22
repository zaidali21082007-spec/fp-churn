# views/2_Churn_Dashboard.py
# ============================================================
#  Churn Dashboard — Executive KPI summary and contract-level
#  monthly revenue exposure for Telco customer churn.
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

REQUIRED_COLS = ["Churn", "Churn_Binary", "MonthlyCharges", "tenure", "Contract"]


@st.cache_data
def get_dashboard_data() -> pd.DataFrame:
    df_clean, _ = load_clean_data()
    df = add_features(df_clean)
    return df[REQUIRED_COLS].copy()


st.title("Churn Dashboard")
st.caption("Overview of customer retention metrics, monthly recurring revenue, and contract-level churn exposure.")

df = get_dashboard_data()

if df.empty:
    st.warning("No customer records available to compute dashboard metrics.")
else:
    # ── Compute KPI metrics from DataFrame ───────────────────────────────────
    total_customers = len(df)
    active_customers = int((df["Churn"] == "No").sum())
    churned_customers = int((df["Churn"] == "Yes").sum())
    churn_rate = (churned_customers / total_customers) * 100.0

    avg_monthly_charge = float(df["MonthlyCharges"].mean())
    avg_tenure = float(df["tenure"].mean())
    total_monthly_revenue = float(df["MonthlyCharges"].sum())
    churned_revenue = float(df.loc[df["Churn"] == "Yes", "MonthlyCharges"].sum())

    # ── KPI Row 1 ────────────────────────────────────────────────────────────
    r1_c1, r1_c2, r1_c3, r1_c4 = st.columns(4)
    r1_c1.metric("Total Customers", f"{total_customers:,}")
    r1_c2.metric("Active Customers", f"{active_customers:,}")
    r1_c3.metric("Churned Customers", f"{churned_customers:,}")
    r1_c4.metric("Churn Rate", f"{churn_rate:.2f}%")

    # ── KPI Row 2 ────────────────────────────────────────────────────────────
    r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4)
    r2_c1.metric("Average Monthly Charge", f"${avg_monthly_charge:,.2f}")
    r2_c2.metric("Average Tenure", f"{avg_tenure:.1f} months")
    r2_c3.metric("Total Monthly Revenue", f"${total_monthly_revenue:,.2f}")
    r2_c4.metric("Churned Revenue", f"${churned_revenue:,.2f}")

    st.markdown("---")

    # ── Revenue at risk by contract ──────────────────────────────────────────
    st.subheader("Revenue at risk by contract")

    churned_rev_by_contract = (
        df[df["Churn"] == "Yes"]
        .groupby("Contract", observed=True)["MonthlyCharges"]
        .sum()
    )

    contract_tbl = (
        df.groupby("Contract", observed=True)
        .agg(
            Customers=("Churn_Binary", "count"),
            Churned_Customers=("Churn_Binary", "sum"),
        )
        .reset_index()
    )
    contract_tbl["Churn Rate (%)"] = (
        (contract_tbl["Churned_Customers"] / contract_tbl["Customers"]) * 100.0
    ).round(2)
    contract_tbl["Churned Monthly Revenue ($)"] = (
        contract_tbl["Contract"].map(churned_rev_by_contract).fillna(0.0).round(2)
    )

    contract_tbl = (
        contract_tbl.sort_values(by="Churned Monthly Revenue ($)", ascending=False)
        .rename(columns={"Churned_Customers": "Churned Customers"})
        .reset_index(drop=True)
    )

    st.dataframe(contract_tbl, use_container_width="stretch", hide_index=True)