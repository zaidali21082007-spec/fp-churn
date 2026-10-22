# views/7_Retention_Dashboard.py
# ============================================================
#  Retention Dashboard — Scores all customers with the saved
#  pipeline to quantify revenue at stake and prioritize actions.
# ============================================================

from __future__ import annotations

import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.preprocessing import load_clean_data
from utils.features import add_features

PIPELINE_PATH = os.path.join(PROJECT_ROOT, "models", "final_pipeline.pkl")


@st.cache_resource
def load_pipeline():
    if not os.path.isfile(PIPELINE_PATH):
        return None
    return joblib.load(PIPELINE_PATH)


@st.cache_data
def get_featured_data() -> pd.DataFrame:
    df_clean, report = load_clean_data()
    df = add_features(df_clean)
    df.insert(0, "customerID", report["customer_ids"].values)
    for col in df.select_dtypes(include=["Int64"]).columns:
        df[col] = df[col].astype(float)
    return df


def build_risk_table(df_all: pd.DataFrame, hr_df: pd.DataFrame, group_col: str, label_col: str) -> pd.DataFrame:
    total_per_group = df_all.groupby(group_col, observed=True).size()
    risk_per_group = hr_df.groupby(group_col, observed=True).size()
    rev_per_group = hr_df.groupby(group_col, observed=True)["Expected_Rev_Risk"].sum()

    tbl = pd.DataFrame({
        label_col: total_per_group.index,
        "Customers at risk": [int(risk_per_group.get(g, 0)) for g in total_per_group.index],
        "Total_Group": total_per_group.values,
        "Expected monthly revenue at risk ($)": [
            round(float(rev_per_group.get(g, 0.0)), 2) for g in total_per_group.index
        ],
    })
    tbl["Share of segment (%)"] = (
        (tbl["Customers at risk"] / tbl["Total_Group"]) * 100.0
    ).round(2)

    cols = [
        label_col,
        "Customers at risk",
        "Share of segment (%)",
        "Expected monthly revenue at risk ($)",
    ]
    return (
        tbl[cols]
        .sort_values(by="Expected monthly revenue at risk ($)", ascending=False)
        .reset_index(drop=True)
    )


st.title("Retention Dashboard")
st.caption("Identifies high-risk customers and the revenue at stake, to prioritize retention actions.")
st.caption(
    "Note: scores cover all customers, including the 80% the model was trained on, "
    "so they can run higher than out-of-sample results. Held-out figures are in the "
    "notebook (Section 6)."
)

pipeline = load_pipeline()
if pipeline is None:
    st.error(f"Saved model pipeline not found at `{PIPELINE_PATH}`. Run `python -m utils.train_model` first.")
    st.stop()

df = get_featured_data()

# Score every customer using only the model's training features
feature_cols = list(pipeline.feature_names_in_)
X = df[feature_cols].copy()
probs = pipeline.predict_proba(X)[:, 1]

df["Churn_Probability"] = probs
df["Expected_Rev_Risk"] = df["MonthlyCharges"] * df["Churn_Probability"]
df["Risk_Level"] = np.select(
    [
        df["Churn_Probability"] < 0.25,
        df["Churn_Probability"] < 0.50,
        df["Churn_Probability"] < 0.75,
    ],
    ["Low", "Medium", "High"],
    default="Critical",
)

# ── Compute KPI values ───────────────────────────────────────────────────────
total_customers = len(df)
hr_mask = df["Risk_Level"].isin(["High", "Critical"])
crit_mask = df["Risk_Level"] == "Critical"

# Shared cutoff variable for High+Critical and Model-Flagged customers
PROB_CUTOFF = 0.50
flagged_mask = df["Churn_Probability"] >= PROB_CUTOFF

hr_df = df[hr_mask].copy()
n_high_risk = int(hr_mask.sum())
high_risk_share = (n_high_risk / total_customers * 100.0) if total_customers > 0 else 0.0
n_critical = int(crit_mask.sum())
expected_rev_at_risk = float(hr_df["Expected_Rev_Risk"].sum())

actual_churned_rev = float(df.loc[df["Churn_Binary"] == 1, "MonthlyCharges"].sum())
n_flagged = int(flagged_mask.sum())
n_tp = int((flagged_mask & (df["Churn_Binary"] == 1)).sum())

# ── KPI Row 1 ────────────────────────────────────────────────────────────────
r1_c1, r1_c2, r1_c3, r1_c4 = st.columns(4)
r1_c1.metric("High-Risk Customers (High + Critical)", f"{n_high_risk:,}")
r1_c2.metric("High-Risk Share", f"{high_risk_share:.2f}%")
r1_c3.metric("Critical Customers (>= 75%)", f"{n_critical:,}")
r1_c4.metric("Expected Monthly Revenue at Risk", f"${expected_rev_at_risk:,.2f}")

# ── KPI Row 2 ────────────────────────────────────────────────────────────────
r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4)
r2_c1.metric("Actual Churned Monthly Revenue", f"${actual_churned_rev:,.2f}")
r2_c2.metric(f"Model-Flagged Customers (>= {PROB_CUTOFF:.2f})", f"{n_flagged:,}")
r2_c3.metric(f"True Positives at Threshold {PROB_CUTOFF:.2f}", f"{n_tp:,}")

st.markdown("---")

if hr_df.empty:
    st.success("No customers are currently classified in the High or Critical risk tiers.")
else:
    # ── High-risk tables by Customer_Segment and Contract ────────────────────
    st.subheader("High-Risk Customers by Customer Segment")
    seg_tbl = build_risk_table(df, hr_df, "Customer_Segment", "Segment")
    st.dataframe(seg_tbl, width="stretch", hide_index=True)

    st.subheader("High-Risk Customers by Contract")
    con_tbl = build_risk_table(df, hr_df, "Contract", "Contract")
    st.dataframe(con_tbl, width="stretch", hide_index=True)

    # Check if one-year and two-year customers have zero High or Critical predictions
    one_two_yr_mask = con_tbl["Contract"].isin(["One year", "Two year"])
    one_two_yr_risk_sum = int(con_tbl.loc[one_two_yr_mask, "Customers at risk"].sum()) if not con_tbl[one_two_yr_mask].empty else 0
    if one_two_yr_risk_sum == 0:
        st.caption("One-year and two-year customers have no High or Critical predictions at the current cutoffs.")

    # ── Dynamic campaign recommendation ──────────────────────────────────────
    top_seg_row = seg_tbl.iloc[0]
    top_con_row = con_tbl.iloc[0]
    top_segment = str(top_seg_row["Segment"])
    top_contract = str(top_con_row["Contract"])

    if top_contract == "Month-to-month":
        action_text = "Offer a discounted annual contract with tech support and online security included."
    else:
        action_text = "Schedule a proactive account review and promote automatic payment enrollment."

    st.info(
        f"Recommended Retention Campaign — Top-risk segment: {top_segment} "
        f"({int(top_seg_row['Customers at risk']):,} customers) | "
        f"Top-risk contract: {top_contract} "
        f"({int(top_con_row['Customers at risk']):,} customers). "
        f"Expected monthly revenue at risk: ${expected_rev_at_risk:,.2f}. "
        f"{action_text} (Risk scores reflect historical associations, not causal effects.)"
    )

    # ── Downloadable High & Critical customer list ───────────────────────────
    st.subheader("High & Critical Risk Customer List")

    export_cols = [
        "customerID",
        "Contract",
        "Customer_Segment",
        "PaymentMethod",
        "tenure",
        "MonthlyCharges",
        "Churn_Probability",
        "Risk_Level",
    ]
    export_df = (
        hr_df.sort_values(by="Churn_Probability", ascending=False)[export_cols]
        .assign(Churn_Probability=lambda x: x["Churn_Probability"].round(2))
        .reset_index(drop=True)
    )

    st.dataframe(export_df.head(50), width="stretch", hide_index=True)

    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download High-Risk Customers (CSV)",
        data=csv_bytes,
        file_name="high_risk_retention_targets.csv",
        mime="text/csv",
    )