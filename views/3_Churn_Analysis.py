# views/3_Churn_Analysis.py
# ============================================================
#  Churn Analysis — Interactive Plotly exploratory analysis
#  across contract, payment, demographic, and service drivers.
#  NOTE: Add "plotly" to requirements.txt (pip install plotly)
# ============================================================

from __future__ import annotations

import os
import sys
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.preprocessing import load_clean_data
from utils.features import add_features

CHURN_COLORS = {"No": "#22c55e", "Yes": "#ef4444"}
TENURE_ORDER = [
    "New Customer",
    "Developing Customer",
    "Established Customer",
    "Loyal Customer",
]


@st.cache_data
def get_analysis_data() -> pd.DataFrame:
    df_clean, _ = load_clean_data()
    return add_features(df_clean)


df = get_analysis_data()

st.title("Churn Analysis")
st.caption(
    "Interactive visual analysis of demographic, contract, service, and billing factors associated with customer churn."
)

# ── Page-body filters ────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)

with c1:
    contract_opts = sorted(df["Contract"].dropna().unique().tolist())
    sel_contract = st.multiselect("Contract", options=contract_opts, default=contract_opts)

with c2:
    internet_opts = sorted(df["InternetService"].dropna().unique().tolist())
    sel_internet = st.multiselect("InternetService", options=internet_opts, default=internet_opts)

with c3:
    segment_opts = sorted(df["Customer_Segment"].dropna().unique().tolist())
    sel_segment = st.multiselect("Customer_Segment", options=segment_opts, default=segment_opts)

t_min, t_max = int(df["tenure"].min()), int(df["tenure"].max())
sel_tenure = st.slider(
    "Tenure (months)", min_value=t_min, max_value=t_max, value=(t_min, t_max)
)

filtered_df = df[
    df["Contract"].isin(sel_contract)
    & df["InternetService"].isin(sel_internet)
    & df["Customer_Segment"].isin(sel_segment)
    & df["tenure"].between(sel_tenure[0], sel_tenure[1])
].copy()

if filtered_df.empty:
    st.warning("No customers match the selected filters. Adjust the filter criteria above.")
    st.stop()

n_filtered = len(filtered_df)
filtered_churn_rate = float(filtered_df["Churn_Binary"].mean() * 100.0)

m1, m2 = st.columns(2)
m1.metric("Filtered Customers", f"{n_filtered:,}")
m2.metric("Filtered Churn Rate", f"{filtered_churn_rate:.2f}%")

st.markdown("---")

# ── 1. Churn Distribution ────────────────────────────────────────────────────
st.subheader("1. Churn Distribution")
dist_df = (
    filtered_df.groupby("Churn", observed=True)
    .size()
    .reset_index(name="Customers")
)
fig1 = px.bar(
    dist_df,
    x="Churn",
    y="Customers",
    color="Churn",
    text="Customers",
    color_discrete_map=CHURN_COLORS,
    title="Customer Count by Churn Status",
    labels={"Churn": "Churn Status", "Customers": "Number of Customers"},
)
st.plotly_chart(fig1, width="stretch")
st.caption("Business question: How many customers in the selected segment stayed versus churned?")

# ── 2. Churn Rate by Contract ────────────────────────────────────────────────
st.subheader("2. Churn Rate by Contract")
contract_df = (
    filtered_df.groupby("Contract", observed=True)["Churn_Binary"]
    .agg(Customers="count", Churn_Rate=lambda x: round(x.mean() * 100.0, 2))
    .reset_index()
    .sort_values("Churn_Rate", ascending=False)
)
fig2 = px.bar(
    contract_df,
    x="Contract",
    y="Churn_Rate",
    text="Churn_Rate",
    title="Churn Rate (%) by Contract Type",
    labels={"Contract": "Contract Type", "Churn_Rate": "Churn Rate (%)"},
)
st.plotly_chart(fig2, width="stretch")
st.caption("Business question: How is contract length associated with customer churn rate?")

# ── 3. Churn Rate by Payment Method ──────────────────────────────────────────
st.subheader("3. Churn Rate by Payment Method")
pm_df = (
    filtered_df.groupby("PaymentMethod", observed=True)["Churn_Binary"]
    .agg(Customers="count", Churn_Rate=lambda x: round(x.mean() * 100.0, 2))
    .reset_index()
    .sort_values("Churn_Rate", ascending=True)
)
fig3 = px.bar(
    pm_df,
    x="Churn_Rate",
    y="PaymentMethod",
    orientation="h",
    text="Churn_Rate",
    title="Churn Rate (%) by Payment Method",
    labels={"PaymentMethod": "Payment Method", "Churn_Rate": "Churn Rate (%)"},
)
st.plotly_chart(fig3, width="stretch")
st.caption("Business question: Which payment methods are associated with higher churn rates?")

# ── 4. Churn Rate by Auto-Pay ────────────────────────────────────────────────
st.subheader("4. Churn Rate by Auto-Pay Enrollment")
autopay_df = (
    filtered_df.assign(
        Auto_Pay=filtered_df["PaymentMethod"]
        .str.contains("automatic", case=False, na=False)
        .map({True: "Auto-Pay Enrolled", False: "Manual Payment"})
    )
    .groupby("Auto_Pay", observed=True)["Churn_Binary"]
    .agg(Customers="count", Churn_Rate=lambda x: round(x.mean() * 100.0, 2))
    .reset_index()
)
fig4 = px.bar(
    autopay_df,
    x="Auto_Pay",
    y="Churn_Rate",
    color="Auto_Pay",
    text="Churn_Rate",
    title="Churn Rate (%): Auto-Pay vs. Manual Payment",
    labels={"Auto_Pay": "Payment Type", "Churn_Rate": "Churn Rate (%)"},
)
st.plotly_chart(fig4, width="stretch")
st.caption(
    "Business question: Is automatic payment enrollment associated with lower churn compared with manual payment?"
)

# ── 5. Monthly Charges by Churn ──────────────────────────────────────────────
st.subheader("5. Monthly Charges by Churn")
fig5 = px.box(
    filtered_df,
    x="Churn",
    y="MonthlyCharges",
    color="Churn",
    color_discrete_map=CHURN_COLORS,
    title="Monthly Charges ($) Distribution by Churn Status",
    labels={"Churn": "Churn Status", "MonthlyCharges": "Monthly Charges ($)"},
)
st.plotly_chart(fig5, width="stretch")
st.caption("Business question: Are higher monthly charges associated with customers who churn?")

# ── 6. Tenure by Churn ───────────────────────────────────────────────────────
st.subheader("6. Tenure by Churn")
fig6 = px.box(
    filtered_df,
    x="Churn",
    y="tenure",
    color="Churn",
    color_discrete_map=CHURN_COLORS,
    title="Tenure (Months) Distribution by Churn Status",
    labels={"Churn": "Churn Status", "tenure": "Tenure (Months)"},
)
st.plotly_chart(fig6, width="stretch")
st.caption("Business question: How is customer tenure associated with retention versus churn?")

# ── 7. Churn Rate by Customer Segment ────────────────────────────────────────
st.subheader("7. Churn Rate by Customer Segment")
seg_df = (
    filtered_df.groupby("Customer_Segment", observed=True)["Churn_Binary"]
    .agg(Customers="count", Churn_Rate=lambda x: round(x.mean() * 100.0, 2))
    .reset_index()
    .sort_values("Churn_Rate", ascending=False)
)
fig7 = px.bar(
    seg_df,
    x="Customer_Segment",
    y="Churn_Rate",
    text="Churn_Rate",
    title="Churn Rate (%) by Customer Segment",
    labels={"Customer_Segment": "Customer Segment", "Churn_Rate": "Churn Rate (%)"},
)
st.plotly_chart(fig7, width="stretch")
st.caption("Business question: Which demographic customer segment has the strongest association with churn?")

# ── 8. Churn Rate by Tenure Group ────────────────────────────────────────────
st.subheader("8. Churn Rate by Tenure Group")
tg_df = (
    filtered_df.groupby("Tenure_Group", observed=True)["Churn_Binary"]
    .agg(Customers="count", Churn_Rate=lambda x: round(x.mean() * 100.0, 2))
    .reindex([g for g in TENURE_ORDER if g in filtered_df["Tenure_Group"].unique()])
    .reset_index()
)
fig8 = px.bar(
    tg_df,
    x="Tenure_Group",
    y="Churn_Rate",
    text="Churn_Rate",
    category_orders={"Tenure_Group": TENURE_ORDER},
    title="Churn Rate (%) by Tenure Lifecycle Group",
    labels={"Tenure_Group": "Tenure Group", "Churn_Rate": "Churn Rate (%)"},
)
st.plotly_chart(fig8, width="stretch")
st.caption("Business question: How does the churn rate vary across tenure lifecycle stages?")

# ── 9. Support Columns by Churn (Internet Customers Only) ────────────────────
st.subheader("9. Support & Security Services by Churn (Internet Customers Only)")
inet_df = filtered_df[filtered_df["InternetService"] != "No"]

if inet_df.empty:
    st.info("Select DSL or Fiber optic in the InternetService filter to view support service associations.")
else:
    sup_rows = []
    for col, label in [("TechSupport", "Tech Support"), ("OnlineSecurity", "Online Security")]:
        for status in ["Yes", "No"]:
            sub = inet_df[inet_df[col] == status]
            rate = round(float(sub["Churn_Binary"].mean() * 100.0), 2) if len(sub) > 0 else 0.0
            sup_rows.append({"Service": label, "Status": status, "Churn_Rate": rate})

    sup_df = pd.DataFrame(sup_rows)
    fig9 = px.bar(
        sup_df,
        x="Service",
        y="Churn_Rate",
        color="Status",
        barmode="group",
        text="Churn_Rate",
        color_discrete_map={"Yes": "#22c55e", "No": "#ef4444"},
        title="Churn Rate (%) by Tech Support & Online Security (Internet Customers Only)",
        labels={"Service": "Service", "Churn_Rate": "Churn Rate (%)", "Status": "Enrolled"},
    )
    st.plotly_chart(fig9, width="stretch")
st.caption(
    "Business question: Among internet subscribers, how strongly is lacking Tech Support or Online Security associated with churn?"
)

# ── 10. Correlation Heatmap ──────────────────────────────────────────────────
st.subheader("10. Correlation Heatmap")
corr_cols = ["tenure", "MonthlyCharges", "TotalCharges", "Churn_Binary"]
corr_matrix = filtered_df[corr_cols].corr(method="pearson").round(2)

fig10 = px.imshow(
    corr_matrix,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1.0,
    zmax=1.0,
    title="Pearson Correlation Heatmap: Tenure, Charges & Churn_Binary",
    labels={"color": "Pearson r"},
)
st.plotly_chart(fig10, width="stretch")
st.caption(
    "Business question: How strongly are tenure, monthly charges, and total charges linearly associated with binary churn?"
)