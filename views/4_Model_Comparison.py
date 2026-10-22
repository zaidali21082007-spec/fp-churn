# views/4_Model_Comparison.py
# ============================================================
#  Model Comparison — Evaluates four trained classifiers from
#  models/metrics.csv and presents the selected model, curves,
#  feature importances, and classification report.
# ============================================================

from __future__ import annotations

import os
import sys
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.preprocessing import load_clean_data
from utils.features import add_features

METRICS_PATH = os.path.join(PROJECT_ROOT, "models", "metrics.csv")
PIPELINE_PATH = os.path.join(PROJECT_ROOT, "models", "final_pipeline.pkl")

DISPLAY_COLS = [
    "Model",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC_AUC",
    "TN",
    "FP",
    "FN",
    "TP",
]
FLOAT_COLS = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]


@st.cache_data
def load_metrics() -> pd.DataFrame | None:
    if not os.path.isfile(METRICS_PATH):
        return None
    df = pd.read_csv(METRICS_PATH)
    return df[[c for c in DISPLAY_COLS if c in df.columns]].copy()


@st.cache_resource
def load_pipeline():
    if not os.path.isfile(PIPELINE_PATH):
        return None
    return joblib.load(PIPELINE_PATH)


@st.cache_data
def load_featured_data() -> pd.DataFrame:
    df_clean, _ = load_clean_data()
    df = add_features(df_clean)
    for col in df.select_dtypes(include=["Int64"]).columns:
        df[col] = df[col].astype(float)
    return df


st.title("Model Comparison")
st.caption(
    "Compares four classifiers trained on the same stratified split and explains the selected model."
)

metrics_df = load_metrics()
if metrics_df is None or metrics_df.empty:
    st.error(
        f"Metrics file not found at `{METRICS_PATH}`. Run `python -m utils.train_model` first."
    )
    st.stop()

# Select RandomForestClassifier if present; otherwise fall back to top-ranked row
rf_mask = metrics_df["Model"] == "RandomForestClassifier"
selected_row = metrics_df[rf_mask].iloc[0] if rf_mask.any() else metrics_df.iloc[0]
selected_name = str(selected_row["Model"])

# Compute test-set churn rate dynamically from confusion matrix counts in CSV
total_test = int(
    selected_row["TN"] + selected_row["FP"] + selected_row["FN"] + selected_row["TP"]
)
churn_test = int(selected_row["FN"] + selected_row["TP"])
churn_pct = (churn_test / total_test) * 100.0 if total_test > 0 else 0.0

st.success(
    f"**Selected Model: {selected_name}** — ROC-AUC: `{selected_row['ROC_AUC']:.3f}`, "
    f"Recall: `{selected_row['Recall']:.3f}`, Precision: `{selected_row['Precision']:.3f}`, "
    f"F1: `{selected_row['F1']:.3f}`."
)

st.subheader("Classifier Evaluation Metrics")
styled_metrics = metrics_df.style.format({col: "{:.3f}" for col in FLOAT_COLS})
st.dataframe(styled_metrics, width="stretch", hide_index=True)

st.info(
    f"Models are close on ROC-AUC, so the final choice depends on recall and the business "
    f"trade-off between missed churners and false alarms. Accuracy alone is misleading "
    f"because {churn_pct:.1f}% of customers churn."
)

# ── Bar chart: ROC_AUC and Recall across models ──────────────────────────────
st.subheader("ROC-AUC and Recall by Model")

fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="#0f172a")
ax.set_facecolor("#1e293b")

x = np.arange(len(metrics_df))
width = 0.35

roc_vals = metrics_df["ROC_AUC"].to_numpy()
rec_vals = metrics_df["Recall"].to_numpy()

bars_roc = ax.bar(x - width / 2, roc_vals, width, label="ROC-AUC", color="#38bdf8")
bars_rec = ax.bar(x + width / 2, rec_vals, width, label="Recall", color="#22c55e")

for bar in bars_roc:
    h = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        h + 0.015,
        f"{h:.3f}",
        ha="center",
        va="bottom",
        color="#e2e8f0",
        fontsize=9,
    )

for bar in bars_rec:
    h = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        h + 0.015,
        f"{h:.3f}",
        ha="center",
        va="bottom",
        color="#e2e8f0",
        fontsize=9,
    )

ax.set_title("Model Comparison: ROC-AUC vs Recall", color="#e2e8f0", fontsize=13, fontweight="bold")
ax.set_xlabel("Model", color="#e2e8f0", fontsize=11)
ax.set_ylabel("Score", color="#e2e8f0", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(metrics_df["Model"], color="#cbd5e1", fontsize=10)
ax.tick_params(axis="y", colors="#cbd5e1")
ax.set_ylim(0, max(float(roc_vals.max()), float(rec_vals.max())) * 1.18)
ax.grid(axis="y", color="#334155", linewidth=0.6, alpha=0.7)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["bottom", "left"]:
    ax.spines[spine].set_color("#334155")
ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")

st.pyplot(fig)
plt.close(fig)

# ── 2x2 Confusion Matrix Table for Selected Model ────────────────────────────
st.subheader(f"Confusion Matrix — {selected_name}")

cm_df = pd.DataFrame(
    [
        [int(selected_row["TN"]), int(selected_row["FP"])],
        [int(selected_row["FN"]), int(selected_row["TP"])],
    ],
    index=["Actual Retained", "Actual Churned"],
    columns=["Predicted Retained", "Predicted Churned"],
)
st.table(cm_df)

# ── Load Saved Pipeline & Recreate Exact Test Split ──────────────────────────
pipeline = load_pipeline()
if pipeline is None:
    st.error(
        f"Saved pipeline not found at `{PIPELINE_PATH}`. Run `python -m utils.train_model` first."
    )
    st.stop()

df_all = load_featured_data()
feature_cols = list(pipeline.feature_names_in_)
X = df_all[feature_cols].copy()
y = df_all["Churn_Binary"].astype(int)

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

y_prob = pipeline.predict_proba(X_test)[:, 1]
y_pred = pipeline.predict(X_test)
test_roc_auc = float(roc_auc_score(y_test, y_prob))

st.markdown("---")
st.metric("Verified Test-Set ROC-AUC (Random Forest)", f"{test_roc_auc:.3f}")

# ── 1. ROC Curve (Random Forest, Test Set) ───────────────────────────────────
st.subheader("1. ROC Curve (Random Forest — Test Set)")

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc_val = float(auc(fpr, tpr))

fig_roc = go.Figure()
fig_roc.add_trace(
    go.Scatter(
        x=fpr,
        y=tpr,
        mode="lines",
        name=f"Random Forest (AUC = {roc_auc_val:.3f})",
        line=dict(color="#38bdf8", width=2.5),
    )
)
fig_roc.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random Chance",
        line=dict(color="#94a3b8", width=1.5, dash="dash"),
    )
)
fig_roc.update_layout(
    title=f"ROC Curve — Random Forest (AUC = {roc_auc_val:.3f})",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate (Recall)",
)
st.plotly_chart(fig_roc, width="stretch")
st.caption("Curves cover the saved Random Forest only.")

# ── 2. Precision-Recall Curve (Random Forest, Test Set) ──────────────────────
st.subheader("2. Precision-Recall Curve (Random Forest — Test Set)")

precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
avg_prec = float(average_precision_score(y_test, y_prob))

fig_pr = go.Figure()
fig_pr.add_trace(
    go.Scatter(
        x=recall_vals,
        y=precision_vals,
        mode="lines",
        name=f"Random Forest (AP = {avg_prec:.3f})",
        line=dict(color="#22c55e", width=2.5),
    )
)
fig_pr.update_layout(
    title=f"Precision-Recall Curve — Random Forest (Average Precision = {avg_prec:.3f})",
    xaxis_title="Recall (Sensitivity)",
    yaxis_title="Precision (Positive Predictive Value)",
)
st.plotly_chart(fig_pr, width="stretch")
st.caption(
    f"Precision-recall is more informative than ROC when churners are the minority class (about {churn_pct:.1f}%)."
)

# ── 3. Feature Importance (Top 15) ───────────────────────────────────────────
st.subheader("3. Top 15 Feature Importances (Random Forest)")

try:
    feat_names = pipeline[:-1].get_feature_names_out()
    importances = pipeline.steps[-1][1].feature_importances_
    fi_df = (
        pd.DataFrame({"Feature": feat_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(15)
        .sort_values("Importance", ascending=True)
    )
    fig_fi = px.bar(
        fi_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 15 Feature Importances — Random Forest",
        labels={"Importance": "Gini Importance", "Feature": "Transformed Feature"},
    )
    st.plotly_chart(fig_fi, width="stretch")
    st.caption(
        "Importance measures how much the model relies on a feature, not whether the feature causes churn."
    )
except Exception as exc:
    st.error(f"Could not extract feature names: {exc}")

# ── 4. Classification Report (Default Threshold 0.50) ────────────────────────
st.subheader("4. Classification Report (Random Forest — Threshold 0.50)")

report_text = classification_report(
    y_test, y_pred, target_names=["Retained", "Churned"], digits=3
)
report_dict = classification_report(
    y_test, y_pred, target_names=["Retained", "Churned"], output_dict=True
)
report_df = pd.DataFrame(report_dict).T.round(3).reset_index(names="Class / Metric")

st.code(report_text, language="text")
st.dataframe(report_df, width="stretch", hide_index=True)