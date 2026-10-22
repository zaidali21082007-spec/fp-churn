# utils/features.py
"""
utils/features.py
=================
Step 3 — Feature Engineering

Public API
----------
    add_features(df: pd.DataFrame) -> pd.DataFrame
"""

from __future__ import annotations

import inspect
import numpy as np
import pandas as pd

_TENURE_BINS   = [0, 12, 24, 48, 72]
_TENURE_LABELS = [
    "New Customer",
    "Developing Customer",
    "Established Customer",
    "Loyal Customer",
]

_MC_BINS   = [0, 35, 65, 90, 120]
_MC_LABELS = [
    "Low (<$35)",
    "Medium ($35–$65)",
    "High ($65–$90)",
    "Premium (>$90)",
]

_PROTECTION_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
]

_CONTRACT_MAP = {
    "Month-to-month": 0,
    "One year":       1,
    "Two year":       2,
}

_PAYMENT_RISK_MAP = {
    "Electronic check":          "High",
    "Mailed check":              "Medium",
    "Bank transfer (automatic)": "Low",
    "Credit card (automatic)":   "Low",
}


def _tenure_group(s: pd.Series) -> pd.Series:
    return pd.cut(
        s,
        bins=_TENURE_BINS,
        labels=_TENURE_LABELS,
        right=True,
        include_lowest=True,
    ).astype(str)


def _monthly_charge_group(s: pd.Series) -> pd.Series:
    return pd.cut(
        s,
        bins=_MC_BINS,
        labels=_MC_LABELS,
        right=True,
        include_lowest=True,
    ).astype(str)


def _customer_segment(df: pd.DataFrame) -> pd.Series:
    is_senior = df["SeniorCitizen"] == 1
    is_family = (df["Partner"] == "Yes") | (df["Dependents"] == "Yes")

    seg = pd.Series("Solo", index=df.index, dtype=str)
    seg[ is_family & ~is_senior] = "Family"
    seg[~is_family &  is_senior] = "Senior Solo"
    seg[ is_family &  is_senior] = "Senior Family"
    return seg


def _contract_level(s: pd.Series) -> pd.Series:
    return s.map(_CONTRACT_MAP).astype("Int64")


def _unprotected_services_count(df: pd.DataFrame) -> pd.Series:
    """
    Count 'No' across OnlineSecurity, OnlineBackup, DeviceProtection, and TechSupport
    for internet customers. Customers with no internet service receive NaN.
    """
    counts = pd.Series(0.0, index=df.index, dtype=float)
    for col in _PROTECTION_COLS:
        counts += (df[col] == "No").astype(float)
    counts[df["InternetService"] == "No"] = np.nan
    return counts


def _has_internet(df: pd.DataFrame) -> pd.Series:
    return (df["InternetService"] != "No").astype(int)


def _payment_risk(s: pd.Series) -> pd.Series:
    return s.map(_PAYMENT_RISK_MAP).fillna("Medium").astype(str)


def _customer_value(df: pd.DataFrame) -> pd.Series:
    return (df["tenure"] * df["MonthlyCharges"]).round(2)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Tenure_Group"]               = _tenure_group(out["tenure"])
    out["Monthly_Charge_Group"]       = _monthly_charge_group(out["MonthlyCharges"])
    out["Customer_Segment"]           = _customer_segment(out)
    out["Contract_Level"]             = _contract_level(out["Contract"])
    out["Unprotected_Services_Count"] = _unprotected_services_count(out)
    out["has_internet"]               = _has_internet(out)
    out["Payment_Risk"]               = _payment_risk(out["PaymentMethod"])
    out["Customer_Value"]             = _customer_value(out)
    return out


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from utils.preprocessing import load_clean_data

    df_clean, _ = load_clean_data()
    df = add_features(df_clean)

    ENGINEERED = [
        "Tenure_Group",
        "Monthly_Charge_Group",
        "Customer_Segment",
        "Contract_Level",
        "Unprotected_Services_Count",
        "has_internet",
        "Payment_Risk",
        "Customer_Value",
    ]

    SEP = "=" * 70
    print(SEP)
    print("STEP 3 — FEATURE ENGINEERING REPORT")
    print(SEP)
    print(f"\nDataFrame shape after feature engineering: {df.shape}")
    print(f"New columns added: {ENGINEERED}")

    # Churn leakage audit
    print(f"\n{'─'*70}")
    print("  CHURN LEAKAGE AUDIT")
    print(f"{'─'*70}")
    CHURN_INPUTS = ["Churn", "Churn_Binary"]
    leakage_found = False
    fns = [
        ("_tenure_group",              _tenure_group),
        ("_monthly_charge_group",      _monthly_charge_group),
        ("_customer_segment",          _customer_segment),
        ("_contract_level",            _contract_level),
        ("_unprotected_services_count", _unprotected_services_count),
        ("_has_internet",              _has_internet),
        ("_payment_risk",              _payment_risk),
        ("_customer_value",            _customer_value),
    ]
    for fname, fn in fns:
        src = inspect.getsource(fn)
        for bad in CHURN_INPUTS:
            if f'"{bad}"' in src or f"'{bad}'" in src:
                print(f"  FAIL: {fname} references {bad}")
                leakage_found = True
    if not leakage_found:
        print("  PASS: No feature function references Churn or Churn_Binary.")

    # Customer_Segment verification
    print(f"\n{'─'*70}")
    print("  CUSTOMER SEGMENT VERIFICATION (4 MUTUALLY EXCLUSIVE GROUPS)")
    print(f"{'─'*70}")
    seg_order = ["Family", "Solo", "Senior Family", "Senior Solo"]
    seg_tbl = (
        df.groupby("Customer_Segment", observed=True)["Churn_Binary"]
        .agg(Count="count", Churned="sum")
        .assign(Churn_Rate=lambda x: (x["Churned"] / x["Count"] * 100).round(2))
        .reindex(seg_order)
    )
    print(seg_tbl.to_string())

    # Feature-by-feature report
    for feat in ENGINEERED:
        print(f"\n{'─'*70}")
        print(f"  FEATURE: {feat}")
        print(f"{'─'*70}")

        if feat == "Customer_Value":
            print(f"    min    = {df[feat].min():.2f}")
            print(f"    mean   = {df[feat].mean():.2f}")
            print(f"    median = {df[feat].median():.2f}")
            print(f"    max    = {df[feat].max():.2f}")
            print(f"    std    = {df[feat].std():.2f}")
            df["_cv_q"] = pd.qcut(
                df[feat],
                q=4,
                labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"],
                duplicates="drop",
            ).astype(str)
            tbl = (
                df.groupby("_cv_q", observed=True)["Churn_Binary"]
                .agg(Count="count", Churned="sum")
                .assign(Churn_Rate=lambda x: (x["Churned"] / x["Count"] * 100).round(2))
            )
            print("\n  Churn rate by quartile of Customer_Value:")
            print(tbl.to_string())
            df.drop(columns=["_cv_q"], inplace=True)

        elif feat == "Contract_Level":
            label_map = {0: "0 (Month-to-month)", 1: "1 (One year)", 2: "2 (Two year)"}
            tbl = (
                df.groupby(feat, observed=True)["Churn_Binary"]
                .agg(Count="count", Churned="sum")
                .assign(Churn_Rate=lambda x: (x["Churned"] / x["Count"] * 100).round(2))
            )
            tbl.index = tbl.index.map(lambda x: label_map.get(int(x), str(x)))
            print(tbl.to_string())

        elif feat == "has_internet":
            label_map = {0: "No Internet", 1: "Has Internet"}
            tmp = df.assign(_label=df[feat].map(label_map))
            tbl = (
                tmp.groupby("_label", observed=True)["Churn_Binary"]
                .agg(Count="count", Churned="sum")
                .assign(Churn_Rate=lambda x: (x["Churned"] / x["Count"] * 100).round(2))
                .sort_values("Count", ascending=False)
            )
            print(tbl.to_string())

        elif feat == "Unprotected_Services_Count":
            inet_df = df[df["has_internet"] == 1]
            no_inet_count = int(df["Unprotected_Services_Count"].isna().sum())
            tbl = (
                inet_df.groupby(feat, observed=True)["Churn_Binary"]
                .agg(Count="count", Churned="sum")
                .assign(Churn_Rate=lambda x: (x["Churned"] / x["Count"] * 100).round(2))
            )
            tbl.index = tbl.index.astype(int)
            print("  Internet customers (0 to 4 unprotected services):")
            print(tbl.to_string())
            print(f"\n  No-internet customers excluded (set to missing): {no_inet_count}")

        else:
            tbl = (
                df.groupby(feat, observed=True)["Churn_Binary"]
                .agg(Count="count", Churned="sum")
                .assign(Churn_Rate=lambda x: (x["Churned"] / x["Count"] * 100).round(2))
                .sort_values("Count", ascending=False)
            )
            print(tbl.to_string())

    print(f"\n{SEP}")
    print(f"END OF STEP 3 — 8 features total. Shape: {df.shape}")
    print(SEP)