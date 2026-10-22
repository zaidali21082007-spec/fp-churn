"""
utils/preprocessing.py
=======================
Step 2 — Data Cleaning

Public API
----------
    load_clean_data(path: str | Path) -> tuple[pd.DataFrame, dict]

Returns
-------
    df_clean   : cleaned DataFrame ready for feature engineering (Step 3).
    report     : dict with cleaning statistics for display / logging.

Cleaning steps (in order)
--------------------------
1.  Load CSV from the explicit Telco file path.
2.  Convert TotalCharges to numeric:
        - Strip whitespace first.
        - Blank / whitespace-only strings  ->  0.0  (11 new customers, tenure=0).
        - Any other coerce failures are reported and also set to 0.0.
3.  Standardise text categories:
        - Strip leading/trailing whitespace from every object column.
        - Normalise known binary Yes/No columns to title-case.
4.  Check numerical ranges (tenure, MonthlyCharges, TotalCharges) and add
        notes to the report — nothing is deleted silently.
5.  Encode Churn: Yes -> 1, No -> 0  (new column Churn_Binary; raw Churn kept).
6.  Drop customerID from the returned DataFrame so it cannot leak into models.
        The column is preserved in a separate customer_id Series returned inside
        the report dict for use in dashboards.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ── Constants ────────────────────────────────────────────────────────────────

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "WA_Fn-UseC_-Telco-Customer-Churn.csv",
)

# Binary yes/no columns — normalised to title-case ("Yes" / "No")
_BINARY_COLS = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "Churn",
]

# Columns with three-value patterns ("Yes" / "No" / "No phone service" etc.)
_TRISTATE_COLS = [
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Columns left as-is after strip (categorical but not yes/no)
_FREEFORM_COLS = [
    "gender",
    "InternetService",
    "Contract",
    "PaymentMethod",
]


# ── Private helpers ───────────────────────────────────────────────────────────

def _normalise_text(series: pd.Series, title: bool = True) -> pd.Series:
    """Strip whitespace; optionally title-case."""
    s = series.astype(str).str.strip()
    if title:
        s = s.str.title()
    return s


def _convert_total_charges(
    series: pd.Series,
) -> tuple[pd.Series, pd.DataFrame, int, int]:
    """
    Convert TotalCharges (stored as str) to float.

    Returns
    -------
    converted   : float Series
    blank_rows  : index positions of blank/whitespace rows
    n_blank     : count of blank/whitespace rows (set to 0.0)
    n_other_err : count of non-blank coerce failures (also set to 0.0)
    """
    stripped = series.astype(str).str.strip()

    blank_mask = stripped == ""
    n_blank = int(blank_mask.sum())

    # Replace blank with "0" before numeric coercion
    patched = stripped.copy()
    patched[blank_mask] = "0"

    converted = pd.to_numeric(patched, errors="coerce")
    other_err_mask = converted.isna() & ~blank_mask
    n_other_err = int(other_err_mask.sum())
    converted[other_err_mask] = 0.0  # safe default

    blank_index = series.index[blank_mask]
    return converted, blank_index, n_blank, n_other_err


def _range_check(df: pd.DataFrame) -> list[str]:
    """
    Return a list of human-readable notes about numerical ranges.
    Nothing is deleted — notes are informational only.
    """
    notes: list[str] = []

    # tenure
    t_min, t_max = int(df["tenure"].min()), int(df["tenure"].max())
    notes.append(f"tenure range: [{t_min}, {t_max}] months — expected [0, 72].")
    if t_min < 0:
        n_neg = int((df["tenure"] < 0).sum())
        notes.append(f"  WARNING: {n_neg} rows have negative tenure — inspect manually.")
    if t_max > 72:
        n_high = int((df["tenure"] > 72).sum())
        notes.append(f"  WARNING: {n_high} rows have tenure > 72 — inspect manually.")

    # MonthlyCharges
    mc_min, mc_max = df["MonthlyCharges"].min(), df["MonthlyCharges"].max()
    notes.append(
        f"MonthlyCharges range: [${mc_min:.2f}, ${mc_max:.2f}] — expected [$18, $119]."
    )
    if mc_min < 0:
        notes.append("  WARNING: Negative MonthlyCharges detected.")

    # TotalCharges (after conversion)
    tc_min, tc_max = df["TotalCharges"].min(), df["TotalCharges"].max()
    notes.append(
        f"TotalCharges range (post-conversion): [${tc_min:.2f}, ${tc_max:.2f}]."
    )
    zeros = int((df["TotalCharges"] == 0).sum())
    if zeros:
        notes.append(
            f"  INFO: {zeros} rows have TotalCharges = 0.0 (new customers, tenure = 0)."
        )

    return notes


# ── Public API ────────────────────────────────────────────────────────────────

def load_clean_data(
    path: str | Path = DATA_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load and clean the Telco customer churn CSV.

    Parameters
    ----------
    path : str or Path
        Absolute or relative path to the CSV file.
        Defaults to the project's known Telco CSV location.

    Returns
    -------
    df_clean : pd.DataFrame
        Cleaned DataFrame.  customerID is dropped; Churn_Binary (0/1) added.
    report   : dict
        Keys:
          "n_rows_raw"          - row count before any drops
          "n_cols_raw"          - column count before any drops
          "blank_tc_index"      - index positions of the 11 blank-TotalCharges rows
          "n_blank_tc"          - count of blank TotalCharges rows (set to 0.0)
          "n_other_tc_errors"   - count of non-blank coerce failures (set to 0.0)
          "range_notes"         - list of range-check strings
          "churn_dist"          - Series: value_counts of Churn (raw)
          "churn_rate"          - float, fraction churned
          "customer_ids"        - Series of customerID (dropped from df)
          "category_uniques"    - dict col->unique values after normalisation
    """
    path = Path(path)
    df = pd.read_csv(path)

    report: dict[str, Any] = {
        "n_rows_raw": df.shape[0],
        "n_cols_raw": df.shape[1],
    }

    # ── Step 1: Preserve customerID then drop ────────────────────────────────
    report["customer_ids"] = df["customerID"].reset_index(drop=True).copy()
    df = df.drop(columns=["customerID"])

    # ── Step 2: Convert TotalCharges to numeric ──────────────────────────────
    tc_converted, blank_index, n_blank, n_other = _convert_total_charges(
        df["TotalCharges"]
    )
    df["TotalCharges"] = tc_converted.astype(float)

    report["blank_tc_index"] = blank_index
    report["n_blank_tc"] = n_blank
    report["n_other_tc_errors"] = n_other

    # ── Step 3: Standardise text categories ─────────────────────────────────
    for col in _BINARY_COLS:
        if col in df.columns:
            df[col] = _normalise_text(df[col], title=True)

    for col in _TRISTATE_COLS:
        if col in df.columns:
            df[col] = _normalise_text(df[col], title=True)

    for col in _FREEFORM_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # SeniorCitizen is already int (0/1) — no text work needed.

    # ── Step 4: Numerical range checks ──────────────────────────────────────
    report["range_notes"] = _range_check(df)

    # ── Step 5: Encode Churn ─────────────────────────────────────────────────
    report["churn_dist"] = df["Churn"].value_counts()
    report["churn_rate"] = (df["Churn"] == "Yes").mean()

    df["Churn_Binary"] = (df["Churn"] == "Yes").astype(int)
    # Keep raw "Churn" string column for display/filtering in the dashboard.

    # ── Step 6: Collect unique values per category column ───────────────────
    cat_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
    report["category_uniques"] = {
        col: sorted(df[col].dropna().unique().tolist()) for col in cat_cols
    }

    return df, report


# ── CLI convenience ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    df_clean, rpt = load_clean_data()

    SEP = "=" * 70

    print(SEP)
    print("STEP 2 — CLEANING REPORT")
    print(SEP)

    print(f"\nRaw shape  : {rpt['n_rows_raw']} rows x {rpt['n_cols_raw']} cols")
    print(f"Clean shape: {df_clean.shape[0]} rows x {df_clean.shape[1]} cols")

    print("\n-- TotalCharges conversion " + "-" * 43)
    print(f"  Blank/whitespace rows set to 0.0 : {rpt['n_blank_tc']}")
    print(f"  Other coerce failures set to 0.0 : {rpt['n_other_tc_errors']}")

    if rpt["n_blank_tc"] > 0:
        print("\n  Context for blank rows (re-read from raw CSV for display):")
        raw_ctx = pd.read_csv(DATA_PATH)[
            ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]
        ]
        print(raw_ctx.loc[rpt["blank_tc_index"]].to_string(index=True))
        print(
            "\n  Decision: set TotalCharges = 0.0 for these rows.\n"
            "  Rationale: tenure = 0 means no billing cycle completed yet.\n"
            "  All 11 are non-churners; imputing 0 is logically correct.\n"
            "  No rows dropped."
        )

    print("\n-- Numerical range checks " + "-" * 44)
    for note in rpt["range_notes"]:
        print(f"  {note}")

    print("\n-- Churn encoding " + "-" * 51)
    print(f"  Raw Churn distribution:\n{rpt['churn_dist'].to_string()}")
    print(f"\n  Churn_Binary added  : Yes -> 1, No -> 0")
    print(f"  Overall churn rate  : {rpt['churn_rate']*100:.2f}%")
    cb = df_clean["Churn_Binary"].value_counts().rename({0: "0 (No)", 1: "1 (Yes)"})
    print(f"\n  Churn_Binary counts:\n{cb.to_string()}")

    print("\n-- Category unique values (after normalisation) " + "-" * 22)
    for col, vals in rpt["category_uniques"].items():
        print(f"  {col:20s}: {vals}")

    print("\n-- Final dtypes " + "-" * 54)
    print(df_clean.dtypes.to_string())

    print("\n-- First 3 rows of cleaned DataFrame " + "-" * 32)
    print(df_clean.head(3).to_string())

    print("\n" + SEP)
    print("END OF STEP 2")
    print("  - No rows dropped.")
    print(f"  - 11 blank TotalCharges converted to 0.0.")
    print("  - customerID dropped from model DataFrame (preserved in report dict).")
    print("  - Churn_Binary column added (Yes=1, No=0).")
    print(SEP)
