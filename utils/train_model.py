# utils/train_model.py
"""
utils/train_model.py
====================
Step 5 — Model Training, Evaluation, and Selection
"""

from __future__ import annotations

import os
import sys
import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.preprocessing import load_clean_data
from utils.features import add_features

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PIPELINE_PATH = os.path.join(MODELS_DIR, "final_pipeline.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.csv")


def train_and_evaluate() -> tuple[Pipeline, pd.DataFrame]:
    df_clean, _ = load_clean_data()
    df = add_features(df_clean)

    # Cast nullable integers (e.g., Contract_Level) to float for sklearn imputer compatibility
    for col in df.select_dtypes(include=["Int64"]).columns:
        df[col] = df[col].astype(float)

    y = df["Churn_Binary"].astype(int)

    drop_cols = ["customerID", "Churn", "Churn_Binary"]
    if "TotalCharges" in df.columns and not pd.api.types.is_numeric_dtype(df["TotalCharges"]):
        drop_cols.append("TotalCharges")

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                num_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                ]),
                cat_cols,
            ),
        ]
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=15),
        "DecisionTreeClassifier": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=20, random_state=42
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=10, random_state=42, n_jobs=-1
        ),
    }

    SEP = "=" * 70
    print(SEP)
    print("STEP 5 — MODEL TRAINING & EVALUATION")
    print(SEP)
    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")
    print(f"Numeric features ({len(num_cols)}): {num_cols}")
    print(f"Categorical features ({len(cat_cols)}): {cat_cols}")

    trained_pipelines: dict[str, Pipeline] = {}
    test_predictions: dict[str, pd.Series] = {}
    records: list[dict] = []

    for name, clf in models.items():
        pipe = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", clf),
        ])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        trained_pipelines[name] = pipe
        test_predictions[name] = y_pred
        records.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "ROC_AUC": round(roc_auc, 4),
            "TN": int(cm[0, 0]),
            "FP": int(cm[0, 1]),
            "FN": int(cm[1, 0]),
            "TP": int(cm[1, 1]),
        })

        print(f"\n{'─' * 70}")
        print(f"MODEL: {name}")
        print(f"{'─' * 70}")
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1-Score  : {f1:.4f}")
        print(f"  ROC-AUC   : {roc_auc:.4f}")
        print("  Confusion Matrix (rows=actual [0, 1], cols=predicted [0, 1]):")
        print(f"    [[TN={cm[0, 0]:4d}, FP={cm[0, 1]:4d}]")
        print(f"     [FN={cm[1, 0]:4d}, TP={cm[1, 1]:4d}]]")

    # Sort by ROC_AUC descending, then Recall on churners descending as tiebreaker
    metrics_df = pd.DataFrame(records).sort_values(
        by=["ROC_AUC", "Recall"], ascending=[False, False]
    ).reset_index(drop=True)

    best_row = metrics_df.iloc[0]
    runner_up = metrics_df.iloc[1]
    best_name = str(best_row["Model"])
    best_pipe = trained_pipelines[best_name]

    print(f"\n{SEP}")
    print("MODEL COMPARISON TABLE (Sorted by ROC-AUC, then Recall)")
    print(SEP)
    print(metrics_df.to_string(index=False))

    print(f"\n{SEP}")
    print(f"SELECTED MODEL: {best_name}")
    print(SEP)
    print(
        f"Reasoning:\n"
        f"  - Accuracy alone is misleading due to class imbalance (~26.5% churners).\n"
        f"  - {best_name} achieved the highest ROC-AUC ({best_row['ROC_AUC']:.4f} vs "
        f"{runner_up['Model']} at {runner_up['ROC_AUC']:.4f}) and recall on churners "
        f"({best_row['Recall']:.4f} vs {runner_up['Recall']:.4f}).\n"
        f"  - Higher ROC-AUC and churner recall ensure better risk ranking and fewer "
        f"missed at-risk customers."
    )

    print(f"\nClassification Report — {best_name}:")
    print(
        classification_report(
            y_test,
            test_predictions[best_name],
            target_names=["Retained (0)", "Churned (1)"],
            digits=4,
        )
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_pipe, PIPELINE_PATH)
    metrics_df.to_csv(METRICS_PATH, index=False)

    print(f"Saved pipeline : {PIPELINE_PATH}")
    print(f"Saved metrics  : {METRICS_PATH}")
    print(SEP)

    return best_pipe, metrics_df


if __name__ == "__main__":
    train_and_evaluate()