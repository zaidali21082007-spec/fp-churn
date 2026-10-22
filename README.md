# Customer Churn Prediction & Retention Dashboard

An end-to-end machine learning project that predicts which telecom customers are likely to leave, explains the factors associated with churn, quantifies the monthly revenue at risk, and presents the results in an 8-page Streamlit dashboard.

**Live app:** https://fp-churn-tsqxrn6e86avqpcszde6rc.streamlit.app

**Notebook:** [`notebooks/project_analysis.ipynb`](notebooks/project_analysis.ipynb) documents the full workflow with code, outputs, and interpretation.

**Author:** Zaid Ali · Data Science and Machine Learning · Project 2 · 2025/2026

---

## Business problem

A subscription telecom company is losing customers to competitors. Without a systematic risk-scoring workflow, retention teams cannot identify at-risk subscribers before they cancel. This project aims to:

- identify **who is likely to leave**,
- uncover **which factors are associated with churn**, and
- quantify **how much monthly recurring revenue is at risk**, so retention actions can be prioritized.

## Dataset

IBM Telco Customer Churn dataset (`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`): 7,043 customers and 21 columns covering demographics, tenure, contract type, payment method, subscribed services, and billing. The target is `Churn` (Yes/No), encoded as `Churn_Binary` (1/0). About 26.5% of customers churned.

## Screenshots

### Home

![Home](screenshots/01_home.png)

### Customer Explorer

![Customer Explorer](screenshots/02_customer_explorer.png)

### Churn Dashboard

![Churn Dashboard](screenshots/03_churn_dashboard.png)

### Churn Analysis

![Churn Analysis](screenshots/04_churn_analysis1.png)

![Churn Analysis](screenshots/04_churn_analysis2.png)

![Churn Analysis](screenshots/04_churn_analysis3.png)

![Churn Analysis](screenshots/04_churn_analysis4.png)

![Churn Analysis](screenshots/04_churn_analysis5.png)

![Churn Analysis](screenshots/04_churn_analysis6.png)

### Model Comparison

![Model Comparison](screenshots/05_model_comparison1.png)

![Model Comparison](screenshots/05_model_comparison2.png)

![Model Comparison](screenshots/05_model_comparison3.png)

![Model Comparison](screenshots/05_model_comparison4.png)

![Model Comparison](screenshots/05_model_comparison5.png)

![Model Comparison](screenshots/05_model_comparison6.png)

### Threshold Analysis

![Threshold Analysis](screenshots/06_threshold_analysis.png)

### Churn Prediction

![Churn Prediction](screenshots/07_churn_prediction1.png)

![Churn Prediction](screenshots/07_churn_prediction2.png)

### Retention Dashboard

![Retention Dashboard](screenshots/08_retention_dashboard1.png)

![Retention Dashboard](screenshots/08_retention_dashboard2.png)

## Project structure

```
├── app.py                      # Streamlit entry point (Home page and page router)
├── views/                      # The 7 dashboard pages, loaded by app.py
│   ├── 1_Customer_Explorer.py
│   ├── 2_Churn_Dashboard.py
│   ├── 3_Churn_Analysis.py
│   ├── 4_Model_Comparison.py
│   ├── 5_Threshold_Analysis.py
│   ├── 6_Churn_Prediction.py
│   └── 7_Retention_Dashboard.py
├── utils/
│   ├── preprocessing.py        # Data loading and cleaning
│   ├── features.py             # Feature engineering (8 features)
│   ├── charts.py               # Reusable EDA chart functions
│   └── train_model.py          # Trains 4 models and saves the selected pipeline
├── models/
│   ├── final_pipeline.pkl      # Saved pipeline (Random Forest)
│   └── metrics.csv             # Model comparison results
├── notebooks/
│   └── project_analysis.ipynb  # Full analysis notebook
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── screenshots/                # Dashboard screenshots
├── requirements.txt
└── README.md
```

## Method

**Cleaning.** `TotalCharges` is stored as text, and 11 rows are blank. These customers have `tenure = 0` (no billing cycle yet), so they are set to 0. No rows are dropped. Text categories are whitespace-stripped and standardized. `customerID` is removed from the modeling table.

**Feature engineering.** Eight features are added: `Tenure_Group`, `Monthly_Charge_Group`, `Customer_Segment` (Family, Solo, Senior Family, Senior Solo), `Contract_Level`, `Unprotected_Services_Count`, `has_internet`, `Payment_Risk`, and `Customer_Value`. A leakage check confirms that no feature function references `Churn` or `Churn_Binary`.

**Modeling.** The data is split 80/20 with stratification on the target and `random_state=42`. Four classifiers are trained on the same preprocessing pipeline: Logistic Regression, KNN, Decision Tree, and Random Forest. The model is selected by ROC-AUC, with recall as the tie-breaker.

**Retention scoring.** Each customer's churn probability is multiplied by their monthly charge to estimate expected monthly revenue at risk. Customers are grouped into Low, Medium, High, and Critical tiers at the 25%, 50%, and 75% probability cutoffs.

## Results (held-out test set, n = 1,409)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest (selected)** | 0.804 | 0.665 | 0.527 | 0.588 | 0.844 |
| Logistic Regression | 0.797 | 0.646 | 0.521 | 0.577 | 0.843 |
| Decision Tree | 0.797 | 0.668 | 0.468 | 0.550 | 0.838 |
| KNN | 0.777 | 0.582 | 0.570 | 0.576 | 0.821 |

Accuracy is the least useful number here because churners are the minority class. The Random Forest catches about half of churners, and about two-thirds of the customers it flags do churn. Actual churn rises from 8% in the Low tier to 79% in the Critical tier, so the risk probabilities are usefully calibrated.

**Key findings:**
- Month-to-month contracts churn at 42.7%, compared with 11.3% on one-year and 2.8% on two-year contracts.
- Automatic payment is associated with lower churn overall (16.0% vs 34.7%). Most of that gap comes from contract type: it narrows to 0.1 points on two-year contracts.

## Rubric mapping and data limitations

The assignment asked for several fields that this Telco dataset does not contain. These omissions were a deliberate, documented choice, and no values were invented for them.

| Requested field | Status | How the project handles it |
|---|---|---|
| Satisfaction score / NPS | Not in dataset | Omitted from all pages and filters |
| Complaints, support tickets | Not in dataset | `TechSupport` is used as the nearest support indicator |
| Late payments | Not in dataset | `PaymentMethod` and `PaperlessBilling` are used as proxies |
| Referral count | Not in dataset | Omitted |
| Region | Not in dataset | Omitted from the Customer Explorer filters and the Retention Dashboard |
| Last interaction date | Not in dataset | Omitted |
| Subscription type | Not in dataset | `Contract` is the closest equivalent and is used throughout |
| Average satisfaction KPI | Not in dataset | Omitted from the Churn Dashboard |

**Association, not causation.** The model identifies patterns in historical data. A high risk score does not show that a service, price, or contract causes a customer to leave. Retention actions should be tested with a controlled experiment before their effect is assumed.

**Single snapshot.** The data is one point in time. The model cannot see how a customer's status changed over time.

**In-sample scoring on the Retention Dashboard.** The dashboard scores all 7,043 customers, including the 80% used for training, so its figures (1,477 High+Critical customers, $76,430 monthly revenue at risk) run higher than out-of-sample results. The page carries a caption saying so. The notebook's Section 6 reports the held-out figures (296 High+Critical customers on the test set, about $15,000 monthly revenue at risk among those 1,409 customers).

**KNN reproducibility.** The notebook reproduces the Random Forest results in `metrics.csv` exactly. The KNN row differs by one test-set prediction. The selected model is unaffected.

## How to run locally

```bash
# 1. Clone and install dependencies
git clone [YOUR REPO URL]
cd [REPO FOLDER]
pip install -r requirements.txt

# 2. Train the model (creates models/final_pipeline.pkl and models/metrics.csv)
python -m utils.train_model

# 3. Launch the dashboard
streamlit run app.py
```

Then open http://localhost:8501 in a browser.

To rerun the analysis notebook, open `notebooks/project_analysis.ipynb` from the `notebooks/` folder in Jupyter or VS Code.

## Deployment

The app is deployed on Streamlit Community Cloud from this repository, with `app.py` as the entry point. The trained model file (`models/final_pipeline.pkl`) must be present in the repository for the deployed app to run.

## Bonus features

- Custom dark theme and layout
- Advanced multi-filter exploration on the Customer Explorer and Churn Analysis pages
- Interactive Plotly charts
- Decision threshold slider with precision, recall, and F1 at each threshold
- Model explainability through feature importances, ROC and precision-recall curves
- Single-customer prediction with CSV download of the result
- Downloadable high-risk customer list for retention campaigns
- Retention recommendations in Finding, Evidence, Recommendation, Expected Result format

## Technologies

Python · Pandas · NumPy · scikit-learn · Streamlit · Plotly · Matplotlib · Seaborn · Joblib
