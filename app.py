# app.py
# ============================================================
#  Customer Churn Prediction & Retention Dashboard
#  app.py — Main entry point
# ============================================================

import importlib.util
import os

import pandas as _pd
# pyrefly: ignore[missing-import]
import streamlit as st

# ── Page config (MUST be the very first Streamlit call) ────
st.set_page_config(
    page_title="Customer Churn Prediction & Retention Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject custom CSS for a polished look ──────────────────
st.markdown(
    """
    <style>
        /* ── Global font & background ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0a0a0a 0%, #1a0a0c 100%);
        }
        [data-testid="stSidebar"] * {
            color: #e2e8f0 !important;
        }
        [data-testid="stSidebar"] .sidebar-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #d4af37 !important;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }

        /* ── Radio nav buttons ── */
        div[role="radiogroup"] label {
            padding: 0.45rem 0.9rem;
            border-radius: 8px;
            margin-bottom: 4px;
            transition: background 0.2s;
            font-weight: 500;
        }
        div[role="radiogroup"] label:hover {
            background: rgba(212, 175, 55, 0.15);
        }

        /* ── Main area ── */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        /* ── Hero card ── */
        .hero-card {
            background: linear-gradient(135deg, #550000 0%, #800000 100%);
            border-radius: 16px;
            padding: 2.5rem 3rem;
            margin-bottom: 2rem;
            color: white;
            box-shadow: 0 20px 60px rgba(128, 0, 0, 0.3);
        }
        .hero-card h1 {
            font-size: 2.2rem;
            font-weight: 700;
            margin: 0 0 0.4rem 0;
        }
        .hero-card p {
            font-size: 1.05rem;
            opacity: 0.9;
            margin: 0;
        }

        /* ── Info cards ── */
        .info-card {
            background: #1a0a0c;
            border-left: 4px solid #d4af37;
            border-radius: 10px;
            padding: 1.2rem 1.5rem;
            margin-bottom: 1rem;
            color: #e2e8f0;
        }
        .info-card h3 {
            color: #d4af37;
            margin: 0 0 0.5rem 0;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 600;
        }
        .info-card p, .info-card ul {
            margin: 0;
            font-size: 0.97rem;
            line-height: 1.7;
            color: #cbd5e1;
        }
        .info-card ul {
            padding-left: 1.2rem;
        }

        /* ── Badge ── */
        .badge {
            display: inline-block;
            background: rgba(212, 175, 55,0.15);
            color: #d4af37;
            border: 1px solid rgba(212, 175, 55,0.4);
            border-radius: 999px;
            padding: 0.2rem 0.8rem;
            font-size: 0.78rem;
            font-weight: 600;
            margin-right: 0.4rem;
            margin-top: 0.3rem;
        }

        /* ── Divider ── */
        hr {
            border: none;
            border-top: 1px solid #334155;
            margin: 1.5rem 0;
        }

        /* ── Metric cards ── */
        [data-testid="metric-container"] {
            background: #1a0a0c;
            border-radius: 12px;
            padding: 1rem;
            border: 1px solid #334155;
        }

        /* ── Scrollbar ── */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #0a0a0a; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
    /* ── Headings ── */
h1 { letter-spacing: -0.02em; }
h2, h3 { letter-spacing: -0.01em; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    border-top: 3px solid #d4af37;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
    transition: transform 0.15s ease;
}
[data-testid="metric-container"]:hover { transform: translateY(-2px); }
[data-testid="stMetricValue"] { font-weight: 600; }

/* ── Selected sidebar item ── */
div[role="radiogroup"] label:has(input:checked) {
    background: rgba(212, 175, 55, 0.18);
    border-left: 3px solid #d4af37;
}

/* ── Tables ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Buttons ── */
.stButton > button,
.stDownloadButton > button,
[data-testid="stFormSubmitButton"] > button {
    border-radius: 8px;
    border: 1px solid #d4af37;
    font-weight: 600;
}
.stButton > button:hover,
.stDownloadButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {
    background: #d4af37;
    color: #0a0a0a;
}

/* ── Alerts ── */
[data-testid="stAlert"] { border-radius: 10px; }

    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ────────────────────────────────────────────────
PAGES_DIR = os.path.join(os.path.dirname(__file__), "views")

# Collect views from the views/ directory (*.py files only)
page_files = sorted(
    [f for f in os.listdir(PAGES_DIR) if f.endswith(".py") and not f.startswith("__")]
) if os.path.isdir(PAGES_DIR) else []

# Build human-readable labels from filenames
# Expected naming convention: 1_Home.py, 2_EDA.py, 3_Model.py …
def file_to_label(filename: str) -> str:
    stem = os.path.splitext(filename)[0]          # remove .py
    parts = stem.split("_", 1)                    # split on first underscore
    label = parts[1] if len(parts) > 1 else stem  # drop leading number
    return label.replace("_", " ").strip()

nav_options = {"🏠  Home": None}  # Home is always app.py itself
for pf in page_files:
    label = file_to_label(pf)
    icon_map = {
        "EDA": "🔍",
        "Model": "🤖",
        "Prediction": "🎯",
        "Report": "📄",
        "About": "ℹ️",
    }
    icon = next((v for k, v in icon_map.items() if k.lower() in label.lower()), "📌")
    nav_options[f"{icon}  {label}"] = pf

with st.sidebar:
    st.markdown('<p class="sidebar-title">📊 Dashboard</p>', unsafe_allow_html=True)
    st.markdown("**Customer Churn Prediction & Retention Dashboard**")
    st.markdown("---")
    selected_label = st.radio(
        "Navigation",
        options=list(nav_options.keys()),
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("© 2026 · Customer Churn Prediction & Retention Dashboard")

# ── Router ─────────────────────────────────────────────────
selected_file = nav_options[selected_label]

if selected_file is None:
    # ── PAGE 1: HOME ───────────────────────────────────────
    _csv_path = os.path.join(
        os.path.dirname(__file__), "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )
    if not os.path.isfile(_csv_path):
        st.error(f"Dataset file not found at expected path: `{_csv_path}`")
        st.stop()

    _df = _pd.read_csv(_csv_path)
    _n_rows, _n_cols = _df.shape
    _churn_rate = (_df["Churn"].astype(str).str.strip().str.title() == "Yes").mean() * 100
    _col_badges = "".join(f'<span class="badge">{c}</span>' for c in _df.columns)
    _dataset_name = os.path.basename(_csv_path)

    # Hero banner
    st.markdown(
        """
        <div class="hero-card">
            <h1>📊 Customer Churn Prediction<br>&amp; Retention Dashboard</h1>
            <p>An end-to-end machine-learning powered platform for customer churn prediction and retention analytics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        # Business Problem
        st.markdown(
            """
            <div class="info-card">
                <h3>🎯 Business Problem</h3>
                <p>
                    A subscription-based telecommunications company is losing customers to
                    competitors. Without a systematic analytics and risk-scoring workflow,
                    retention teams cannot proactively identify at-risk subscribers before they
                    cancel their service.
                </p>
                <p style="margin-top:0.6rem;">
                    The goal of this system is to identify <strong>who is likely to leave</strong>,
                    uncover <strong>what factors are associated with churn</strong>, and quantify
                    <strong>how much monthly recurring revenue is at risk</strong> so targeted
                    retention actions can be prioritized.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Project Objectives
        st.markdown(
            """
            <div class="info-card">
                <h3>🚀 Project Objectives</h3>
                <ul>
                    <li>Identify key demographic, service, and billing <strong>churn drivers</strong>.</li>
                    <li>Train and compare supervised <strong>classification models</strong> (Logistic Regression, KNN, Decision Tree, Random Forest).</li>
                    <li>Provide interactive <strong>threshold analysis</strong> to balance precision and recall trade-offs.</li>
                    <li>Score <strong>individual customers for churn risk</strong> using the selected pipeline.</li>
                    <li>Support data-driven <strong>retention action planning</strong> for high-risk customer segments.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Dataset Description
        st.markdown(
            f"""
            <div class="info-card">
                <h3>🗄️ Dataset Description</h3>
                <p>
                    The project uses the <strong>IBM/Telco Customer Churn Dataset</strong>
                    (<code>{_dataset_name}</code>). It contains
                    <strong>{_n_rows:,} customers</strong> and
                    <strong>{_n_cols} columns</strong> capturing customer demographics,
                    account tenure, contract terms, payment methods, subscribed services,
                    and billing charges.
                </p>
                <p style="margin-top:0.6rem;"><strong>Dataset columns:</strong></p>
                <p>{_col_badges}</p>
                <p style="margin-top:0.7rem;">
                    <strong>Target variable:</strong> <code>Churn</code>
                    (Yes / No, encoded as binary <code>Churn_Binary</code> 1 / 0).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        # Student / Course info
        st.markdown(
            """
            <div class="info-card">
                <h3>🎓 Student Information</h3>
                <p><strong>Name 1:</strong> Zaid Ali</p>
                <p><strong>Course:</strong> Data Science and Machine Learning</p>
                <p><strong>Project:</strong> Customer Churn Prediction (Project 2)</p>
                <p><strong>Academic Year:</strong> 2025 / 2026</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quick stats computed directly from the CSV
        st.markdown("#### 📈 Dataset at a Glance")
        m1, m2 = st.columns(2)
        m1.metric("Total Records", f"{_n_rows:,}")
        m2.metric("Columns", _n_cols)
        m3, m4 = st.columns(2)
        m3.metric("Churn Rate", f"{_churn_rate:.2f}%")
        m4.metric("Target", "Churn")

        st.markdown("<br>", unsafe_allow_html=True)

        # Tech stack badges
        st.markdown(
            """
            <div class="info-card">
                <h3>🛠️ Tech Stack</h3>
                <span class="badge">Python 3.13</span>
                <span class="badge">Streamlit</span>
                <span class="badge">Pandas</span>
                <span class="badge">NumPy</span>
                <span class="badge">Scikit-learn</span>
                <span class="badge">Seaborn</span>
                <span class="badge">Matplotlib</span>
                <span class="badge">Joblib</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Navigation hint
        st.info(
            "👈 Use the **sidebar** to navigate between pages: "
            "Customer Explorer, Churn Dashboard, Churn Analysis, "
            "Model Comparison, Threshold Analysis, Churn Prediction, "
            "and Retention Dashboard.",
            icon="💡",
        )

else:
    # ── Dynamic page loader ────────────────────────────────
    page_path = os.path.join(PAGES_DIR, selected_file)
    try:
        spec = importlib.util.spec_from_file_location("page_module", page_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create a module spec for {selected_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:
        st.error(f"⚠️  Could not load page **{selected_file}**: `{exc}`")
# ── FINAL MAROON / GOLD OVERRIDE (loads last, so it wins over earlier blue rules) ──
st.markdown(
    """
    <style>
    html body [data-testid="stSidebar"] {
        background-image: linear-gradient(180deg, #2a0006 0%, #0a0a0a 100%) !important;
        background-color: #0a0a0a !important;
        border-right: 1px solid #d4af37 !important;
    }
    html body .sidebar-title { color: #d4af37 !important; }
    html body .hero-card {
        background: linear-gradient(135deg, #3a0000 0%, #550000 55%, #800000 100%) !important;
        border: 1px solid #d4af37 !important;
        box-shadow: 0 20px 60px rgba(85, 0, 0, 0.45) !important;
    }
    html body .hero-card h1,
    html body .hero-card p { color: #f5e6b3 !important; }
    html body .info-card {
        background: #1a0a0c !important;
        border-left: 4px solid #d4af37 !important;
    }
    html body .info-card h3 { color: #d4af37 !important; }
    html body .info-card p,
    html body .info-card li { color: #e8e0d0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── FINAL OVERRIDE v3: maroon / gold everywhere (loads last) ──
st.markdown(
    """
    <style>
    html body .badge {
        background: rgba(212, 175, 55, 0.12) !important;
        color: #f0d77a !important;
        border: 1px solid rgba(212, 175, 55, 0.5) !important;
    }
    html body .info-card h3 { color: #d4af37 !important; }
    html body [data-testid="stAlert"],
    html body [data-baseweb="notification"] {
        background: #1a0a0c !important;
        border: 1px solid rgba(212, 175, 55, 0.35) !important;
        border-left: 4px solid #d4af37 !important;
        border-radius: 8px !important;
    }
    html body [data-testid="stAlert"] *,
    html body [data-baseweb="notification"] * {
        color: #f5e6b3 !important;
    }
    html body [data-testid="stDownloadButton"] button,
    html body .stDownloadButton > button,
    html body .stButton > button,
    html body [data-testid="stFormSubmitButton"] > button {
        background: transparent !important;
        color: #f0d77a !important;
        border: 1px solid #d4af37 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }
    html body [data-testid="stDownloadButton"] button:hover,
    html body .stDownloadButton > button:hover,
    html body .stButton > button:hover,
    html body [data-testid="stFormSubmitButton"] > button:hover {
        background: #550000 !important;
        color: #f5e6b3 !important;
        border-color: #f0d77a !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
