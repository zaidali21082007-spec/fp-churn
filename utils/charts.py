# utils/charts.py
"""
utils/charts.py
===============
Step 4a & 4b — Reusable chart functions (charts 1-10)

Chart inventory
---------------
    1. chart_churn_distribution       — overall Yes/No donut
    2. chart_churn_by_contract        — churn rate per contract type
    3. chart_churn_by_payment_method  — churn rate per payment method
    4. chart_churn_by_autopay         — auto-pay vs manual pay comparison
    5. chart_monthly_charges_by_churn — violin: MonthlyCharges split by Churn
    6. chart_tenure_by_churn          — violin: tenure split by Churn
    7. chart_support_cols_by_churn    — TechSupport & OnlineSecurity (internet only)
    8. chart_correlation_heatmap      — tenure, MonthlyCharges, TotalCharges, Churn_Binary
    9. chart_churn_by_segment         — churn rate per Customer_Segment
   10. chart_churn_by_tenure_group    — churn rate per Tenure_Group
"""

from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")   # non-interactive; safe in Streamlit and CLI

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Colour palette ─────────────────────────────────────────────────────────────
_BG     = "#0f172a"   # figure background  (darkest)
_PANEL  = "#1e293b"   # axes background
_GRID   = "#334155"   # grid / spine colour
_TEXT   = "#e2e8f0"   # primary text / titles
_MUTED  = "#94a3b8"   # secondary text / tick labels
_ACCENT = "#38bdf8"   # highlight / accent blue
_GREEN  = "#22c55e"   # Churn = No  (retained)
_RED    = "#ef4444"   # Churn = Yes (churned)
_AMBER  = "#f59e0b"   # intermediate risk

CHURN_PALETTE = {"No": _GREEN, "Yes": _RED}

# ── Shared rcParams context dict ───────────────────────────────────────────────
_RC: dict = {
    "figure.facecolor":    _BG,
    "axes.facecolor":      _PANEL,
    "axes.edgecolor":      _GRID,
    "axes.labelcolor":     _TEXT,
    "axes.titlecolor":     _TEXT,
    "axes.titlesize":      15,
    "axes.labelsize":      11,
    "xtick.color":         _MUTED,
    "ytick.color":         _MUTED,
    "xtick.labelsize":     10,
    "ytick.labelsize":     10,
    "text.color":          _TEXT,
    "grid.color":          _GRID,
    "grid.linewidth":      0.6,
    "legend.facecolor":    _PANEL,
    "legend.edgecolor":    _GRID,
    "legend.labelcolor":   _TEXT,
    "legend.fontsize":     10,
    "font.family":         "DejaVu Sans",
    "axes.spines.top":     False,
    "axes.spines.right":   False,
}


# ── Private helpers ────────────────────────────────────────────────────────────

def _style_ax(
    ax: plt.Axes,
    xlabel: str = "",
    ylabel: str = "",
    grid_axis: str = "y",
) -> None:
    """Apply dark theme to a single axes object."""
    ax.set_xlabel(xlabel, color=_TEXT, fontsize=11, labelpad=8)
    ax.set_ylabel(ylabel, color=_TEXT, fontsize=11, labelpad=8)
    ax.tick_params(colors=_MUTED, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
    if grid_axis:
        ax.grid(axis=grid_axis, color=_GRID, linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)


def _add_caption(fig: plt.Figure, caption: str) -> None:
    """Render a small italic caption at the bottom of the figure."""
    fig.text(
        0.5, 0.01,
        f"\u25b8  {caption}",
        ha="center", va="bottom",
        fontsize=9, color=_MUTED, style="italic",
    )


def _bar_color(rate: float) -> str:
    """Return colour based on churn rate severity."""
    if rate >= 35:
        return _RED
    if rate >= 20:
        return _AMBER
    return _GREEN


# ── Chart 1 ───────────────────────────────────────────────────────────────────

def chart_churn_distribution(
    df: pd.DataFrame,
    figsize: tuple[int, int] = (8, 6),
) -> tuple[plt.Figure, str]:
    """
    Donut chart: overall customer churn split (Active vs Churned).

    Business question: What percentage of customers churned in this period?
    """
    caption = "What percentage of customers churned in this period?"

    with plt.rc_context(_RC):
        counts = df["Churn"].value_counts()
        n_no   = int(counts.get("No",  0))
        n_yes  = int(counts.get("Yes", 0))
        total  = n_no + n_yes
        values = [n_no, n_yes]
        colors = [_GREEN, _RED]

        fig, ax = plt.subplots(figsize=figsize)
        fig.subplots_adjust(bottom=0.12)

        wedges, _, autotexts = ax.pie(
            values,
            labels=None,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            pctdistance=0.72,
            wedgeprops=dict(width=0.52, edgecolor=_BG, linewidth=2.5),
        )
        for at in autotexts:
            at.set(color=_TEXT, fontsize=13, fontweight="bold")

        # Centre annotation
        ax.text(
            0, 0.08,
            f"{total:,}",
            ha="center", va="center",
            fontsize=18, color=_TEXT, fontweight="bold",
        )
        ax.text(
            0, -0.18,
            "Customers",
            ha="center", va="center",
            fontsize=11, color=_MUTED,
        )

        legend_handles = [
            mpatches.Patch(
                color=_GREEN,
                label=f"Active \u2014 {n_no:,}  ({n_no/total*100:.1f}%)",
            ),
            mpatches.Patch(
                color=_RED,
                label=f"Churned \u2014 {n_yes:,}  ({n_yes/total*100:.1f}%)",
            ),
        ]
        ax.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=2,
            framealpha=0.15,
            fontsize=10,
        )
        ax.set_title("Customer Churn Distribution", fontsize=15, fontweight="bold",
                     color=_TEXT, pad=16)
        _add_caption(fig, caption)

    return fig, caption


# ── Chart 2 ───────────────────────────────────────────────────────────────────

def chart_churn_by_contract(
    df: pd.DataFrame,
    figsize: tuple[int, int] = (9, 5),
) -> tuple[plt.Figure, str]:
    """
    Vertical bar chart: churn rate per contract type.

    Business question: Does contract type predict churn risk?
    """
    caption = "Does contract type predict churn risk?"

    with plt.rc_context(_RC):
        grp = (
            df.groupby("Contract")["Churn_Binary"]
            .agg(Count="count", Churned="sum")
            .assign(Churn_Rate=lambda x: x["Churned"] / x["Count"] * 100)
            .sort_values("Churn_Rate", ascending=False)
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=figsize)
        fig.subplots_adjust(bottom=0.18)

        bar_colors = [_bar_color(r) for r in grp["Churn_Rate"]]
        bars = ax.bar(
            grp["Contract"], grp["Churn_Rate"],
            color=bar_colors, width=0.52,
            edgecolor=_BG, linewidth=1.5,
        )

        for bar, row in zip(bars, grp.itertuples()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.0,
                f"{row.Churn_Rate:.1f}%\n(n={row.Count:,})",
                ha="center", va="bottom",
                fontsize=10, color=_TEXT, fontweight="bold",
            )

        ax.set_title("Churn Rate by Contract Type", fontweight="bold", pad=14)
        _style_ax(ax, xlabel="Contract Type", ylabel="Churn Rate (%)")
        ax.set_ylim(0, grp["Churn_Rate"].max() * 1.28)
        _add_caption(fig, caption)

    return fig, caption


# ── Chart 3 ───────────────────────────────────────────────────────────────────

def chart_churn_by_payment_method(
    df: pd.DataFrame,
    figsize: tuple[int, int] = (10, 5),
) -> tuple[plt.Figure, str]:
    """
    Horizontal bar chart: churn rate per payment method (highest churn first).

    Business question: Which payment methods are associated with higher churn?
    """
    caption = "Which payment methods are associated with higher churn?"

    # Readable short labels
    _PM_LABELS = {
        "Electronic check":           "Electronic Check",
        "Mailed check":               "Mailed Check",
        "Bank transfer (automatic)":  "Bank Transfer\n(Auto-Pay)",
        "Credit card (automatic)":    "Credit Card\n(Auto-Pay)",
    }

    with plt.rc_context(_RC):
        grp = (
            df.groupby("PaymentMethod")["Churn_Binary"]
            .agg(Count="count", Churned="sum")
            .assign(Churn_Rate=lambda x: x["Churned"] / x["Count"] * 100)
            .sort_values("Churn_Rate", ascending=True)   # ascending for hbar
            .reset_index()
        )
        grp["Label"] = grp["PaymentMethod"].map(_PM_LABELS).fillna(grp["PaymentMethod"])

        fig, ax = plt.subplots(figsize=figsize)
        fig.subplots_adjust(left=0.20, bottom=0.18)

        bar_colors = [_bar_color(r) for r in grp["Churn_Rate"]]
        bars = ax.barh(
            grp["Label"], grp["Churn_Rate"],
            color=bar_colors, height=0.52,
            edgecolor=_BG, linewidth=1.5,
        )

        max_rate = grp["Churn_Rate"].max()
        for bar, row in zip(bars, grp.itertuples()):
            ax.text(
                bar.get_width() + max_rate * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{row.Churn_Rate:.1f}%  (n={row.Count:,})",
                ha="left", va="center",
                fontsize=10, color=_TEXT, fontweight="bold",
            )

        ax.set_title("Churn Rate by Payment Method", fontweight="bold", pad=14)
        _style_ax(ax, xlabel="Churn Rate (%)", ylabel="Payment Method", grid_axis="x")
        ax.set_xlim(0, max_rate * 1.38)
        _add_caption(fig, caption)

    return fig, caption


# ── Chart 4 ───────────────────────────────────────────────────────────────────

def chart_churn_by_autopay(
    df: pd.DataFrame,
    figsize: tuple[int, int] = (8, 5),
) -> tuple[plt.Figure, str]:
    """
    Grouped bar chart: churn rate for auto-pay enrolled vs manual payment.

    Business question: Is auto-pay linked to lower churn, and does that hold
    once contract type is considered?
    """
    caption = (
        "Auto-pay is linked to lower churn overall, but the gap narrows from "
        "13 points on month-to-month contracts to under 1 point on two-year "
        "contracts, so contract type accounts for much of it."
    )

    with plt.rc_context(_RC):
        tmp = df.copy()
        tmp["Auto_Pay"] = tmp["PaymentMethod"].str.contains(
            "automatic", case=False, na=False
        ).map({True: "Auto-Pay\nEnrolled", False: "Manual\nPayment"})

        grp = (
            tmp.groupby("Auto_Pay")["Churn_Binary"]
            .agg(Count="count", Churned="sum")
            .assign(Churn_Rate=lambda x: x["Churned"] / x["Count"] * 100)
            .reindex(["Auto-Pay\nEnrolled", "Manual\nPayment"])
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=figsize)
        fig.subplots_adjust(bottom=0.18)

        bar_colors = [_GREEN, _RED]
        bars = ax.bar(
            grp["Auto_Pay"], grp["Churn_Rate"],
            color=bar_colors, width=0.42,
            edgecolor=_BG, linewidth=1.5,
        )

        for bar, row in zip(bars, grp.itertuples()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.6,
                f"{row.Churn_Rate:.1f}%\n(n={row.Count:,})",
                ha="center", va="bottom",
                fontsize=11, color=_TEXT, fontweight="bold",
            )

        # Overall gap annotation (across all customers, before contract type)
        auto_rate   = grp.loc[grp["Auto_Pay"] == "Auto-Pay\nEnrolled", "Churn_Rate"].values[0]
        manual_rate = grp.loc[grp["Auto_Pay"] == "Manual\nPayment",    "Churn_Rate"].values[0]
        diff = manual_rate - auto_rate
        ax.annotate(
            f"+{diff:.1f} pts higher\nwithout auto-pay",
            xy=(1, manual_rate / 2),
            xytext=(0.55, manual_rate * 0.6),
            color=_MUTED, fontsize=9, style="italic",
            arrowprops=dict(arrowstyle="->", color=_MUTED, lw=0.8),
        )

        ax.set_title("Churn Rate: Auto-Pay vs Manual Payment", fontweight="bold", pad=14)
        _style_ax(ax, xlabel="Payment Enrollment Type", ylabel="Churn Rate (%)")
        ax.set_ylim(0, grp["Churn_Rate"].max() * 1.30)
        _add_caption(fig, caption)

    return fig, caption
# ── Chart 5 ───────────────────────────────────────────────────────────────────

def chart_monthly_charges_by_churn(
    df: pd.DataFrame,
    figsize: tuple[int, int] = (9, 5),
) -> tuple[plt.Figure, str]:
    """
    Violin plot: MonthlyCharges distribution split by churn status.

    Business question: Do churning customers pay higher monthly charges?
    """
    caption = "Do churning customers pay higher monthly charges than retained ones?"

    order        = ["No", "Yes"]
    tick_labels  = ["Active Customers\n(Retained)", "Churned Customers\n(Churned)"]
    pal          = [_GREEN, _RED]

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=figsize)
        fig.subplots_adjust(bottom=0.18)

        sns.violinplot(
            data=df,
            x="Churn", y="MonthlyCharges",
            order=order,
            palette=pal,
            inner="quartile",
            linewidth=1.2,
            ax=ax,
        )
        ax.set_xticklabels(tick_labels)

        # Median annotations
        for i, label in enumerate(order):
            med = df[df["Churn"] == label]["MonthlyCharges"].median()
            ax.text(
                i, med + 2.5,
                f"Median: ${med:.0f}",
                ha="center", va="bottom",
                fontsize=9, color=_TEXT, fontweight="bold",
            )

        ax.set_title("Monthly Charges Distribution by Churn Status",
                     fontweight="bold", pad=14)
        _style_ax(ax, xlabel="Churn Status", ylabel="Monthly Charges ($)")
        _add_caption(fig, caption)

    return fig, caption


# ── Chart 6 ───────────────────────────────────────────────────────────────────

def chart_tenure_by_churn(
    df: pd.DataFrame,
    figsize: tuple[int, int] = (9, 5),
) -> tuple[plt.Figure, str]:
    """
    Violin plot: tenure distribution split by churn status.

    Business question: Are churning customers shorter-tenured than retained ones?
    """
    caption = "Are churning customers shorter-tenured (newer) than retained customers?"

    order       = ["No", "Yes"]
    tick_labels = ["Active Customers\n(Retained)", "Churned Customers\n(Churned)"]
    pal         = [_GREEN, _RED]

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=figsize)
        fig.subplots_adjust(bottom=0.18)

        sns.violinplot(
            data=df,
            x="Churn", y="tenure",
            order=order,
            palette=pal,
            inner="quartile",
            linewidth=1.2,
            ax=ax,
        )
        ax.set_xticklabels(tick_labels)

        # Median annotations
        for i, label in enumerate(order):
            med = df[df["Churn"] == label]["tenure"].median()
            ax.text(
                i, med + 1.5,
                f"Median: {med:.0f} mo",
                ha="center", va="bottom",
                fontsize=9, color=_TEXT, fontweight="bold",
            )

        ax.set_title("Tenure Distribution by Churn Status",
                     fontweight="bold", pad=14)
        _style_ax(ax, xlabel="Churn Status", ylabel="Tenure (months)")
        _add_caption(fig, caption)

    return fig, caption


# ── Chart 7 (4b-1) ────────────────────────────────────────────────────────────

def chart_support_cols_by_churn(
    df: pd.DataFrame,
    figsize: tuple[int, int] = (9, 5),
) -> tuple[plt.Figure, str]:
    """
    Grouped bar chart: churn rate by TechSupport and OnlineSecurity status
    for internet customers only.
    """
    caption = "How strongly is lacking Tech Support or Online Security associated with churn among internet subscribers?"

    inet_df = df[df["has_internet"] == 1]
    services = {
        "TechSupport":    "Tech Support",
        "OnlineSecurity": "Online Security",
    }

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=figsize)
        fig.subplots_adjust(bottom=0.20)

        x = np.arange(len(services))
        width = 0.34

        yes_rates, no_rates, yes_counts, no_counts = [], [], [], []
        for col in services:
            sub_yes = inet_df[inet_df[col] == "Yes"]
            sub_no  = inet_df[inet_df[col] == "No"]
            yes_rates.append(sub_yes["Churn_Binary"].mean() * 100)
            no_rates.append(sub_no["Churn_Binary"].mean() * 100)
            yes_counts.append(len(sub_yes))
            no_counts.append(len(sub_no))

        bars_yes = ax.bar(
            x - width / 2, yes_rates, width,
            color=_GREEN, label="Service Active (Yes)",
            edgecolor=_BG, linewidth=1.5,
        )
        bars_no = ax.bar(
            x + width / 2, no_rates, width,
            color=_RED, label="Service Inactive (No)",
            edgecolor=_BG, linewidth=1.5,
        )

        for bar, rate, n in zip(bars_yes, yes_rates, yes_counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{rate:.1f}%\n(n={n:,})",
                ha="center", va="bottom",
                fontsize=9.5, color=_TEXT, fontweight="bold",
            )
        for bar, rate, n in zip(bars_no, no_rates, no_counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{rate:.1f}%\n(n={n:,})",
                ha="center", va="bottom",
                fontsize=9.5, color=_TEXT, fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(list(services.values()), fontsize=11)
        ax.legend(framealpha=0.2, fontsize=10)
        ax.set_title(
            "Churn Rate by Support & Security Services (Internet Customers Only)",
            fontweight="bold", pad=14,
        )
        _style_ax(ax, xlabel="Support / Security Service", ylabel="Churn Rate (%)")
        ax.set_ylim(0, max(no_rates) * 1.28)
        _add_caption(fig, caption)

    return fig, caption


# ── Chart 8 (4b-2) ────────────────────────────────────────────────────────────

def chart_correlation_heatmap(
    df: pd.DataFrame,
    figsize: tuple[int, int] = (8, 6),
) -> tuple[plt.Figure, str]:
    """
    Pearson correlation heatmap for tenure, MonthlyCharges, TotalCharges,
    and Churn_Binary.
    """
    caption = "How strongly are tenure and billing charges linearly associated with customer churn?"

    cols = {
        "tenure":         "Tenure (Months)",
        "MonthlyCharges": "Monthly Charges ($)",
        "TotalCharges":   "Total Charges ($)",
        "Churn_Binary":   "Churn (1=Yes, 0=No)",
    }
    corr = df[list(cols.keys())].rename(columns=cols).corr(method="pearson")

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=figsize)
        fig.subplots_adjust(bottom=0.22, left=0.22)

        sns.heatmap(
            corr,
            ax=ax,
            cmap="RdBu_r",
            center=0,
            vmin=-1, vmax=1,
            annot=True, fmt=".2f",
            annot_kws={"size": 10, "color": _TEXT, "fontweight": "bold"},
            linewidths=1.0,
            linecolor=_BG,
            cbar_kws={"shrink": 0.8},
        )

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(colors=_MUTED, labelsize=9)

        ax.set_title(
            "Correlation Heatmap: Tenure, Charges & Churn",
            fontweight="bold", pad=14, color=_TEXT,
        )
        ax.set_xlabel("Features", color=_TEXT, fontsize=11, labelpad=8)
        ax.set_ylabel("Features", color=_TEXT, fontsize=11, labelpad=8)
        ax.tick_params(axis="x", colors=_MUTED, labelsize=9.5, rotation=20)
        ax.tick_params(axis="y", colors=_MUTED, labelsize=9.5, rotation=0)
        _add_caption(fig, caption)

    return fig, caption


# ── Chart 9 (4b-3) ────────────────────────────────────────────────────────────

def chart_churn_by_segment(
    df: pd.DataFrame,
    figsize: tuple[int, int] = (9, 5),
) -> tuple[plt.Figure, str]:
    """
    Horizontal bar chart: churn rate by Customer_Segment.
    """
    caption = "Which customer demographic segment is most strongly associated with churn?"

    with plt.rc_context(_RC):
        grp = (
            df.groupby("Customer_Segment", observed=True)["Churn_Binary"]
            .agg(Count="count", Churned="sum")
            .assign(Churn_Rate=lambda x: x["Churned"] / x["Count"] * 100)
            .sort_values("Churn_Rate", ascending=True)
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=figsize)
        fig.subplots_adjust(left=0.20, bottom=0.18)

        bar_colors = [_bar_color(r) for r in grp["Churn_Rate"]]
        bars = ax.barh(
            grp["Customer_Segment"], grp["Churn_Rate"],
            color=bar_colors, height=0.52,
            edgecolor=_BG, linewidth=1.5,
        )

        max_rate = grp["Churn_Rate"].max()
        for bar, row in zip(bars, grp.itertuples()):
            ax.text(
                bar.get_width() + max_rate * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{row.Churn_Rate:.1f}%  (n={row.Count:,})",
                ha="left", va="center",
                fontsize=10, color=_TEXT, fontweight="bold",
            )

        ax.set_title("Churn Rate by Customer Segment", fontweight="bold", pad=14)
        _style_ax(ax, xlabel="Churn Rate (%)", ylabel="Customer Segment", grid_axis="x")
        ax.set_xlim(0, max_rate * 1.38)
        _add_caption(fig, caption)

    return fig, caption


# ── Chart 10 (4b-4) ───────────────────────────────────────────────────────────

def chart_churn_by_tenure_group(
    df: pd.DataFrame,
    figsize: tuple[int, int] = (9, 5),
) -> tuple[plt.Figure, str]:
    """
    Vertical bar chart: churn rate by Tenure_Group.
    """
    caption = "How does customer churn rate vary across tenure lifecycle stages?"

    order = [
        "New Customer",
        "Developing Customer",
        "Established Customer",
        "Loyal Customer",
    ]
    readable_labels = {
        "New Customer":         "New\n(0–12 mo)",
        "Developing Customer":  "Developing\n(13–24 mo)",
        "Established Customer": "Established\n(25–48 mo)",
        "Loyal Customer":       "Loyal\n(49–72 mo)",
    }

    with plt.rc_context(_RC):
        grp = (
            df.groupby("Tenure_Group", observed=True)["Churn_Binary"]
            .agg(Count="count", Churned="sum")
            .assign(Churn_Rate=lambda x: x["Churned"] / x["Count"] * 100)
            .reindex(order)
            .reset_index()
        )
        grp["Label"] = grp["Tenure_Group"].map(readable_labels)

        fig, ax = plt.subplots(figsize=figsize)
        fig.subplots_adjust(bottom=0.18)

        bar_colors = [_bar_color(r) for r in grp["Churn_Rate"]]
        bars = ax.bar(
            grp["Label"], grp["Churn_Rate"],
            color=bar_colors, width=0.52,
            edgecolor=_BG, linewidth=1.5,
        )

        for bar, row in zip(bars, grp.itertuples()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{row.Churn_Rate:.1f}%\n(n={row.Count:,})",
                ha="center", va="bottom",
                fontsize=10, color=_TEXT, fontweight="bold",
            )

        ax.set_title("Churn Rate by Tenure Group", fontweight="bold", pad=14)
        _style_ax(ax, xlabel="Tenure Lifecycle Group", ylabel="Churn Rate (%)")
        ax.set_ylim(0, grp["Churn_Rate"].max() * 1.28)
        _add_caption(fig, caption)

    return fig, caption


# ── CLI: generate all 10 charts ──────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from utils.preprocessing import load_clean_data
    from utils.features import add_features

    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "reports", "charts"
    )
    os.makedirs(out_dir, exist_ok=True)

    df_clean, _ = load_clean_data()
    df = add_features(df_clean)

    ALL_CHARTS = [
        ("01_churn_distribution",       chart_churn_distribution),
        ("02_churn_by_contract",        chart_churn_by_contract),
        ("03_churn_by_payment_method",  chart_churn_by_payment_method),
        ("04_churn_by_autopay",         chart_churn_by_autopay),
        ("05_monthly_charges_by_churn", chart_monthly_charges_by_churn),
        ("06_tenure_by_churn",          chart_tenure_by_churn),
        ("07_support_cols_by_churn",    chart_support_cols_by_churn),
        ("08_correlation_heatmap",      chart_correlation_heatmap),
        ("09_churn_by_segment",         chart_churn_by_segment),
        ("10_churn_by_tenure_group",    chart_churn_by_tenure_group),
    ]

    print("=" * 65)
    print("STEP 4 (4a + 4b) — Generating all 10 charts")
    print("=" * 65)

    for name, fn in ALL_CHARTS:
        fig, caption = fn(df)
        out_path = os.path.join(out_dir, f"{name}.png")
        fig.savefig(
            out_path, dpi=130, bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        plt.close(fig)
        print(f"  Saved  : {out_path}")
        print(f"  Caption: {caption}\n")