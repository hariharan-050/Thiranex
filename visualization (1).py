"""
visualization.py — Phase 2 (continued): Sales Dashboard
Retail Sales ML Project

Generates a 6-panel dashboard:
  1. Revenue by Category
  2. Payment Method distribution (donut)
  3. Monthly Revenue Trend
  4. Revenue by Region
  5. Customer Ratings histogram
  6. Region × Category average revenue heatmap

Output: sales_dashboard.png

Run:  python visualization.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.use("Agg")

from utils import load_data, add_high_rating_label, PALETTE, save_fig, style_axis

DATA_PATH   = "cleaned_dataset.csv"
OUTPUT_PATH = "sales_dashboard.png"

BG   = "#0d1117"
CARD = "#161b22"


# ──────────────────────────────────────────────
# Panel helpers
# ──────────────────────────────────────────────

def panel_revenue_by_category(ax, df):
    grp = df.groupby("category")["revenue"].sum().sort_values()
    colors = sns.color_palette("Greens_d", len(grp))
    bars = ax.barh(grp.index, grp.values, color=colors, edgecolor="none")
    for bar, val in zip(bars, grp.values):
        ax.text(val + grp.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"₹{val/1000:.0f}K", va="center", fontsize=8, color="white")
    style_axis(ax, "Revenue by Category", "Revenue (₹)", "")
    ax.set_facecolor(CARD)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.title.set_color("white")


def panel_payment_donut(ax, df):
    grp = df["payment_method"].value_counts()
    colors = ["#2ecc71", "#3498db", "#e67e22", "#9b59b6"]
    wedges, texts, autotexts = ax.pie(
        grp.values, labels=grp.index, autopct="%1.0f%%",
        colors=colors, pctdistance=0.78,
        wedgeprops=dict(width=0.5, edgecolor=BG, linewidth=2),
    )
    for t in texts + autotexts:
        t.set_color("white")
        t.set_fontsize(8)
    ax.set_title("Payment Methods", fontsize=13, fontweight="bold",
                 color="white", pad=10)
    ax.set_facecolor(CARD)


def panel_monthly_trend(ax, df):
    monthly = df.groupby("month")["revenue"].sum()
    ax.plot(monthly.index, monthly.values, color="#2ecc71",
            marker="o", lw=2, ms=5)
    ax.fill_between(monthly.index, monthly.values, alpha=0.15, color="#2ecc71")
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    ax.set_xticks(monthly.index)
    ax.set_xticklabels([month_names[m-1] for m in monthly.index],
                       fontsize=8, color="white")
    for x, y in zip(monthly.index, monthly.values):
        ax.annotate(f"₹{y/1000:.0f}K", (x, y), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7, color="#2ecc71")
    style_axis(ax, "Monthly Revenue Trend", "", "Revenue (₹)")
    ax.set_facecolor(CARD)
    ax.tick_params(colors="white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")


def panel_revenue_by_region(ax, df):
    grp = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
    colors = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c"]
    bars = ax.bar(grp.index, grp.values, color=colors, edgecolor="none", width=0.6)
    for bar, val in zip(bars, grp.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + grp.max() * 0.01,
                f"₹{val/1000:.0f}K", ha="center", fontsize=8, color="white")
    style_axis(ax, "Revenue by Region", "", "Revenue (₹)")
    ax.set_facecolor(CARD)
    ax.tick_params(colors="white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")


def panel_ratings(ax, df):
    counts = df["rating"].value_counts().sort_index()
    colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#2ecc71"]
    bars = ax.bar(counts.index.astype(str), counts.values,
                  color=colors[:len(counts)], edgecolor="none", width=0.6)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1,
                str(val), ha="center", fontsize=9, color="white")
    style_axis(ax, "Customer Ratings", "Rating (stars)", "Count")
    ax.set_facecolor(CARD)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")


def panel_heatmap(ax, df):
    pivot = df.pivot_table(
        values="revenue", index="region",
        columns="category", aggfunc="mean"
    ).round(0)
    sns.heatmap(
        pivot, ax=ax, cmap="YlGn", annot=True, fmt=".0f",
        linewidths=0.5, linecolor=BG,
        annot_kws={"size": 7},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Avg Revenue (Region × Category)",
                 fontsize=13, fontweight="bold", color="white", pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(colors="white", labelsize=8)
    ax.set_facecolor(CARD)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # rotate x labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")


# ──────────────────────────────────────────────
# Assemble dashboard
# ──────────────────────────────────────────────

def build_dashboard(df: pd.DataFrame, output: str = OUTPUT_PATH) -> None:
    fig = plt.figure(figsize=(18, 14), facecolor=BG)
    fig.suptitle(
        "Retail Sales Dashboard\n"
        "Cleaned & Processed from Raw Sales Data  •  Jan–Oct 2024",
        fontsize=16, fontweight="bold", color="white", y=0.98
    )

    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35,
                          top=0.92, bottom=0.05, left=0.07, right=0.97)

    ax1 = fig.add_subplot(gs[0, :2])   # wide: category
    ax2 = fig.add_subplot(gs[0, 2])    # donut
    ax3 = fig.add_subplot(gs[1, :])    # full-width trend
    ax4 = fig.add_subplot(gs[2, 0])    # region
    ax5 = fig.add_subplot(gs[2, 1])    # ratings
    ax6 = fig.add_subplot(gs[2, 2])    # heatmap

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.set_facecolor(CARD)
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    panel_revenue_by_category(ax1, df)
    panel_payment_donut(ax2, df)
    panel_monthly_trend(ax3, df)
    panel_revenue_by_region(ax4, df)
    panel_ratings(ax5, df)
    panel_heatmap(ax6, df)

    save_fig(fig, output)
    print(f"[viz] Dashboard saved → '{output}'")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def run_visualization(path: str = DATA_PATH) -> None:
    print("\n" + "─" * 50)
    print("  Phase 2 — Visualization")
    print("─" * 50)
    df = load_data(path)
    build_dashboard(df)
    print("[viz] Visualization complete ✓\n")


if __name__ == "__main__":
    run_visualization()
