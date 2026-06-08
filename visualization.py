"""
visualization.py — Sales Data Visualization
=============================================
Reads cleaned_data.csv (run cleaning.py first) and produces:
  1. Revenue by product        (horizontal bar chart)
  2. Revenue by region         (pie chart)
  3. Monthly revenue trend     (line chart)
  4. Orders by category        (bar chart)
  5. Quantity vs revenue       (scatter plot)

Output: saves dashboard.png in the current directory
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Load cleaned data ─────────────────────────────────────────────────────────
df = pd.read_csv("cleaned_data.csv")
df["date"] = pd.to_datetime(df["date"])

# ── Style setup ───────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
BLUE    = "#3266AD"
COLORS  = ["#3266AD", "#1D9E75", "#D85A30", "#9E6B1D", "#7B3DAD", "#AD3258"]
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Sales Dashboard — 2024", fontsize=18, fontweight="bold", y=1.01)
plt.subplots_adjust(hspace=0.45, wspace=0.35)

# ── 1. Revenue by product ─────────────────────────────────────────────────────
ax1 = axes[0, 0]
rev_prod = df.groupby("product")["revenue"].sum().sort_values()
bars = ax1.barh(rev_prod.index, rev_prod.values, color=BLUE, edgecolor="none", height=0.6)
ax1.set_title("Revenue by product", fontweight="bold")
ax1.set_xlabel("Revenue ($)")
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
ax1.bar_label(bars, labels=[f"${v/1000:.1f}k" for v in rev_prod.values], padding=4, fontsize=9)
ax1.set_xlim(0, rev_prod.max() * 1.2)
sns.despine(ax=ax1, left=True, bottom=False)
ax1.grid(axis="x", alpha=0.3)
ax1.set_yticks(range(len(rev_prod)))
ax1.set_yticklabels(rev_prod.index, fontsize=10)

# ── 2. Revenue by region (pie) ────────────────────────────────────────────────
ax2 = axes[0, 1]
rev_region = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
wedges, texts, autotexts = ax2.pie(
    rev_region.values,
    labels=rev_region.index,
    autopct="%1.1f%%",
    colors=COLORS[:len(rev_region)],
    startangle=140,
    wedgeprops={"edgecolor": "white", "linewidth": 2},
    pctdistance=0.75,
)
for t in autotexts:
    t.set_fontsize(9)
ax2.set_title("Revenue by region", fontweight="bold")

# ── 3. Monthly revenue trend (line) ──────────────────────────────────────────
ax3 = axes[0, 2]
monthly = (
    df.groupby(["month_num", "month"])["revenue"]
    .sum()
    .reset_index()
    .sort_values("month_num")
)
ax3.plot(monthly["month"], monthly["revenue"], marker="o", color=BLUE,
         linewidth=2, markersize=5, markerfacecolor="white", markeredgewidth=2)
ax3.fill_between(range(len(monthly)), monthly["revenue"], alpha=0.1, color=BLUE)
ax3.set_xticks(range(len(monthly)))
ax3.set_xticklabels(monthly["month"], rotation=45, fontsize=8)
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
ax3.set_title("Monthly revenue trend", fontweight="bold")
ax3.set_xlabel("Month")
ax3.set_ylabel("Revenue ($)")
sns.despine(ax=ax3)

# ── 4. Orders by category (bar) ──────────────────────────────────────────────
ax4 = axes[1, 0]
orders_cat = df.groupby("category")["order_id"].count()
bars4 = ax4.bar(orders_cat.index, orders_cat.values,
                color=["#3266AD", "#1D9E75"], edgecolor="none", width=0.5)
ax4.set_title("Orders by category", fontweight="bold")
ax4.set_ylabel("Number of orders")
ax4.bar_label(bars4, padding=4, fontsize=10, fontweight="bold")
ax4.set_ylim(0, orders_cat.max() * 1.2)
sns.despine(ax=ax4, left=True)
ax4.grid(axis="y", alpha=0.3)

# ── 5. Quantity vs Revenue scatter ────────────────────────────────────────────
ax5 = axes[1, 1]
cat_colors = {"Electronics": "#3266AD", "Peripherals": "#1D9E75"}
for cat, grp in df.groupby("category"):
    ax5.scatter(grp["quantity"], grp["revenue"],
                color=cat_colors[cat], alpha=0.5, s=30, label=cat, edgecolors="none")
ax5.set_title("Quantity vs revenue", fontweight="bold")
ax5.set_xlabel("Quantity")
ax5.set_ylabel("Revenue ($)")
ax5.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
ax5.legend(fontsize=9, frameon=False)
sns.despine(ax=ax5)

# ── 6. Top 5 customers by revenue ────────────────────────────────────────────
ax6 = axes[1, 2]
top_customers = (
    df[df["customer_name"] != "Unknown"]
    .groupby("customer_name")["revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)
bars6 = ax6.bar(top_customers.index, top_customers.values,
                color=COLORS[:len(top_customers)], edgecolor="none", width=0.5)
ax6.set_title("Top customers by revenue", fontweight="bold")
ax6.set_ylabel("Revenue ($)")
ax6.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
ax6.bar_label(bars6, labels=[f"${v/1000:.1f}k" for v in top_customers.values],
              padding=4, fontsize=9)
ax6.set_ylim(0, top_customers.max() * 1.2)
sns.despine(ax=ax6, left=True)
ax6.grid(axis="y", alpha=0.3)
ax6.tick_params(axis="x", rotation=15)

# ── Save ──────────────────────────────────────────────────────────────────────
plt.savefig("dashboard.png", dpi=150, bbox_inches="tight")
print("Dashboard saved → dashboard.png")
plt.show()
