"""
eda.py — Phase 2: Exploratory Data Analysis
Retail Sales ML Project

Produces text summaries and prints key insights for:
  - Distribution of numeric features
  - Class balance for the target label
  - Correlation analysis
  - Group-level aggregations (category, region, payment method)

Run:  python eda.py
"""

import pandas as pd
import numpy as np
from utils import load_data, summarise, add_high_rating_label, engineer_features, encode_categoricals

DATA_PATH = "cleaned_dataset.csv"

CAT_COLS = ["category", "region", "payment_method"]
NUM_COLS = ["quantity", "unit_price", "discount_pct", "customer_age", "rating", "revenue"]




def check_class_balance(df: pd.DataFrame) -> None:
    df = add_high_rating_label(df)
    counts = df["high_rating"].value_counts()
    pct    = df["high_rating"].value_counts(normalize=True) * 100
    print("\n--- Target class balance ---")
    print(pd.DataFrame({"count": counts, "pct (%)": pct.round(1)}).to_string())
    ratio = counts[0] / counts[1] if counts[1] else float("inf")
    print(f"  Imbalance ratio (neg:pos): {ratio:.2f}:1")
    if ratio > 3:
        print("  ⚠ Significant imbalance — consider class_weight='balanced' or SMOTE")



def numeric_distributions(df: pd.DataFrame) -> None:
    print("\n--- Numeric feature distributions ---")
    for col in NUM_COLS:
        if col not in df.columns:
            continue
        s = df[col]
        skew = s.skew()
        kurt = s.kurtosis()
        flag = " ← highly skewed" if abs(skew) > 1 else ""
        print(f"  {col:<18}  mean={s.mean():>9.2f}  std={s.std():>8.2f}  "
              f"skew={skew:>6.2f}  kurt={kurt:>6.2f}{flag}")




def correlation_analysis(df: pd.DataFrame) -> None:
    print("\n--- Correlation with 'revenue' ---")
    num_df = df[NUM_COLS].dropna()
    corr = num_df.corr()["revenue"].drop("revenue").sort_values(ascending=False)
    for feat, val in corr.items():
        bar = "█" * int(abs(val) * 20)
        sign = "+" if val >= 0 else "−"
        print(f"  {feat:<20} {sign}{abs(val):.3f}  {bar}")

    print("\n--- Correlation with 'rating' ---")
    if "rating" in num_df.columns:
        corr2 = num_df.corr()["rating"].drop("rating").sort_values(ascending=False)
        for feat, val in corr2.items():
            bar = "█" * int(abs(val) * 20)
            sign = "+" if val >= 0 else "−"
            print(f"  {feat:<20} {sign}{abs(val):.3f}  {bar}")


def group_analysis(df: pd.DataFrame) -> None:
    for col in CAT_COLS:
        if col not in df.columns:
            continue
        print(f"\n--- Revenue & rating by {col} ---")
        grp = (
            df.groupby(col)
              .agg(
                  orders   = ("revenue", "count"),
                  avg_rev  = ("revenue", "mean"),
                  total_rev= ("revenue", "sum"),
                  avg_rating=("rating", "mean"),
              )
              .sort_values("total_rev", ascending=False)
              .round(2)
        )
        print(grp.to_string())




def monthly_trend(df: pd.DataFrame) -> None:
    if "month" not in df.columns:
        return
    print("\n--- Monthly revenue trend ---")
    trend = df.groupby("month")["revenue"].sum().round(0)
    for month, rev in trend.items():
        bar = "█" * int(rev / trend.max() * 30)
        print(f"  Month {month:>2}: {bar}  ₹{rev:,.0f}")




def feature_engineering_preview(df: pd.DataFrame) -> None:
    df = engineer_features(df)
    df = encode_categoricals(df, ["category", "region", "payment_method",
                                  "age_group", "quarter"])
    print("\n--- After feature engineering, final columns ---")
    print([c for c in df.columns])
    print(f"\n  Shape: {df.shape}")




def run_eda(path: str = DATA_PATH) -> None:
    print("\n" + "─" * 50)
    print("  Phase 2 — Exploratory Data Analysis")
    print("─" * 50)

    df = load_data(path)
    summarise(df)

    check_class_balance(df)
    numeric_distributions(df)
    correlation_analysis(df)
    group_analysis(df)
    monthly_trend(df)
    feature_engineering_preview(df)

    print("\n[eda] Analysis complete ✓\n")


if __name__ == "__main__":
    run_eda()
