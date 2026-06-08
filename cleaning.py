"""
cleaning.py — Data Cleaning Pipeline
=====================================
Reads raw_data.csv, fixes common issues, and saves cleaned_data.csv

Issues handled:
  - Duplicate rows
  - Missing category (filled via product lookup)
  - Missing region (filled with mode)
  - Missing customer name (filled with 'Unknown')
  - Invalid quantities (negative values removed)
  - Missing unit prices (filled with per-product median)
  - Outliers in price & quantity (capped using IQR method)
"""

import pandas as pd

# ── 1. Load ──────────────────────────────────────────────────────────────────
df = pd.read_csv("raw_data.csv")

print("=== RAW DATA SUMMARY ===")
print(f"Rows         : {len(df)}")
print(f"Columns      : {list(df.columns)}")
print(f"Missing vals : {df.isnull().sum().sum()}")
print(f"Duplicates   : {df.duplicated().sum()}")
print()

# ── 2. Remove duplicates ─────────────────────────────────────────────────────
before = len(df)
df = df.drop_duplicates()
print(f"[Step 1] Removed {before - len(df)} duplicate rows")

# ── 3. Fill missing category from product lookup ──────────────────────────────
cat_map = {
    "Laptop": "Electronics",
    "Phone": "Electronics",
    "Tablet": "Electronics",
    "Monitor": "Peripherals",
    "Keyboard": "Peripherals",
    "Mouse": "Peripherals",
}
df["category"] = df["product"].map(cat_map)
print("[Step 2] Filled missing categories using product → category mapping")

# ── 4. Fill missing region with most common region ───────────────────────────
mode_region = df["region"].mode()[0]
df["region"] = df["region"].fillna(mode_region)
print(f"[Step 3] Filled missing regions with mode: '{mode_region}'")

# ── 5. Fill missing customer names ───────────────────────────────────────────
df["customer_name"] = df["customer_name"].fillna("Unknown")
print("[Step 4] Filled missing customer names with 'Unknown'")

# ── 6. Remove rows with invalid (negative) quantity ──────────────────────────
before = len(df)
df = df[df["quantity"] > 0]
print(f"[Step 5] Removed {before - len(df)} rows with negative quantity")

# ── 7. Fill missing unit prices with per-product median ──────────────────────
df["unit_price"] = df.groupby("product")["unit_price"].transform(
    lambda x: x.fillna(x.median())
)
print("[Step 6] Filled missing unit prices with per-product median")

# ── 8. Cap outliers using the IQR method ─────────────────────────────────────
def cap_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return series.clip(lower, upper)

df["unit_price"] = cap_outliers(df["unit_price"])
df["quantity"] = cap_outliers(df["quantity"])
print("[Step 7] Capped outliers in unit_price and quantity using IQR method")

# ── 9. Add derived columns ───────────────────────────────────────────────────
df["revenue"] = (df["quantity"] * df["unit_price"]).round(2)
df["month"] = pd.to_datetime(df["date"]).dt.strftime("%b")
df["month_num"] = pd.to_datetime(df["date"]).dt.month
print("[Step 8] Added derived columns: revenue, month, month_num")

# ── 10. Save ─────────────────────────────────────────────────────────────────
df.to_csv("cleaned_data.csv", index=False)

print()
print("=== CLEANED DATA SUMMARY ===")
print(f"Rows         : {len(df)}")
print(f"Missing vals : {df.isnull().sum().sum()}")
print(f"Total revenue: ${df['revenue'].sum():,.2f}")
print(f"Avg order val: ${df['revenue'].mean():,.2f}")
print()
print("Saved → cleaned_data.csv")
