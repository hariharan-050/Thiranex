"""
data_cleaning.py — Phase 1: Data Cleaning Pipeline
Retail Sales ML Project

Steps:
  1. Load raw data
  2. Drop exact duplicates
  3. Fill missing numeric values with column medians
  4. Cap outliers using the IQR method
  5. Engineer the 'revenue' column (quantity × unit_price × (1 - discount_pct/100))
  6. Save the cleaned dataset

Run:  python data_cleaning.py
"""

import numpy as np
import pandas as pd
from utils import load_data, save_data, summarise

RAW_PATH     = "cleaned_dataset.csv"   # re-used as source (already cleaned once)
CLEANED_PATH = "cleaned_dataset.csv"
NUMERIC_COLS = ["quantity", "unit_price", "discount_pct", "customer_age", "rating"]




def load(path: str = RAW_PATH) -> pd.DataFrame:
    return load_data(path)


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    print(f"[cleaning] Duplicates removed: {removed}  (rows left: {len(df)})")
    return df




def fill_missing(df: pd.DataFrame,
                 cols: list[str] = NUMERIC_COLS) -> pd.DataFrame:
    total_filled = 0
    for col in cols:
        if col in df.columns:
            n = df[col].isna().sum()
            if n:
                df[col] = df[col].fillna(df[col].median())
                total_filled += n
    print(f"[cleaning] Missing values filled: {total_filled} cells (median imputation)")
    return df



def cap_outliers(df: pd.DataFrame,
                 cols: list[str] = NUMERIC_COLS,
                 factor: float = 1.5) -> pd.DataFrame:
    """
    Replace values outside [Q1 - factor*IQR, Q3 + factor*IQR]
    with the boundary value (Winsorising / capping).
    """
    capped = 0
    for col in cols:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - factor * iqr, q3 + factor * iqr
        mask = (df[col] < lo) | (df[col] > hi)
        capped += mask.sum()
        df[col] = df[col].clip(lower=lo, upper=hi)
    print(f"[cleaning] Outliers capped: {capped} values (IQR × {factor})")
    return df




def engineer_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute revenue = quantity × unit_price × (1 - discount_pct / 100).
    Overwrites the column if it already exists.
    """
    df["revenue"] = (
        df["quantity"] * df["unit_price"] * (1 - df["discount_pct"] / 100)
    ).round(2)
    print(f"[cleaning] 'revenue' column engineered  "
          f"(min={df['revenue'].min():.2f}, max={df['revenue'].max():.2f})")
    return df




def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parse 'date' and add month + month_name columns."""
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["month"]      = df["date"].dt.month
        df["month_name"] = df["date"].dt.strftime("%b")
    return df




def clean_pipeline(raw_path: str = RAW_PATH,
                   save_path: str = CLEANED_PATH) -> pd.DataFrame:
    print("\n" + "─" * 50)
    print("  Phase 1 — Data Cleaning Pipeline")
    print("─" * 50)

    df = load(raw_path)
    summarise(df)

    df = drop_duplicates(df)
    df = fill_missing(df)
    df = cap_outliers(df)
    df = engineer_revenue(df)
    df = add_time_columns(df)

    summarise(df)
    save_data(df, save_path)
    print("[cleaning] Pipeline complete ✓\n")
    return df


if __name__ == "__main__":
    clean_pipeline()
