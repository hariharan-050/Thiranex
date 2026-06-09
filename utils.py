"""
utils.py — Shared utility functions used across all project modules.
Retail Sales ML Project | Phase 1–4
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for scripts)


# ──────

def load_data(filepath: str) -> pd.DataFrame:
    """Load a CSV file and return a DataFrame. Raises FileNotFoundError if missing."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[utils] Loaded {df.shape[0]} rows × {df.shape[1]} columns from '{filepath}'")
    return df


def save_data(df: pd.DataFrame, filepath: str) -> None:
    """Save a DataFrame to CSV without the index."""
    df.to_csv(filepath, index=False)
    print(f"[utils] Saved {df.shape[0]} rows → '{filepath}'")




def summarise(df: pd.DataFrame) -> None:
    """Print a concise summary: shape, dtypes, nulls, and numeric stats."""
    print("\n" + "=" * 55)
    print(f"  Dataset Summary: {df.shape[0]} rows × {df.shape[1]} cols")
    print("=" * 55)

    print("\n--- Column types & null counts ---")
    info = pd.DataFrame({
        "dtype":    df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "nulls":    df.isna().sum(),
        "null_%":   (df.isna().mean() * 100).round(1),
    })
    print(info.to_string())

    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        print("\n--- Numeric statistics ---")
        print(numeric.describe().T.round(2).to_string())
    print("=" * 55 + "\n")




def add_high_rating_label(df: pd.DataFrame, threshold: float = 4.0) -> pd.DataFrame:
    """
    Add a binary column 'high_rating':
        1  →  rating >= threshold  (positive class)
        0  →  rating <  threshold
    """
    df = df.copy()
    df["high_rating"] = (df["rating"] >= threshold).astype(int)
    pos = df["high_rating"].mean() * 100
    print(f"[utils] 'high_rating' label added  "
          f"(threshold={threshold}) | positive class = {pos:.1f}%")
    return df



def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived numeric features on top of the cleaned dataset.
    Returns a new DataFrame — does not mutate the original.
    """
    df = df.copy()

    df["revenue_per_unit"] = df["revenue"] / df["quantity"].replace(0, np.nan)

    df["has_discount"] = (df["discount_pct"] > 0).astype(int)

    df["age_group"] = pd.cut(
        df["customer_age"],
        bins=[0, 30, 45, 60, 120],
        labels=["18-30", "31-45", "46-60", "61+"],
    )

    
    df["quarter"] = pd.cut(
        df["month"],
        bins=[0, 3, 6, 9, 12],
        labels=["Q1", "Q2", "Q3", "Q4"],
    )

    print("[utils] Feature engineering done — added: "
          "revenue_per_unit, has_discount, age_group, quarter")
    return df




def encode_categoricals(df: pd.DataFrame,
                         columns: list[str]) -> pd.DataFrame:
    """
    Label-encode a list of categorical columns in-place (new col = col + '_enc').
    Returns the modified DataFrame.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col + "_enc"] = df[col].astype("category").cat.codes
    print(f"[utils] Encoded columns: {columns}")
    return df




PALETTE = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c", "#9b59b6"]


def save_fig(fig: plt.Figure, path: str) -> None:
    """Save a matplotlib figure, then close it to free memory."""
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[utils] Figure saved → '{path}'")


def style_axis(ax: plt.Axes, title: str = "",
               xlabel: str = "", ylabel: str = "") -> None:
    """Apply consistent axis styling across all plots."""
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
