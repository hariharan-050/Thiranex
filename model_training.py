"""
model_training.py — Phase 3: Baseline ML Models
Retail Sales ML Project

Models: Logistic Regression, Decision Tree, Random Forest
Task:   Binary classification — predict high_rating (rating >= 4)
Output: ml_evaluation.png (model comparison report)

Run:  python model_training.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use("Agg")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
from sklearn.preprocessing import StandardScaler

from utils import (
    load_data, add_high_rating_label, engineer_features,
    encode_categoricals, save_fig, style_axis
)

DATA_PATH   = "cleaned_dataset.csv"
OUTPUT_PATH = "ml_evaluation.png"

FEATURES = [
    "quantity", "unit_price", "discount_pct", "customer_age",
    "month", "revenue", "revenue_per_unit", "has_discount",
    "category_enc", "region_enc", "payment_method_enc",
]

BG   = "#f0f4f8"
CARD = "#ffffff"




def prepare_data(path: str = DATA_PATH):
    df = load_data(path)
    df = add_high_rating_label(df)
    df = engineer_features(df)
    df = encode_categoricals(df, ["category", "region", "payment_method"])

    feat_cols = [f for f in FEATURES if f in df.columns]
    X = df[feat_cols].fillna(df[feat_cols].median())
    y = df["high_rating"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale for Logistic Regression
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"[model] Train: {X_train.shape}  Test: {X_test.shape}  "
          f"Features: {feat_cols}")
    return X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, feat_cols




def train_models(X_train, X_test, y_train, y_test,
                 X_train_sc, X_test_sc):
    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            X_train_sc, X_test_sc
        ),
        "Decision Tree": (
            DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
            X_train, X_test
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
            X_train, X_test
        ),
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, (model, Xtr, Xte) in models.items():
        model.fit(Xtr, y_train)
        y_pred  = model.predict(Xte)
        y_proba = model.predict_proba(Xte)[:, 1]

        acc  = accuracy_score(y_test, y_pred) * 100
        auc  = roc_auc_score(y_test, y_proba)
        cm   = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        cv_scores = cross_val_score(model, Xtr, y_train,
                                    cv=cv, scoring="accuracy") * 100

        results[name] = {
            "model":     model,
            "acc":       acc,
            "auc":       auc,
            "cm":        cm,
            "fpr":       fpr,
            "tpr":       tpr,
            "cv_scores": cv_scores,
            "report":    classification_report(y_test, y_pred),
        }
        print(f"  {name:<22}  Acc={acc:.1f}%  AUC={auc:.3f}  "
              f"CV={cv_scores.mean():.1f}% ± {cv_scores.std():.1f}%")

    return results




def build_report(results: dict, feat_cols: list, output: str = OUTPUT_PATH):
    names  = list(results.keys())
    accs   = [results[n]["acc"]  for n in names]
    aucs   = [results[n]["auc"] * 100 for n in names]
    colors = ["#2d6a4f", "#e67e22", "#3498db"]

    fig = plt.figure(figsize=(16, 14), facecolor=BG)
    fig.suptitle("ML Model Evaluation Report\nLogistic Regression · Decision Tree · Random Forest",
                 fontsize=15, fontweight="bold", y=0.99, color="#1a1a2e")

    gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.4,
                          top=0.94, bottom=0.05, left=0.07, right=0.97)

    # ── Panel 1: Accuracy & AUC bar chart ──────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    x   = np.arange(len(names))
    w   = 0.35
    ax1.bar(x - w/2, accs, width=w, label="Test Accuracy (%)", color=colors, alpha=0.9)
    ax1.bar(x + w/2, aucs, width=w, label="ROC-AUC (%)",
            color=colors, alpha=0.45, hatch="//", edgecolor="white")
    for i, (a, u) in enumerate(zip(accs, aucs)):
        ax1.text(i - w/2, a + 0.3, f"{a:.1f}", ha="center", fontsize=8)
        ax1.text(i + w/2, u + 0.3, f"{u:.1f}", ha="center", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylim(0, 110)
    ax1.legend(fontsize=8)
    style_axis(ax1, "Model Comparison: Accuracy & AUC", "", "Score (%)")

    
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random (0.50)")
    for (name, res), col in zip(results.items(), colors):
        ax2.plot(res["fpr"], res["tpr"], color=col, lw=2,
                 label=f"{name.split()[0]} ({res['auc']:.2f})")
    ax2.legend(fontsize=7, loc="lower right")
    style_axis(ax2, "ROC Curves", "False Positive Rate", "True Positive Rate")

  
    cm_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    for ax, name, col in zip(cm_axes, names, colors):
        cm = results[name]["cm"]
        sns.heatmap(cm, annot=True, fmt="d", ax=ax,
                    cmap=sns.light_palette(col, as_cmap=True),
                    cbar=False, linewidths=1, linecolor="white",
                    annot_kws={"size": 14, "weight": "bold"})
        ax.set_xticklabels(["Low (0)", "High (1)"], fontsize=8)
        ax.set_yticklabels(["Low (0)", "High (1)"], fontsize=8, rotation=0)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)
        ax.set_title(f"{name}\nConfusion Matrix", fontsize=10, fontweight="bold")

    
    ax_fi = fig.add_subplot(gs[2, :2])
    rf_model = results["Random Forest"]["model"]
    importances = rf_model.feature_importances_
    fi_df = pd.DataFrame({
        "feature": feat_cols, "importance": importances
    }).sort_values("importance")
    n = min(9, len(fi_df))
    fi_df = fi_df.tail(n)
    thresh_high   = fi_df["importance"].quantile(0.67)
    thresh_medium = fi_df["importance"].quantile(0.33)
    bar_colors = [
        "#2d6a4f" if v >= thresh_high else
        "#52b788" if v >= thresh_medium else
        "#b7e4c7"
        for v in fi_df["importance"]
    ]
    bars = ax_fi.barh(fi_df["feature"], fi_df["importance"],
                      color=bar_colors, edgecolor="none")
    for bar, val in zip(bars, fi_df["importance"]):
        ax_fi.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                   f"{val:.3f}", va="center", fontsize=8)
    style_axis(ax_fi, "Feature Importance — Random Forest",
               "Importance Score", "")
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="#2d6a4f", label="High impact"),
        Patch(facecolor="#52b788", label="Medium impact"),
        Patch(facecolor="#b7e4c7", label="Lower impact"),
    ]
    ax_fi.legend(handles=legend_elems, fontsize=8, loc="lower right")

   
    ax_cv = fig.add_subplot(gs[2, 2])
    cv_data = [results[n]["cv_scores"] for n in names]
    bp = ax_cv.boxplot(cv_data, patch_artist=True, widths=0.5,
                       medianprops=dict(color="black", lw=2))
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    ax_cv.set_xticklabels(["Logistic", "Decision", "Random"], fontsize=8)
    style_axis(ax_cv, "5-Fold CV Accuracy (%)", "", "Accuracy (%)")

    save_fig(fig, output)
    print(f"[model] Report saved → '{output}'")




def print_reports(results: dict) -> None:
    for name, res in results.items():
        print(f"\n{'─'*45}\n  {name}\n{'─'*45}")
        print(res["report"])



def run_training(path: str = DATA_PATH) -> dict:
    print("\n" + "─" * 50)
    print("  Phase 3 — Model Training & Evaluation")
    print("─" * 50)

    X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, feat_cols = prepare_data(path)

    print("\n[model] Training models...")
    results = train_models(X_train, X_test, y_train, y_test,
                           X_train_sc, X_test_sc)

    print_reports(results)
    build_report(results, feat_cols)

    best = max(results, key=lambda n: results[n]["auc"])
    print(f"\n[model] Best model by AUC: {best}  "
          f"(AUC = {results[best]['auc']:.3f})")
    print("[model] Training complete ✓\n")
    return results


if __name__ == "__main__":
    run_training()
