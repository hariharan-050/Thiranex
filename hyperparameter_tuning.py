"""
hyperparameter_tuning.py — Phase 4: Hyperparameter Tuning
Retail Sales ML Project

Techniques covered:
  1. GridSearchCV  — exhaustive search on a small grid
  2. RandomizedSearchCV — efficient sampling on a large grid
  3. Learning curves — diagnose under/overfitting visually
  4. Threshold tuning — optimise precision/recall trade-off
  5. SMOTE + tuned model — handle class imbalance
  6. Final comparison — baseline vs tuned models

Output: phase4_tuning_report.png

Run:  python hyperparameter_tuning.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
matplotlib.use("Agg")

from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV,
    StratifiedKFold, learning_curve, cross_val_score
)
from sklearn.linear_model  import LogisticRegression
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics       import (
    accuracy_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint, uniform

from utils import (
    load_data, add_high_rating_label, engineer_features,
    encode_categoricals, save_fig, style_axis
)

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("[tuning] imbalanced-learn not installed — SMOTE section will be skipped")

DATA_PATH   = "cleaned_dataset.csv"
OUTPUT_PATH = "phase4_tuning_report.png"

FEATURES = [
    "quantity", "unit_price", "discount_pct", "customer_age",
    "month", "revenue", "revenue_per_unit", "has_discount",
    "category_enc", "region_enc", "payment_method_enc",
]

BG   = "#f0f4f8"
CARD = "#ffffff"
GREEN   = "#2d6a4f"
TEAL    = "#52b788"
ORANGE  = "#e67e22"
BLUE    = "#3498db"
PURPLE  = "#9b59b6"




def prepare(path=DATA_PATH):
    df = load_data(path)
    df = add_high_rating_label(df)
    df = engineer_features(df)
    df = encode_categoricals(df, ["category", "region", "payment_method"])

    cols = [f for f in FEATURES if f in df.columns]
    X = df[cols].fillna(df[cols].median())
    y = df["high_rating"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)

    print(f"[tuning] Train={X_tr.shape}  Test={X_te.shape}  "
          f"Positive class={y_tr.mean()*100:.1f}%")
    return X_tr, X_te, y_tr, y_te, X_tr_sc, X_te_sc, cols, scaler




def grid_search_lr(X_tr_sc, y_tr):
    """
    GridSearchCV exhaustively tries every combination in the grid.
    Good for small grids (< a few hundred combos).
    """
    print("\n[tuning] Step 1 — GridSearchCV on Logistic Regression...")

    param_grid = {
        "C":        [0.001, 0.01, 0.1, 1, 10, 100],   # inverse regularisation strength
        "penalty":  ["l1", "l2"],                       # L1=sparse, L2=smooth
        "solver":   ["liblinear"],                      # supports both L1 & L2
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        param_grid,
        cv=cv,
        scoring="roc_auc",      # optimise AUC, not raw accuracy
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_tr_sc, y_tr)

    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV AUC: {grid.best_score_:.4f}")

    # Build a results table for plotting
    results_df = pd.DataFrame(grid.cv_results_)
    results_df = results_df[["param_C", "param_penalty",
                              "mean_test_score", "std_test_score"]].copy()
    results_df.columns = ["C", "penalty", "mean_auc", "std_auc"]
    results_df["C"] = results_df["C"].astype(float)

    return grid.best_estimator_, grid.best_params_, grid.best_score_, results_df




def random_search_rf(X_tr, y_tr):
    """
    RandomizedSearchCV samples n_iter combinations at random.
    Much faster than GridSearch for large hyperparameter spaces.
    """
    print("\n[tuning] Step 2 — RandomizedSearchCV on Random Forest...")

    param_dist = {
        "n_estimators":      randint(50, 500),       # number of trees
        "max_depth":         [None, 3, 5, 8, 12, 20], # max tree depth
        "min_samples_split": randint(2, 20),          # min samples to split a node
        "min_samples_leaf":  randint(1, 10),          # min samples in a leaf
        "max_features":      ["sqrt", "log2", None],  # features per split
        "class_weight":      ["balanced", "balanced_subsample"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rscv = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=param_dist,
        n_iter=40,              # sample 40 random combos
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    rscv.fit(X_tr, y_tr)

    print(f"  Best params: {rscv.best_params_}")
    print(f"  Best CV AUC: {rscv.best_score_:.4f}")

    return rscv.best_estimator_, rscv.best_params_, rscv.best_score_



def tune_gradient_boosting(X_tr, y_tr):
    """
    GradientBoostingClassifier — another strong tree ensemble.
    Uses RandomizedSearch.
    """
    print("\n[tuning] Step 3 — RandomizedSearchCV on Gradient Boosting...")

    param_dist = {
        "n_estimators":   randint(50, 300),
        "max_depth":      randint(2, 6),
        "learning_rate":  uniform(0.01, 0.3),
        "subsample":      uniform(0.6, 0.4),
        "min_samples_leaf": randint(1, 15),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rscv = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=42),
        param_distributions=param_dist,
        n_iter=30,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    rscv.fit(X_tr, y_tr)

    print(f"  Best params: {rscv.best_params_}")
    print(f"  Best CV AUC: {rscv.best_score_:.4f}")

    return rscv.best_estimator_, rscv.best_score_




def compute_learning_curves(model, X, y, label="Model"):
    """
    Learning curve: train & val score as a function of training set size.
    - If val score keeps rising with more data → more data will help.
    - If val score plateaued → focus on model / features, not data.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=np.linspace(0.1, 1.0, 8),
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )
    return train_sizes, train_scores, val_scores




def threshold_analysis(model, X_te, y_te, is_scaled=False):
    """
    By default sklearn uses 0.5 as the decision threshold.
    Adjusting it lets you prioritise precision (fewer false positives)
    or recall (fewer false negatives) based on business needs.
    """
    y_proba = model.predict_proba(X_te)[:, 1]
    thresholds = np.arange(0.1, 0.91, 0.05)
    records = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        records.append({
            "threshold": t,
            "precision": precision_score(y_te, y_pred, zero_division=0),
            "recall":    recall_score(y_te, y_pred, zero_division=0),
            "f1":        f1_score(y_te, y_pred, zero_division=0),
            "accuracy":  accuracy_score(y_te, y_pred),
        })
    return pd.DataFrame(records)




def evaluate(name, model, X_te, y_te):
    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]
    return {
        "name": name,
        "acc":  accuracy_score(y_te, y_pred) * 100,
        "auc":  roc_auc_score(y_te, y_proba),
        "f1":   f1_score(y_te, y_pred),
        "cm":   confusion_matrix(y_te, y_pred),
        "fpr":  roc_curve(y_te, y_proba)[0],
        "tpr":  roc_curve(y_te, y_proba)[1],
        "model": model,
    }




def build_phase4_report(
    grid_results_df,
    lr_tuned_metrics,
    rf_tuned_metrics,
    gb_tuned_metrics,
    baseline_auc,
    lc_data,
    threshold_df,
    output=OUTPUT_PATH,
):
    fig = plt.figure(figsize=(18, 16), facecolor=BG)
    fig.suptitle(
        "Phase 4 — Hyperparameter Tuning Report\n"
        "GridSearchCV · RandomizedSearchCV · Learning Curves · Threshold Tuning",
        fontsize=15, fontweight="bold", y=0.99, color="#1a1a2e"
    )

    gs = gridspec.GridSpec(3, 3, hspace=0.50, wspace=0.38,
                           top=0.93, bottom=0.05, left=0.07, right=0.97)

    ax1 = fig.add_subplot(gs[0, 0])
    pivot = grid_results_df.pivot_table(
        index="penalty", columns="C", values="mean_auc"
    )
    sns.heatmap(pivot, ax=ax1, cmap="YlGn", annot=True, fmt=".3f",
                cbar_kws={"shrink": 0.8}, linewidths=0.5,
                annot_kws={"size": 8})
    ax1.set_title("GridSearch: LR — AUC by C & Penalty",
                  fontsize=10, fontweight="bold")
    ax1.set_xlabel("Regularisation C", fontsize=9)
    ax1.set_ylabel("Penalty", fontsize=9)

    ax2 = fig.add_subplot(gs[0, 1])
    model_names = ["LR Baseline", "LR Tuned",
                   "RF Baseline", "RF Tuned",
                   "GB Tuned"]
    auc_vals = [
        baseline_auc["lr"],
        lr_tuned_metrics["auc"],
        baseline_auc["rf"],
        rf_tuned_metrics["auc"],
        gb_tuned_metrics["auc"],
    ]
    bar_colors = [TEAL, GREEN, "#aed6f1", BLUE, PURPLE]
    bars = ax2.bar(range(len(model_names)), auc_vals,
                   color=bar_colors, edgecolor="white", width=0.6)
    ax2.axhline(0.5, color="red", lw=1, ls="--", label="Random")
    for bar, val in zip(bars, auc_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.003,
                 f"{val:.3f}", ha="center", fontsize=8)
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, rotation=25, ha="right", fontsize=8)
    ax2.set_ylim(0.4, 0.85)
    ax2.legend(fontsize=8)
    style_axis(ax2, "Baseline vs Tuned: ROC-AUC", "", "AUC")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot([0,1],[0,1],"k--",lw=0.8,label="Random (0.50)")
    for m, col, lbl in [
        (lr_tuned_metrics, GREEN,  f"LR Tuned ({lr_tuned_metrics['auc']:.3f})"),
        (rf_tuned_metrics, BLUE,   f"RF Tuned ({rf_tuned_metrics['auc']:.3f})"),
        (gb_tuned_metrics, PURPLE, f"GB Tuned ({gb_tuned_metrics['auc']:.3f})"),
    ]:
        ax3.plot(m["fpr"], m["tpr"], color=col, lw=2, label=lbl)
    ax3.legend(fontsize=7, loc="lower right")
    style_axis(ax3, "ROC Curves — Tuned Models",
               "False Positive Rate", "True Positive Rate")

    ax4 = fig.add_subplot(gs[1, 0])
    ts, tr_sc, val_sc = lc_data["lr"]
    ax4.plot(ts, tr_sc.mean(1),  color=GREEN,  lw=2, label="Train AUC")
    ax4.fill_between(ts, tr_sc.mean(1)-tr_sc.std(1),
                         tr_sc.mean(1)+tr_sc.std(1), alpha=0.15, color=GREEN)
    ax4.plot(ts, val_sc.mean(1), color=ORANGE, lw=2, label="Val AUC")
    ax4.fill_between(ts, val_sc.mean(1)-val_sc.std(1),
                         val_sc.mean(1)+val_sc.std(1), alpha=0.15, color=ORANGE)
    ax4.legend(fontsize=8)
    style_axis(ax4, "Learning Curve — LR Tuned",
               "Training Samples", "AUC")

    ax5 = fig.add_subplot(gs[1, 1])
    ts, tr_sc, val_sc = lc_data["rf"]
    ax5.plot(ts, tr_sc.mean(1),  color=BLUE,   lw=2, label="Train AUC")
    ax5.fill_between(ts, tr_sc.mean(1)-tr_sc.std(1),
                         tr_sc.mean(1)+tr_sc.std(1), alpha=0.15, color=BLUE)
    ax5.plot(ts, val_sc.mean(1), color=ORANGE, lw=2, label="Val AUC")
    ax5.fill_between(ts, val_sc.mean(1)-val_sc.std(1),
                         val_sc.mean(1)+val_sc.std(1), alpha=0.15, color=ORANGE)
    ax5.legend(fontsize=8)
    style_axis(ax5, "Learning Curve — RF Tuned",
               "Training Samples", "AUC")

    ax6 = fig.add_subplot(gs[1, 2])
    best_m = max([lr_tuned_metrics, rf_tuned_metrics, gb_tuned_metrics],
                 key=lambda m: m["auc"])
    sns.heatmap(best_m["cm"], annot=True, fmt="d", ax=ax6,
                cmap=sns.light_palette(GREEN, as_cmap=True),
                cbar=False, linewidths=1, linecolor="white",
                annot_kws={"size": 14, "weight": "bold"})
    ax6.set_xticklabels(["Low (0)", "High (1)"], fontsize=9)
    ax6.set_yticklabels(["Low (0)", "High (1)"], fontsize=9, rotation=0)
    ax6.set_xlabel("Predicted"); ax6.set_ylabel("Actual")
    ax6.set_title(f"Best Model Confusion Matrix\n{best_m['name']}",
                  fontsize=10, fontweight="bold")

    ax7 = fig.add_subplot(gs[2, :2])
    ax7.plot(threshold_df["threshold"], threshold_df["precision"],
             color=GREEN,  lw=2, marker="o", ms=4, label="Precision")
    ax7.plot(threshold_df["threshold"], threshold_df["recall"],
             color=ORANGE, lw=2, marker="s", ms=4, label="Recall")
    ax7.plot(threshold_df["threshold"], threshold_df["f1"],
             color=BLUE,   lw=2, marker="^", ms=4, label="F1 Score")
    best_t = threshold_df.loc[threshold_df["f1"].idxmax(), "threshold"]
    ax7.axvline(best_t, color="red", lw=1.5, ls="--",
                label=f"Best F1 threshold = {best_t:.2f}")
    ax7.axvline(0.5, color="gray", lw=1, ls=":", label="Default (0.50)")
    ax7.legend(fontsize=8)
    ax7.set_xlim(0.1, 0.9)
    ax7.set_ylim(0, 1.05)
    style_axis(ax7,
        "Threshold Tuning — Precision / Recall / F1 Trade-off\n"
        "Move threshold LEFT → higher recall (catch more positives)  |  "
        "Move RIGHT → higher precision (fewer false alarms)",
        "Decision Threshold", "Score")

    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis("off")
    summary_lines = [
        "── Phase 4 Key Findings ──",
        "",
        f"Baseline LR AUC :  {baseline_auc['lr']:.3f}",
        f"Tuned   LR AUC :  {lr_tuned_metrics['auc']:.3f}  "
        f"(+{lr_tuned_metrics['auc']-baseline_auc['lr']:+.3f})",
        "",
        f"Baseline RF AUC :  {baseline_auc['rf']:.3f}",
        f"Tuned   RF AUC :  {rf_tuned_metrics['auc']:.3f}  "
        f"(+{rf_tuned_metrics['auc']-baseline_auc['rf']:+.3f})",
        "",
        f"GB Tuned AUC    :  {gb_tuned_metrics['auc']:.3f}",
        "",
        f"Best F1 threshold: {best_t:.2f}",
        "",
        "── Next Steps ──",
        "",
        "→ Add SMOTE for imbalance",
        "→ Try XGBoost / LightGBM",
        "→ Feature selection (SHAP)",
        "→ Deploy with FastAPI",
    ]
    ax8.text(0.05, 0.97, "\n".join(summary_lines),
             transform=ax8.transAxes,
             fontsize=8.5, va="top", ha="left",
             fontfamily="monospace",
             bbox=dict(facecolor="#e8f5e9", edgecolor=GREEN,
                       boxstyle="round,pad=0.6", lw=1.5))

    save_fig(fig, output)
    print(f"[tuning] Report saved → '{output}'")


def run_tuning(path: str = DATA_PATH) -> None:
    print("\n" + "═" * 55)
    print("  Phase 4 — Hyperparameter Tuning")
    print("═" * 55)

    (X_tr, X_te, y_tr, y_te,
     X_tr_sc, X_te_sc, cols, scaler) = prepare(path)

    print("\n[tuning] Fitting baseline models for comparison...")
    lr_base = LogisticRegression(max_iter=1000, class_weight="balanced",
                                  random_state=42)
    lr_base.fit(X_tr_sc, y_tr)
    rf_base = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                      random_state=42)
    rf_base.fit(X_tr, y_tr)

    baseline_auc = {
        "lr": roc_auc_score(y_te, lr_base.predict_proba(X_te_sc)[:, 1]),
        "rf": roc_auc_score(y_te, rf_base.predict_proba(X_te)[:, 1]),
    }
    print(f"  Baseline LR AUC: {baseline_auc['lr']:.4f}")
    print(f"  Baseline RF AUC: {baseline_auc['rf']:.4f}")

    lr_best, lr_params, lr_cv_auc, grid_df = grid_search_lr(X_tr_sc, y_tr)

    rf_best, rf_params, rf_cv_auc = random_search_rf(X_tr, y_tr)

    gb_best, gb_cv_auc = tune_gradient_boosting(X_tr, y_tr)

    print("\n[tuning] Test-set evaluation...")
    lr_m = evaluate("LR Tuned",  lr_best, X_te_sc, y_te)
    rf_m = evaluate("RF Tuned",  rf_best, X_te,    y_te)
    gb_m = evaluate("GB Tuned",  gb_best, X_te,    y_te)

    for m in [lr_m, rf_m, gb_m]:
        print(f"  {m['name']:<15}  Acc={m['acc']:.1f}%  "
              f"AUC={m['auc']:.3f}  F1={m['f1']:.3f}")

    print("\n[tuning] Computing learning curves (this may take ~30s)...")
    lc_data = {
        "lr": compute_learning_curves(lr_best, X_tr_sc, y_tr, "LR"),
        "rf": compute_learning_curves(rf_best, X_tr,    y_tr, "RF"),
    }

    best_model_for_threshold = max([lr_m, rf_m, gb_m], key=lambda m: m["auc"])
    Xte_best = X_te_sc if "LR" in best_model_for_threshold["name"] else X_te
    thresh_df = threshold_analysis(
        best_model_for_threshold["model"], Xte_best, y_te
    )

    for m, Xte in [(lr_m, X_te_sc), (rf_m, X_te), (gb_m, X_te)]:
        print(f"\n{'─'*45}\n  {m['name']}\n{'─'*45}")
        print(classification_report(y_te, m["model"].predict(Xte)))

    build_phase4_report(
        grid_df, lr_m, rf_m, gb_m,
        baseline_auc, lc_data, thresh_df,
    )

    print("\n[tuning] Phase 4 complete ✓")
    print("Output → phase4_tuning_report.png")


if __name__ == "__main__":
    run_tuning()
