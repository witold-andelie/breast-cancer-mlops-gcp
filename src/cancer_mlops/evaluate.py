"""Evaluation: metrics, ROC/PR/calibration plots, subgroup slices, and permutation importance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from cancer_mlops.features import QuantileBinner


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    proba: np.ndarray | pd.Series,
    threshold: float = 0.50,
) -> dict[str, float | int]:
    """Compute binary classification metrics treating 1 (M) as the positive class."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "recall_M": round(float(recall_score(y, pred, zero_division=0)), 4),
        "precision_M": round(float(precision_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "f2": round(float(fbeta_score(y, pred, beta=2.0, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "brier_score": round(float(brier_score_loss(y, p)), 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def _plot_roc_pr(
    y_test: np.ndarray,
    probas: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        roc_v = roc_auc_score(y_test, p)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_v:.3f})")

        prec, rec, _ = precision_recall_curve(y_test, p)
        pr_v = average_precision_score(y_test, p)
        axes[1].plot(rec, prec, label=f"{name} (AP={pr_v:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8, label="random")
    axes[0].set_title("Test ROC Curves")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate (Recall_M)")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].set_title("Test Precision-Recall Curves")
    axes[1].set_xlabel("Recall_M")
    axes[1].set_ylabel("Precision_M")
    axes[1].legend(loc="lower left", fontsize=8)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_calibration(
    y_test: np.ndarray,
    probas: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.plot([0, 1], [0, 1], "k--", lw=0.9, label="Perfectly calibrated")
    for name, p in probas.items():
        prob_true, prob_pred = calibration_curve(y_test, p, n_bins=8, strategy="quantile")
        brier = brier_score_loss(y_test, p)
        ax.plot(prob_pred, prob_true, marker="o", markersize=4, label=f"{name} (Brier={brier:.3f})")

    ax.set_title("Probability Calibration Curve (Test Partition)")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed malignancy fraction")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def evaluate_slices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    bin_feature: str = "radius_worst",
    bin_labels: tuple[str, ...] = ("low", "mid", "high"),
) -> pd.DataFrame:
    """Evaluate champion model performance across train-fitted tumor size subgroups."""
    binner = QuantileBinner(feature=bin_feature, labels=bin_labels, replace=False)
    binner.fit(train_df[[bin_feature]])
    sliced = binner.transform(test_df[[bin_feature]])
    slice_series = sliced[f"{bin_feature}_bin"]

    rows: list[dict[str, Any]] = []
    for label in binner.used_labels_:
        mask = (slice_series == label).to_numpy()
        if not mask.any():
            continue
        y_s = y_test[mask]
        p_s = proba[mask]
        pred_s = (p_s >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_s, pred_s, labels=[0, 1]).ravel()
        rows.append(
            {
                "slice": f"{bin_feature}={label}",
                "n": int(mask.sum()),
                "malignant_cnt": int(y_s.sum()),
                "accuracy": round(float(accuracy_score(y_s, pred_s)), 4),
                "recall_M": round(float(recall_score(y_s, pred_s, zero_division=0)), 4),
                "precision_M": round(float(precision_score(y_s, pred_s, zero_division=0)), 4),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            }
        )
    return pd.DataFrame(rows)


def evaluate_models_on_test(
    models: dict[str, Any],
    tuned_thresholds: dict[str, float],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    raw_numeric_features: list[str],
    reports_dir: Path,
    champion_name: str,
    bin_feature: str = "radius_worst",
    bin_labels: tuple[str, ...] = ("low", "mid", "high"),
) -> dict[str, Any]:
    """Evaluate all models on held-out test set, plots, slices, and permutation importance."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    X_test = test_df[raw_numeric_features].copy()
    y_test = test_df["y_m"].astype(int).to_numpy()

    test_rows: list[dict[str, Any]] = []
    test_probas: dict[str, np.ndarray] = {}
    tuned_metrics_by_model: dict[str, dict[str, Any]] = {}

    for name, est in models.items():
        proba = est.predict_proba(X_test)[:, 1]
        test_probas[name] = proba
        m_default = compute_metrics(y_test, proba, threshold=0.50)
        thr = tuned_thresholds[name]
        m_tuned = compute_metrics(y_test, proba, threshold=thr)
        test_rows.append({"model": name, "eval_mode": "default_0.50", **m_default})
        test_rows.append({"model": name, "eval_mode": "cv_tuned", **m_tuned})
        tuned_metrics_by_model[name] = m_tuned

    test_df_summary = pd.DataFrame(test_rows)
    test_df_summary.to_csv(reports_dir / "test_comparison.csv", index=False)

    _plot_roc_pr(y_test, test_probas, reports_dir / "roc_pr_curves.png")
    _plot_calibration(y_test, test_probas, reports_dir / "calibration_curve.png")

    champ_est = models[champion_name]
    champ_proba = test_probas[champion_name]
    champ_thr = tuned_thresholds[champion_name]

    slice_df = evaluate_slices(
        train_df=train_df,
        test_df=test_df,
        y_test=y_test,
        proba=champ_proba,
        threshold=champ_thr,
        bin_feature=bin_feature,
        bin_labels=bin_labels,
    )
    slice_df.to_csv(reports_dir / "slice_evaluation.csv", index=False)

    perm = permutation_importance(
        champ_est,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=15,
        random_state=42,
    )
    perm_df = (
        pd.DataFrame(
            {
                "feature": raw_numeric_features,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    perm_df["importance_mean"] = perm_df["importance_mean"].round(5)
    perm_df["importance_std"] = perm_df["importance_std"].round(5)
    perm_df.to_csv(reports_dir / "permutation_importance.csv", index=False)

    return {
        "test_rows": test_rows,
        "test_df_summary": test_df_summary,
        "tuned_metrics_by_model": tuned_metrics_by_model,
        "test_probas": test_probas,
        "slice_df": slice_df,
        "perm_importance_df": perm_df,
    }
