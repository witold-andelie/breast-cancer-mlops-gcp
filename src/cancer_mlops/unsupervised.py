"""Unsupervised learning module: PCA, K-Means clustering, and Isolation Forest anomaly analysis."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from cancer_mlops.config import AppConfig


def run_unsupervised_analysis(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    cfg: AppConfig,
    reports_dir: Path,
) -> dict:
    """Run leak-free PCA, K-Means, and Isolation Forest fitted strictly on train_df."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg.unsupervised.random_state

    X_train = train_df[feature_cols].astype(float).to_numpy()
    X_test = test_df[feature_cols].astype(float).to_numpy()
    y_train = train_df["y_m"].astype(int).to_numpy()
    y_test = test_df["y_m"].astype(int).to_numpy()

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 1. PCA dimensionality & separability analysis
    pca = PCA(random_state=seed)
    pca.fit(X_train_s)
    X_test_pca = pca.transform(X_test_s)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_90 = int(np.searchsorted(cum_var, 0.90) + 1)
    n_95 = int(np.searchsorted(cum_var, cfg.unsupervised.pca_variance_threshold) + 1)

    pc1_top = sorted(
        zip(feature_cols, pca.components_[0], strict=True),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )[:5]
    pc2_top = sorted(
        zip(feature_cols, pca.components_[1], strict=True),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )[:5]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].plot(range(1, len(cum_var) + 1), cum_var, marker="o", markersize=3)
    axes[0].axhline(0.95, color="tab:red", linestyle="--", linewidth=1, label="95% threshold")
    axes[0].axvline(n_95, color="tab:green", linestyle=":", linewidth=1.2, label=f"k={n_95}")
    axes[0].set_title("PCA Cumulative Explained Variance (Train)")
    axes[0].set_xlabel("Number of components")
    axes[0].set_ylabel("Cumulative variance ratio")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    for label, name, color in ((0, "B (benign)", "tab:blue"), (1, "M (malignant)", "tab:red")):
        mask = y_test == label
        axes[1].scatter(
            X_test_pca[mask, 0],
            X_test_pca[mask, 1],
            alpha=0.65,
            s=22,
            label=name,
            color=color,
        )
    axes[1].set_title(
        f"Test Projection onto PC1/PC2 (var={cum_var[1]:.1%})"
    )
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(reports_dir / "unsupervised_pca.png", dpi=120)
    plt.close(fig)

    # 2. K-Means clustering (strictly unlabeled fit; post-hoc ARI/NMI evaluation)
    k_sweep: list[dict] = []
    km2_model = None
    for k in range(cfg.unsupervised.kmeans_k_min, cfg.unsupervised.kmeans_k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels_tr = km.fit_predict(X_train_s)
        sil = float(silhouette_score(X_train_s, labels_tr))
        k_sweep.append({"k": k, "inertia": float(km.inertia_), "silhouette": round(sil, 4)})
        if k == 2:
            km2_model = km

    best_sil = max(k_sweep, key=lambda r: r["silhouette"])
    assert km2_model is not None
    train_k2 = km2_model.predict(X_train_s)
    test_k2 = km2_model.predict(X_test_s)
    ari_train = float(adjusted_rand_score(y_train, train_k2))
    nmi_train = float(normalized_mutual_info_score(y_train, train_k2))
    ari_test = float(adjusted_rand_score(y_test, test_k2))
    nmi_test = float(normalized_mutual_info_score(y_test, test_k2))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ks = [r["k"] for r in k_sweep]
    axes[0].plot(ks, [r["inertia"] for r in k_sweep], marker="o")
    axes[0].set_title("K-Means Elbow Curve (Train)")
    axes[0].set_xlabel("Number of clusters k")
    axes[0].set_ylabel("Inertia")
    axes[0].grid(alpha=0.25)

    axes[1].plot(ks, [r["silhouette"] for r in k_sweep], marker="s", color="tab:orange")
    axes[1].set_title("K-Means Silhouette Score (Train)")
    axes[1].set_xlabel("Number of clusters k")
    axes[1].set_ylabel("Silhouette score")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(reports_dir / "unsupervised_kmeans.png", dpi=120)
    plt.close(fig)

    # 3. Isolation Forest anomaly detection
    iso = IsolationForest(
        contamination=cfg.unsupervised.isolation_contamination,
        random_state=seed,
    )
    iso.fit(X_train_s)
    iso_tr = iso.predict(X_train_s) == -1
    iso_te = iso.predict(X_test_s) == -1

    # Overlap with classical 1.5 * IQR rule on area_se (fit bounds on train)
    q1 = float(train_df["area_se"].quantile(0.25))
    q3 = float(train_df["area_se"].quantile(0.75))
    iqr = q3 - q1
    iqr_tr = (train_df["area_se"] < q1 - 1.5 * iqr) | (train_df["area_se"] > q3 + 1.5 * iqr)
    overlap_cnt = int((iso_tr & iqr_tr.to_numpy()).sum())

    summary = {
        "pca": {
            "n_features": len(feature_cols),
            "n_components_90pct": n_90,
            "n_components_95pct": n_95,
            "pc1_var_ratio": round(float(pca.explained_variance_ratio_[0]), 4),
            "pc2_var_ratio": round(float(pca.explained_variance_ratio_[1]), 4),
            "pc1_top_loadings": [{"feature": f, "loading": round(float(v), 4)} for f, v in pc1_top],
            "pc2_top_loadings": [{"feature": f, "loading": round(float(v), 4)} for f, v in pc2_top],
        },
        "kmeans": {
            "k_sweep": k_sweep,
            "best_k_by_silhouette": best_sil["k"],
            "best_silhouette": best_sil["silhouette"],
            "k2_posthoc_alignment": {
                "train_ari": round(ari_train, 4),
                "train_nmi": round(nmi_train, 4),
                "test_ari": round(ari_test, 4),
                "test_nmi": round(nmi_test, 4),
            },
        },
        "isolation_forest": {
            "contamination": cfg.unsupervised.isolation_contamination,
            "train_anomalies": int(iso_tr.sum()),
            "train_anomaly_rate": round(float(iso_tr.mean()), 4),
            "test_anomalies": int(iso_te.sum()),
            "test_anomaly_rate": round(float(iso_te.mean()), 4),
            "train_malignant_rate_in_anomalies": round(float(y_train[iso_tr].mean()), 4)
            if iso_tr.any()
            else 0.0,
            "train_malignant_rate_in_normal": round(float(y_train[~iso_tr].mean()), 4),
            "test_malignant_rate_in_anomalies": round(float(y_test[iso_te].mean()), 4)
            if iso_te.any()
            else 0.0,
            "train_iqr_area_se_outliers": int(iqr_tr.sum()),
            "overlap_with_area_se_iqr": overlap_cnt,
        },
    }
    (reports_dir / "unsupervised_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
