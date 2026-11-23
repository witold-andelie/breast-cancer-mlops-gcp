"""Tests for unsupervised PCA, K-Means clustering, and Isolation Forest analysis."""

from __future__ import annotations

from cancer_mlops.config import load_config
from cancer_mlops.data import load_and_partition
from cancer_mlops.unsupervised import run_unsupervised_analysis


def test_unsupervised_analysis_outputs_and_structure(tmp_path) -> None:
    cfg = load_config()
    cfg.artifacts.root_dir = str(tmp_path)
    train_df, test_df, audit = load_and_partition(cfg)

    reports_dir = tmp_path / "reports"
    summary = run_unsupervised_analysis(
        train_df=train_df,
        test_df=test_df,
        feature_cols=audit["raw_numeric_features"],
        cfg=cfg,
        reports_dir=reports_dir,
    )

    assert 1 <= summary["pca"]["n_components_90pct"] <= summary["pca"]["n_components_95pct"] < 30
    assert summary["kmeans"]["k2_posthoc_alignment"]["test_ari"] > 0.30
    assert summary["isolation_forest"]["train_anomalies"] > 0
    assert (
        summary["isolation_forest"]["train_malignant_rate_in_anomalies"]
        > summary["isolation_forest"]["train_malignant_rate_in_normal"]
    )
    assert (reports_dir / "unsupervised_pca.png").exists()
    assert (reports_dir / "unsupervised_kmeans.png").exists()
    assert (reports_dir / "unsupervised_summary.json").exists()
