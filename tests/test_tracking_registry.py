"""Tests for SQLite ExperimentTracker, QualityGate, ModelRegistry, and rollback."""

from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier

from cancer_mlops.config import QualityGateConfig
from cancer_mlops.tracking import ExperimentTracker, ModelRegistry, QualityGate


def test_experiment_tracker_and_quality_gate(tmp_path) -> None:
    db = tmp_path / "mlops.db"
    tracker = ExperimentTracker(db)
    run_id = tracker.start_run("unit_test_exp", {"k": 1}, data_sha256="abc123", notes="test")
    tracker.log_metrics(run_id, "test", "dummy", {"recall_M": 0.97, "roc_auc": 0.99})
    art = tmp_path / "dummy.txt"
    art.write_text("ok", encoding="utf-8")
    tracker.log_artifact(run_id, "dummy", art)
    tracker.finish_run(run_id)

    runs = tracker.list_runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == run_id
    assert runs[0]["status"] == "finished"
    assert len(tracker.get_run_metrics(run_id)) == 2

    gate = QualityGate(QualityGateConfig(min_recall_m=0.95, min_roc_auc=0.98, max_brier_score=0.06))
    ok, _ = gate.evaluate(
        {"recall_M": 0.97, "roc_auc": 0.99, "brier_score": 0.03},
        np.array([0.1, 0.9]),
    )
    assert ok

    bad, reasons = gate.evaluate(
        {"recall_M": 0.80, "roc_auc": 0.99, "brier_score": 0.03},
        np.array([0.1, np.nan]),
    )
    assert not bad
    assert any("FAIL recall_M" in r for r in reasons)
    assert any("non-finite" in r for r in reasons)


def test_registry_promotion_and_rollback(tmp_path) -> None:
    db = tmp_path / "mlops.db"
    reg_dir = tmp_path / "registry"
    registry = ModelRegistry(db, reg_dir)

    m1 = DummyClassifier(strategy="most_frequent").fit([[0], [1]], [0, 1])
    v1 = registry.register_model(
        "logistic_l2",
        "run1",
        {"estimator": m1, "threshold": 0.4},
        {"raw_numeric_features": ["a"]},
        {"recall_M": 0.96},
        {"a": {"mean": 0.0, "std": 1.0, "min": 0.0, "max": 1.0}},
        stage="production",
    )
    v2 = registry.register_model(
        "logistic_l2",
        "run2",
        {"estimator": m1, "threshold": 0.45},
        {"raw_numeric_features": ["a"]},
        {"recall_M": 0.98},
        {"a": {"mean": 0.0, "std": 1.0, "min": 0.0, "max": 1.0}},
        stage="staging",
    )
    assert (v1, v2) == (1, 2)

    registry.promote_to_production("logistic_l2", v2)
    prod = registry.load_production("logistic_l2")
    assert prod["metadata"]["version"] == 2
    assert prod["metadata"]["stage"] == "production"

    restored_v = registry.rollback_production("logistic_l2")
    assert restored_v == 1
    prod_after = registry.load_production("logistic_l2")
    assert prod_after["metadata"]["version"] == 1
    assert prod_after["metadata"]["stage"] == "production"
