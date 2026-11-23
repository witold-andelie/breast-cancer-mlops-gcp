"""FastAPI endpoint integration tests using TestClient against an isolated test registry."""

from __future__ import annotations

from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from cancer_mlops.api import create_app
from cancer_mlops.config import load_config
from cancer_mlops.data import load_and_partition
from cancer_mlops.features import CollinearityPruner, DataFrameStandardScaler, FeatureEngineer
from cancer_mlops.monitoring import compute_baseline_stats
from cancer_mlops.tracking import ModelRegistry


def _setup_isolated_api_app(tmp_path):
    cfg = load_config()
    cfg.artifacts.root_dir = str(tmp_path)
    train_df, test_df, audit = load_and_partition(cfg)
    raw_cols = audit["raw_numeric_features"]

    pipe = Pipeline(
        [
            ("fe", FeatureEngineer(numeric_cols=raw_cols)),
            ("prune", CollinearityPruner()),
            ("scale", DataFrameStandardScaler()),
            ("clf", LogisticRegression(max_iter=500, random_state=42)),
        ]
    ).fit(train_df[raw_cols], train_df["y_m"])

    db_path = tmp_path / "mlops.db"
    reg_dir = tmp_path / "registry"
    registry = ModelRegistry(db_path, reg_dir)
    baseline = compute_baseline_stats(train_df, raw_cols)
    registry.register_model(
        model_name="logistic_l2",
        run_id="api_test_run",
        bundle={
            "estimator": pipe,
            "threshold": 0.42,
            "model_name": "logistic_l2",
            "raw_numeric_features": raw_cols,
        },
        schema={"raw_numeric_features": raw_cols, "data_sha256": audit["data_sha256"]},
        metrics={"recall_M": 0.98, "roc_auc": 0.99, "brier_score": 0.02, "threshold": 0.42},
        baseline_stats=baseline,
        stage="production",
    )
    app = create_app(db_path=db_path, registry_dir=reg_dir)
    return app, test_df, raw_cols


def test_api_predict_batch_and_validation(tmp_path) -> None:
    app, test_df, raw_cols = _setup_isolated_api_app(tmp_path)
    sample = {c: float(test_df.iloc[0][c]) for c in raw_cols}

    with TestClient(app) as client:
        r_health = client.get("/health")
        assert r_health.status_code == 200
        assert r_health.json()["model_loaded"] is True

        r_info = client.get("/model/info")
        assert r_info.status_code == 200
        assert r_info.json()["model_name"] == "logistic_l2"

        r_pred = client.post("/predict", json=sample)
        assert r_pred.status_code == 200
        body = r_pred.json()
        assert 0.0 <= body["probability_malignant"] <= 1.0
        assert body["prediction"] in ("M", "B")
        assert body["risk_tier"] in ("low", "medium", "high")
        assert len(body["top_contributions"]) > 0

        extreme = dict(sample)
        extreme["area_se"] = 5000.0
        r_drift = client.post("/predict", json=extreme)
        assert r_drift.status_code == 200
        assert any("area_se" in w for w in r_drift.json()["drift_warnings"])

        invalid = dict(sample)
        invalid["radius_mean"] = -1.0
        r_bad = client.post("/predict", json=invalid)
        assert r_bad.status_code == 422

        batch_records = [{c: float(test_df.iloc[i][c]) for c in raw_cols} for i in range(5)]
        r_batch = client.post("/predict/batch", json={"records": batch_records})
        assert r_batch.status_code == 200
        batch_body = r_batch.json()
        assert len(batch_body["predictions"]) == 5
        assert "overall_status" in batch_body["drift_summary"]
