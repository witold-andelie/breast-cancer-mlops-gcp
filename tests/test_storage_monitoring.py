"""Tests for storage backends (Local & mocked GCS) and PSI/KS/Z-score drift monitoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cancer_mlops.config import MonitoringConfig
from cancer_mlops.monitoring import (
    check_single_sample_drift,
    compute_baseline_stats,
    detect_batch_drift,
)
from cancer_mlops.storage import GCSStorage, LocalStorage, get_storage, parse_gcs_uri


class _FakeBlob:
    def __init__(self, store: dict[tuple[str, str], bytes], bucket: str, name: str) -> None:
        self.store = store
        self.key = (bucket, name)

    def download_as_bytes(self) -> bytes:
        return self.store[self.key]

    def upload_from_string(self, data: bytes) -> None:
        self.store[self.key] = data

    def exists(self) -> bool:
        return self.key in self.store


class _FakeBucket:
    def __init__(self, store: dict[tuple[str, str], bytes], name: str) -> None:
        self.store = store
        self.name = name

    def blob(self, blob_name: str) -> _FakeBlob:
        return _FakeBlob(self.store, self.name, blob_name)


class _FakeGCSClient:
    def __init__(self) -> None:
        self.store: dict[tuple[str, str], bytes] = {}

    def bucket(self, name: str) -> _FakeBucket:
        return _FakeBucket(self.store, name)


def test_local_and_gcs_storage_roundtrip(tmp_path) -> None:
    local = LocalStorage()
    f = tmp_path / "sample.txt"
    local.write_bytes(str(f), b"hello-local")
    assert local.exists(str(f))
    assert local.read_bytes(str(f)) == b"hello-local"
    assert local.resolve_local_file(f.as_uri(), tmp_path) == f.resolve()

    assert parse_gcs_uri("gs://my-bucket/data/file.csv") == ("my-bucket", "data/file.csv")
    with pytest.raises(ValueError):
        parse_gcs_uri("gs://only-bucket")

    fake_client = _FakeGCSClient()
    gcs = get_storage("gs://my-bucket/data/file.csv", gcs_client=fake_client)
    assert isinstance(gcs, GCSStorage)
    gcs.write_bytes("gs://my-bucket/data/file.csv", b"col1,col2\n1,2\n")
    assert gcs.exists("gs://my-bucket/data/file.csv")
    cached = gcs.resolve_local_file("gs://my-bucket/data/file.csv", tmp_path / "cache")
    assert cached.read_bytes() == b"col1,col2\n1,2\n"


def test_drift_detection_stable_vs_shifted() -> None:
    rng = np.random.default_rng(7)
    train_df = pd.DataFrame(
        {
            "radius_worst": rng.normal(15.0, 2.0, size=500),
            "area_se": rng.normal(40.0, 10.0, size=500),
        }
    )
    baseline = compute_baseline_stats(train_df, ["radius_worst", "area_se"], n_bins=10)
    cfg = MonitoringConfig(psi_moderate=0.10, psi_significant=0.25, z_score_alert=3.5)

    same_report = detect_batch_drift(train_df, baseline, cfg)
    assert same_report["overall_status"] == "stable"
    assert same_report["max_psi"] < 0.05

    shifted = train_df.copy()
    shifted["radius_worst"] = shifted["radius_worst"] + 8.0
    drift_report = detect_batch_drift(shifted, baseline, cfg)
    assert drift_report["overall_status"] == "significant_drift"
    assert "radius_worst" in drift_report["significant_drift_features"]

    warnings = check_single_sample_drift(
        {"radius_worst": 50.0, "area_se": 40.0},
        baseline,
        z_threshold=3.5,
    )
    assert len(warnings) == 1
    assert "radius_worst" in warnings[0]
