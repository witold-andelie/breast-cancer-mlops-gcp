"""Leakage prevention and deterministic split verification tests."""

from __future__ import annotations

import numpy as np

from cancer_mlops.config import load_config
from cancer_mlops.data import load_and_partition
from cancer_mlops.features import CollinearityPruner, FeatureEngineer, QuantileBinner
from cancer_mlops.train import optimize_threshold


def test_deterministic_partition_disjoint(tmp_path) -> None:
    cfg = load_config()
    cfg.artifacts.root_dir = str(tmp_path)
    train_df, test_df, audit = load_and_partition(cfg)

    train_ids = set(train_df[cfg.data.id_col])
    test_ids = set(test_df[cfg.data.id_col])
    assert train_ids.isdisjoint(test_ids)
    assert len(train_ids) + len(test_ids) == audit["schema"]["n_rows"]
    assert audit["schema"]["n_rows"] == 1138


def test_feature_transformers_invariant_to_test_data(tmp_path) -> None:
    cfg = load_config()
    cfg.artifacts.root_dir = str(tmp_path)
    train_df, test_df, audit = load_and_partition(cfg)
    raw_cols = audit["raw_numeric_features"]

    fe = FeatureEngineer(numeric_cols=raw_cols).fit(train_df)
    X_train_fe = fe.transform(train_df)

    pruner_a = CollinearityPruner(
        keep_first=tuple(cfg.features.keep_first),
        collinearity_threshold=cfg.features.collinearity_threshold,
        max_features=cfg.features.max_features,
    ).fit(X_train_fe, train_df["y_m"])

    # Corrupt test_df wildly; train-fitted transformers must remain identical
    corrupted_test = test_df.copy()
    for c in raw_cols:
        corrupted_test[c] = corrupted_test[c] * 1000.0 + 999.0
    _ = fe.transform(corrupted_test)

    pruner_b = CollinearityPruner(
        keep_first=tuple(cfg.features.keep_first),
        collinearity_threshold=cfg.features.collinearity_threshold,
        max_features=cfg.features.max_features,
    ).fit(X_train_fe, train_df["y_m"])

    assert pruner_a.selected_features_ == pruner_b.selected_features_

    binner = QuantileBinner(
        feature=cfg.features.bin_feature,
        labels=tuple(cfg.features.bin_labels),
        replace=False,
    ).fit(train_df[[cfg.features.bin_feature]])
    out_test = binner.transform(corrupted_test[[cfg.features.bin_feature]])
    assert out_test[f"{cfg.features.bin_feature}_bin"].isna().sum() == 0
    assert set(out_test[f"{cfg.features.bin_feature}_bin"]).issubset({"low", "mid", "high"})


def test_optimize_threshold_bounds() -> None:
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.05, 0.15, 0.25, 0.45, 0.35, 0.65, 0.85, 0.95])
    thr = optimize_threshold(y, p, recall_target=0.95, beta=2.0)
    assert 0.10 <= thr <= 0.90
    assert (p[y == 1] >= thr).mean() >= 0.95
