"""Data drift monitoring: baselines, PSI, KS two-sample tests, and online Z-score checks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from cancer_mlops.config import MonitoringConfig


def compute_baseline_stats(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    n_bins: int = 10,
) -> dict[str, dict[str, Any]]:
    """Compute per-feature summary stats, histogram bins, and reference quantiles from train_df."""
    baseline: dict[str, dict[str, Any]] = {}
    q_grid = np.linspace(0.0, 1.0, 101)
    for col in feature_cols:
        vals = train_df[col].astype(float).dropna().to_numpy()
        if vals.size == 0:
            continue
        raw_edges = np.unique(np.round(np.quantile(vals, np.linspace(0.0, 1.0, n_bins + 1)), 6))
        if len(raw_edges) < 2:
            v0 = float(vals[0])
            raw_edges = np.array([v0 - 0.5, v0 + 0.5], dtype=float)
        cut_edges = [-np.inf, *raw_edges[1:-1], np.inf]
        counts, _ = np.histogram(vals, bins=cut_edges)
        eps = 1e-4
        expected_props = (counts + eps) / (counts.sum() + eps * len(counts))

        baseline[col] = {
            "mean": round(float(np.mean(vals)), 6),
            "std": round(float(np.std(vals, ddof=0)), 6),
            "min": round(float(np.min(vals)), 6),
            "max": round(float(np.max(vals)), 6),
            "q05": round(float(np.quantile(vals, 0.05)), 6),
            "q95": round(float(np.quantile(vals, 0.95)), 6),
            "edges": [round(float(e), 6) for e in raw_edges],
            "expected_props": [round(float(p), 8) for p in expected_props],
            "ref_quantiles": [round(float(q), 6) for q in np.quantile(vals, q_grid)],
        }
    return baseline


def compute_psi(
    expected_props: list[float],
    edges: list[float],
    actual_values: np.ndarray,
) -> float:
    """Compute Population Stability Index (PSI) for a 1D feature array."""
    if actual_values.size == 0:
        return 0.0
    cut_edges = [-np.inf, *edges[1:-1], np.inf] if len(edges) >= 2 else [-np.inf, np.inf]
    counts, _ = np.histogram(actual_values, bins=cut_edges)
    eps = 1e-4
    actual_props = (counts + eps) / (counts.sum() + eps * len(counts))
    exp_arr = np.clip(np.asarray(expected_props, dtype=float), 1e-6, None)
    exp_arr = exp_arr / exp_arr.sum()
    psi = float(np.sum((actual_props - exp_arr) * np.log(actual_props / exp_arr)))
    return round(max(psi, 0.0), 6)


def detect_batch_drift(
    batch_df: pd.DataFrame,
    baseline_stats: dict[str, dict[str, Any]],
    cfg: MonitoringConfig,
) -> dict[str, Any]:
    """Run batch PSI + Kolmogorov-Smirnov + range-check drift diagnostics across features."""
    per_feature: dict[str, dict[str, Any]] = {}
    significant: list[str] = []
    moderate: list[str] = []

    for col, stats in baseline_stats.items():
        if col not in batch_df.columns:
            continue
        actual = batch_df[col].astype(float).dropna().to_numpy()
        if actual.size == 0:
            continue
        psi = compute_psi(stats["expected_props"], stats["edges"], actual)
        ks_res = ks_2samp(np.asarray(stats["ref_quantiles"], dtype=float), actual)
        oor_pct = float(((actual < stats["min"]) | (actual > stats["max"])).mean())

        if psi >= cfg.psi_significant:
            status = "significant_drift"
            significant.append(col)
        elif psi >= cfg.psi_moderate:
            status = "moderate_drift"
            moderate.append(col)
        else:
            status = "stable"

        per_feature[col] = {
            "psi": psi,
            "ks_stat": round(float(ks_res.statistic), 4),
            "ks_pvalue": round(float(ks_res.pvalue), 4),
            "out_of_range_pct": round(oor_pct, 4),
            "status": status,
        }

    overall_status = (
        "significant_drift"
        if significant
        else ("moderate_drift" if moderate else "stable")
    )
    max_psi = max((v["psi"] for v in per_feature.values()), default=0.0)

    return {
        "n_samples": int(len(batch_df)),
        "overall_status": overall_status,
        "max_psi": round(float(max_psi), 6),
        "significant_drift_features": significant,
        "moderate_drift_features": moderate,
        "features": per_feature,
    }


def check_single_sample_drift(
    record: dict[str, float],
    baseline_stats: dict[str, dict[str, Any]],
    z_threshold: float = 3.5,
) -> list[str]:
    """Check an online inference payload against training Z-scores and range bounds."""
    warnings: list[str] = []
    for col, stats in baseline_stats.items():
        if col not in record:
            continue
        val = float(record[col])
        std = float(stats["std"])
        mean = float(stats["mean"])
        z = abs(val - mean) / std if std > 1e-9 else 0.0
        out_of_bounds = val < float(stats["min"]) or val > float(stats["max"])
        if z > z_threshold or out_of_bounds:
            warnings.append(
                f"{col}={val:.3f} (z={z:.2f}, train_range=[{stats['min']:.3f}, {stats['max']:.3f}])"
            )
    return warnings
