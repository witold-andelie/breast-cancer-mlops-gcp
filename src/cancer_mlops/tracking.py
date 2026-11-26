"""Zero-dependency SQLite experiment tracker, model registry, quality gate, and model card."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from cancer_mlops.config import PROJECT_ROOT, QualityGateConfig

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    experiment_name TEXT NOT NULL,
    status TEXT NOT NULL,
    git_commit TEXT NOT NULL,
    code_hash TEXT NOT NULL,
    data_sha256 TEXT NOT NULL,
    config_json TEXT NOT NULL,
    notes TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    split TEXT NOT NULL,
    model_name TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS registered_models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    version INTEGER NOT NULL,
    run_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    artifact_rel_dir TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    schema_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(model_name, version)
);
"""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _get_git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def _compute_code_hash() -> str:
    h = hashlib.sha256()
    src_dir = PROJECT_ROOT / "src" / "cancer_mlops"
    if not src_dir.exists():
        return "unknown"
    for py_file in sorted(src_dir.glob("*.py")):
        h.update(py_file.name.encode("utf-8"))
        h.update(py_file.read_bytes())
    return h.hexdigest()[:12]


def _init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.executescript(_SCHEMA_SQL)


class ExperimentTracker:
    """SQLite-backed experiment, parameter, metric, and artifact tracker."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        _init_db(self.db_path)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def start_run(
        self,
        experiment_name: str,
        config_dict: dict[str, Any],
        data_sha256: str,
        notes: str = "",
    ) -> str:
        run_id = uuid.uuid4().hex[:12]
        now = _utc_now()
        with self._connect() as con:
            con.execute(
                "INSERT OR IGNORE INTO experiments (name, created_at) VALUES (?, ?)",
                (experiment_name, now),
            )
            con.execute(
                """
                INSERT INTO runs (
                    run_id, experiment_name, status, git_commit, code_hash,
                    data_sha256, config_json, notes, started_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    experiment_name,
                    "running",
                    _get_git_commit(),
                    _compute_code_hash(),
                    data_sha256,
                    json.dumps(config_dict, sort_keys=True),
                    notes,
                    now,
                ),
            )
        return run_id

    def log_metrics(
        self,
        run_id: str,
        split: str,
        model_name: str,
        metrics_dict: dict[str, Any],
    ) -> None:
        rows = []
        for k, v in metrics_dict.items():
            if isinstance(v, (int, float)) and np.isfinite(float(v)):
                rows.append((run_id, split, model_name, str(k), float(v)))
        with self._connect() as con:
            con.executemany(
                """
                INSERT INTO metrics (run_id, split, model_name, metric_name, metric_value)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )

    def log_artifact(self, run_id: str, name: str, path: Path) -> None:
        p = Path(path)
        sha = hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else ""
        with self._connect() as con:
            con.execute(
                "INSERT INTO artifacts (run_id, name, path, sha256) VALUES (?, ?, ?, ?)",
                (run_id, name, str(p), sha),
            )

    def finish_run(self, run_id: str, status: str = "finished") -> None:
        with self._connect() as con:
            con.execute(
                "UPDATE runs SET status = ?, finished_at = ? WHERE run_id = ?",
                (status, _utc_now(), run_id),
            )

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as con:
            cur = con.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?",
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]

    def get_run_metrics(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as con:
            cur = con.execute(
                "SELECT split, model_name, metric_name, metric_value FROM metrics WHERE run_id = ?",
                (run_id,),
            )
            return [dict(r) for r in cur.fetchall()]


class QualityGate:
    """Automated gate enforcing clinical recall, discrimination, calibration, and finiteness."""

    def __init__(self, cfg: QualityGateConfig) -> None:
        self.cfg = cfg

    def evaluate(
        self,
        test_metrics: dict[str, Any],
        proba: np.ndarray | None = None,
    ) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        passed = True

        recall_m = float(test_metrics["recall_M"])
        roc_auc = float(test_metrics["roc_auc"])
        brier = float(test_metrics["brier_score"])

        if recall_m >= self.cfg.min_recall_m:
            reasons.append(f"PASS recall_M={recall_m:.4f} >= {self.cfg.min_recall_m:.4f}")
        else:
            passed = False
            reasons.append(f"FAIL recall_M={recall_m:.4f} < {self.cfg.min_recall_m:.4f}")

        if roc_auc >= self.cfg.min_roc_auc:
            reasons.append(f"PASS roc_auc={roc_auc:.4f} >= {self.cfg.min_roc_auc:.4f}")
        else:
            passed = False
            reasons.append(f"FAIL roc_auc={roc_auc:.4f} < {self.cfg.min_roc_auc:.4f}")

        if brier <= self.cfg.max_brier_score:
            reasons.append(f"PASS brier_score={brier:.4f} <= {self.cfg.max_brier_score:.4f}")
        else:
            passed = False
            reasons.append(f"FAIL brier_score={brier:.4f} > {self.cfg.max_brier_score:.4f}")

        if proba is not None:
            arr = np.asarray(proba, dtype=float)
            valid = bool(np.isfinite(arr).all() and ((arr >= 0.0) & (arr <= 1.0)).all())
            if valid:
                reasons.append("PASS probabilities finite and bounded in [0, 1]")
            else:
                passed = False
                reasons.append("FAIL probabilities contain non-finite or out-of-range values")

        return passed, reasons


class ModelRegistry:
    """Versioned SQLite + filesystem model registry with promotion and rollback."""

    def __init__(self, db_path: Path, registry_dir: Path) -> None:
        self.db_path = Path(db_path)
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        _init_db(self.db_path)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def register_model(
        self,
        model_name: str,
        run_id: str,
        bundle: dict[str, Any],
        schema: dict[str, Any],
        metrics: dict[str, Any],
        baseline_stats: dict[str, Any],
        stage: str = "staging",
    ) -> int:
        with self._connect() as con:
            cur = con.execute(
                "SELECT COALESCE(MAX(version), 0) AS max_v "
                "FROM registered_models WHERE model_name = ?",
                (model_name,),
            )
            version = int(cur.fetchone()["max_v"]) + 1

        rel_dir = f"{model_name}/v{version}"
        model_dir = self.registry_dir / rel_dir
        model_dir.mkdir(parents=True, exist_ok=True)

        created_at = _utc_now()
        joblib.dump(bundle, model_dir / "model.joblib")
        metadata = {
            "model_name": model_name,
            "version": version,
            "run_id": run_id,
            "stage": stage,
            "created_at": created_at,
            "schema": schema,
            "metrics": metrics,
            "baseline_stats": baseline_stats,
        }
        (model_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        with self._connect() as con:
            if stage == "production":
                con.execute(
                    "UPDATE registered_models SET stage = 'archived' WHERE stage = 'production'"
                )
            con.execute(
                """
                INSERT INTO registered_models (
                    model_name, version, run_id, stage, artifact_rel_dir,
                    metrics_json, schema_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_name,
                    version,
                    run_id,
                    stage,
                    rel_dir,
                    json.dumps(metrics),
                    json.dumps(schema),
                    created_at,
                ),
            )
        return version

    def _sync_metadata_stages(self) -> None:
        with self._connect() as con:
            rows = con.execute(
                "SELECT artifact_rel_dir, stage FROM registered_models"
            ).fetchall()
        for r in rows:
            meta_path = self.registry_dir / r["artifact_rel_dir"] / "metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("stage") != r["stage"]:
                    meta["stage"] = r["stage"]
                    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def promote_to_production(self, model_name: str, version: int) -> None:
        with self._connect() as con:
            target = con.execute(
                "SELECT id FROM registered_models WHERE model_name = ? AND version = ?",
                (model_name, version),
            ).fetchone()
            if target is None:
                raise ValueError(f"Model {model_name} v{version} not found in registry.")
            con.execute(
                "UPDATE registered_models SET stage = 'archived' WHERE stage = 'production'"
            )
            con.execute(
                "UPDATE registered_models SET stage = 'production' "
                "WHERE model_name = ? AND version = ?",
                (model_name, version),
            )
        self._sync_metadata_stages()

    def rollback_production(
        self,
        model_name: str,
        target_version: int | None = None,
    ) -> int:
        with self._connect() as con:
            if target_version is None:
                row = con.execute(
                    """
                    SELECT version FROM registered_models
                    WHERE model_name = ? AND stage = 'archived'
                    ORDER BY version DESC LIMIT 1
                    """,
                    (model_name,),
                ).fetchone()
                if row is None:
                    raise ValueError(f"No archived version available to rollback for {model_name}.")
                target_version = int(row["version"])
        self.promote_to_production(model_name, target_version)
        return target_version

    def list_models(self) -> list[dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT model_name, version, run_id, stage, artifact_rel_dir, created_at "
                "FROM registered_models ORDER BY id DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def load_production(self, model_name: str | None = None) -> dict[str, Any]:
        with self._connect() as con:
            if model_name is not None:
                row = con.execute(
                    """
                    SELECT * FROM registered_models
                    WHERE model_name = ? AND stage = 'production'
                    ORDER BY id DESC LIMIT 1
                    """,
                    (model_name,),
                ).fetchone()
            else:
                row = con.execute(
                    """
                    SELECT * FROM registered_models
                    WHERE stage = 'production'
                    ORDER BY id DESC LIMIT 1
                    """
                ).fetchone()
        if row is None:
            raise FileNotFoundError("No production model found in registry.")

        model_dir = self.registry_dir / row["artifact_rel_dir"]
        bundle = joblib.load(model_dir / "model.joblib")
        metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        return {"bundle": bundle, "metadata": metadata}


def _df_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def generate_model_card(
    out_path: Path,
    production_meta: dict[str, Any],
    cv_df: pd.DataFrame,
    test_df: pd.DataFrame,
    slice_df: pd.DataFrame,
    unsupervised_summary: dict[str, Any],
    audit_summary: dict[str, Any],
    gate_reasons: list[str],
) -> Path:
    """Write a structured Markdown Model Card covering lineage, metrics, and governance."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tuned_cv = cv_df[cv_df["eval_mode"] == "cv_tuned"]
    tuned_test = test_df[test_df["eval_mode"] == "cv_tuned"]
    m = production_meta["metrics"]
    schema = production_meta["schema"]
    pca_info = unsupervised_summary["pca"]
    km_info = unsupervised_summary["kmeans"]
    iso_info = unsupervised_summary["isolation_forest"]
    unique_vecs = audit_summary["schema"].get(
        "unique_feature_vectors", audit_summary["schema"]["n_rows"]
    )
    raw_rows = audit_summary.get("raw_rows", audit_summary["schema"]["n_rows"])
    dedup_removed = audit_summary.get("duplicates_removed", 0)
    dedup_enabled = audit_summary.get("deduplicate_features", False)

    gate_bullets = "\n".join(f"- {r}" for r in gate_reasons)
    lines = [
        f"# Model Card: `{production_meta['model_name']}` (v{production_meta['version']})",
        "",
        "## 1. Model Overview & Lineage",
        f"- **Stage**: `{production_meta['stage']}`",
        f"- **Run ID**: `{production_meta['run_id']}`",
        f"- **Created At (UTC)**: `{production_meta['created_at']}`",
        f"- **Dataset SHA-256**: `{schema.get('data_sha256', '')}`",
        f"- **Deduplication**: `deduplicate_features={dedup_enabled}` "
        f"(raw={raw_rows} rows -> clean={audit_summary['schema']['n_rows']} unique patients, "
        f"removed={dedup_removed} duplicate twins)",
        f"- **Partition**: Train = {audit_summary['train_rows']} rows, "
        f"Test = {audit_summary['test_rows']} rows (deterministic DuckDB hash split)",
        f"- **Clinical Decision Threshold**: `{m['threshold']}` "
        "(tuned on CV out-of-fold probabilities to prioritize malignant Recall_M)",
        "",
        "## 2. Quality Gate Verification",
        gate_bullets,
        "",
        "## 3. Held-Out Test Performance (`cv_tuned` thresholds)",
        _df_to_markdown(tuned_test),
        "",
        "## 4. 5-Fold Cross-Validation Summary (`cv_tuned` thresholds)",
        _df_to_markdown(tuned_cv),
        "",
        "## 5. Clinical Subgroup Slice Evaluation (`radius_worst` tertiles)",
        _df_to_markdown(slice_df),
        "",
        "## 6. Unsupervised Structure & Anomaly Insights (Train-Fitted)",
        f"- **PCA**: {pca_info['n_components_90pct']} components explain 90% variance; "
        f"{pca_info['n_components_95pct']} components explain 95% variance.",
        f"- **K-Means (k=2, unlabeled)**: "
        f"Train ARI={km_info['k2_posthoc_alignment']['train_ari']}, "
        f"Test ARI={km_info['k2_posthoc_alignment']['test_ari']}; "
        f"Best silhouette k={km_info['best_k_by_silhouette']} ({km_info['best_silhouette']}).",
        f"- **Isolation Forest ({iso_info['contamination']:.0%} contamination)**: "
        "Malignancy rate inside flagged train anomalies is "
        f"{iso_info['train_malignant_rate_in_anomalies']:.1%} "
        f"vs {iso_info['train_malignant_rate_in_normal']:.1%} in normal samples.",
        "",
        "## 7. Intended Use, Limitations & Ethical Notes",
        "- **Intended Use**: Decision-support triage prototype for breast mass malignancy scoring; "
        "positive (`M`) misses carry high clinical cost, so recall is prioritized over accuracy.",
        f"- **Data Hygiene & Leakage Guard**: Raw CSV contains {raw_rows} rows across "
        f"{unique_vecs} unique feature vectors (`A`/`B` paired ID prefixes). "
        "Pre-split SQL deduplication removes exact duplicate twins before partitioning so zero "
        "patient feature signatures leak between train and test.",
        "- **Limitations**: Trained on static diagnostic measurements; production use requires "
        "continuous PSI/KS drift checks and clinician oversight.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
