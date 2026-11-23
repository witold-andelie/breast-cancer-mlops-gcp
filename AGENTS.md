# AGENTS.md — Agentic AI Engineering Guide for `cancer_mlops`

This repository is designed for disciplined human-agent collaboration across the machine learning lifecycle. Coding agents (IDE assistants, CLI agents, CI review bots) operating in this project must follow the invariants, commands, and safety boundaries below.

## 1. Repository Map

- `configs/default.yaml` — Single source of truth for data URIs, split rules, feature caps, CHAID params, quality gate thresholds, and drift alerts.
- `sql/` — DuckDB SQL templates (`01_audit.sql` for data profiling/null/IQR checks, `02_partition.sql` for deterministic hash splitting).
- `src/cancer_mlops/config.py` — Pydantic v2 config loader with `CANCER_MLOPS_*` env var overrides.
- `src/cancer_mlops/storage.py` — Storage abstraction (`LocalStorage`, `GCSStorage` for `gs://`).
- `src/cancer_mlops/data.py` — DuckDB ingestion, SHA-256 dataset versioning, schema validation, deterministic 70/30 split.
- `src/cancer_mlops/features.py` — Leak-free sklearn transformers (`FeatureEngineer`, `CollinearityPruner`, `QuantileBinner`, `DataFrameStandardScaler`).
- `src/cancer_mlops/unsupervised.py` — Train-fitted PCA, K-Means clustering, and Isolation Forest anomaly diagnostics.
- `src/cancer_mlops/chaid.py` — Sklearn-compatible CHAID tree with open outer bin bounds and ordinal Bonferroni correction.
- `src/cancer_mlops/train.py` — 5-fold StratifiedKFold CV, hyperparameter tuning, OOF clinical threshold tuning.
- `src/cancer_mlops/evaluate.py` — Held-out evaluation, ROC/PR/Calibration plots, clinical subgroup slicing, permutation importance.
- `src/cancer_mlops/monitoring.py` — Training baseline serialization, PSI, KS tests, and online Z-score drift checks.
- `src/cancer_mlops/tracking.py` — SQLite `ExperimentTracker`, `ModelRegistry`, `QualityGate`, rollback, and `model_card.md` generator.
- `src/cancer_mlops/api.py` — FastAPI production inference service (`/health`, `/model/info`, `/predict`, `/predict/batch`).
- `cloud/gcp/` — GCP Cloud Run, Artifact Registry, Cloud Build, and Vertex AI architecture blueprint.

## 2. Canonical Commands (PowerShell / Bash)

```powershell
# Run full end-to-end MLOps pipeline
$env:PYTHONPATH = "src"; $env:PYTHONIOENCODING = "utf-8"
py -m cancer_mlops.pipeline --stage all

# Lint & automated test suite
ruff check src tests
py -m pytest -v

# Start local inference API (port 8080)
$env:PYTHONPATH = "src"
uvicorn cancer_mlops.api:app --host 0.0.0.0 --port 8080
```

## 3. Non-Negotiable ML & Engineering Invariants

1. **Zero Test-Set Leakage**: Any statistics (`CollinearityPruner`, `QuantileBinner`, `DataFrameStandardScaler`, `PCA`, `KMeans`, `IsolationForest`, `optimize_threshold`, `compute_baseline_stats`) MUST fit strictly on `train_df` or CV out-of-fold predictions inside `train_df`. Never compute quantiles, correlations, or thresholds across the combined dataset.
2. **Deterministic Reproducibility**: Partitioning must use DuckDB `hash(id_pat) % hash_mod < train_pct`, never Python's salted `hash()` or unseeded random splits.
3. **Config-Driven Thresholds**: Do not hardcode decision thresholds, feature lists, or paths inside modules; read from `configs/default.yaml` via `load_config()`.
4. **Quality Gate Enforcement**: Never promote a model to `production` if `Recall_M < min_recall_m`, `ROC-AUC < min_roc_auc`, `Brier > max_brier_score`, or predictions contain non-finite values.
5. **Test-First Changes**: Every change to feature transforms, CHAID logic, registry behavior, or API schemas must pass `ruff check src tests` and `pytest -v`.

## 4. Agent Prompt Recipes for Daily Engineering

- **Data Drift Investigation**:
  *"Read `artifacts/reports/drift_report.json` and `src/cancer_mlops/monitoring.py`. Identify which features crossed `psi_moderate` or `psi_significant`, explain clinical/scale causes, and propose whether retraining or threshold recalibration is required."*
- **PR Review / Leakage Audit**:
  *"Inspect git diff across `src/cancer_mlops/`. Verify no transformer fits on `test_df`, `QualityGate` checks remain intact, and `tests/test_no_leakage.py` covers any new feature logic."*
- **Adding a Candidate Model**:
  *"Add `<ModelName>` inside `build_candidate_pipelines()` in `src/cancer_mlops/train.py`, run `--stage all`, compare CV/Test `Recall_M`, `PR-AUC`, and `Brier`, and verify `model_card.md` updates."*

## 5. Human-in-the-Loop Approval Boundaries

Agents may autonomously run offline training, tests, linting, and static checks, **but MUST stop and ask for explicit human approval before**:
- Executing `cloud/gcp/deploy_cloud_run.sh` or modifying live GCP resources (`gcloud run deploy`, GCS buckets, IAM roles).
- Lowering `quality_gate` thresholds (`min_recall_m`, `min_roc_auc`, `max_brier_score`) in `configs/default.yaml`.
- Manually overriding `ModelRegistry.promote_to_production()` or `rollback_production()` on a live deployment database.
