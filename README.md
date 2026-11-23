# Cancer MLOps — End-to-End Breast Cancer Risk Scoring System (GCP-Ready)

A production-oriented machine learning engineering project built on the breast cancer diagnostic dataset ([`data/Cancer_Data.csv`](data/Cancer_Data.csv)). It implements a leak-free, observable, containerized MLOps lifecycle designed for Google Cloud Platform (Cloud Run, Cloud Storage, Artifact Registry, Cloud Build, and Vertex AI).

---

## 1. Architecture Overview

```mermaid
flowchart LR
    CSV["Data Source\n(Local / gs:// GCS)"] --> Data["DuckDB Audit & Hash Split\n(sql/01_audit.sql, 02_partition.sql)"]
    Data -->|805 train / 333 test| Unsup["Unsupervised Diagnostics\n(PCA, K-Means, Isolation Forest)"]
    Data --> Train["5-Fold Stratified CV\n& OOF Threshold Tuning"]
    Train --> Eval["Held-Out Evaluation\n(Slices, Calibration, Perm Imp)"]
    Eval --> Gate{"Quality Gate\nRecall_M >= 0.95\nAUC >= 0.98\nBrier <= 0.06"}
    Gate -->|PASS| Reg["SQLite Model Registry\n(Versioned + Rollback)"]
    Gate -->|FAIL| Stage["Staging Only"]
    Reg --> Card["Model Card & Drift Baseline\n(model_card.md, PSI/KS)"]
    Reg --> API["FastAPI Inference Service\n(/health, /predict, /predict/batch)"]
    API --> CR["Docker + Cloud Run\n(cloud/gcp/)"]
```

---

## 2. Engineering Competency Alignment

| Domain | Implemented Capabilities | Key Modules |
|---|---|---|
| **Data Prep & SQL** | DuckDB SQL audit (`UNPIVOT`, skewness, 1.5x IQR outliers), schema validation, SHA-256 data lineage, deterministic hash partition (`train=805`, `test=333`) | `sql/01_audit.sql`, `sql/02_partition.sql`, `src/cancer_mlops/data.py` |
| **Leak-Free Feature Pipelines** | Sklearn `Pipeline` transformers fitted strictly inside training folds (`FeatureEngineer`, `CollinearityPruner`, `QuantileBinner`, `DataFrameStandardScaler`) | `src/cancer_mlops/features.py` |
| **Unsupervised Learning** | Train-fitted PCA (7 comps = 90%, 10 comps = 95%), unlabeled K-Means (`k=2..8`, Test ARI = 0.6507), Isolation Forest anomaly detection | `src/cancer_mlops/unsupervised.py` |
| **Supervised & Custom Modeling** | L2 Logistic Regression, Statsmodels Logit (OR & VIF tables), Random Forest, HistGradientBoosting, plus custom CHAID tree with ordinal Bonferroni adjustment | `src/cancer_mlops/train.py`, `src/cancer_mlops/chaid.py` |
| **Clinical Evaluation & Thresholds** | CV out-of-fold threshold tuning targeting `Recall_M >= 0.96` (`F2` score), ROC/PR/calibration curves, subgroup slicing, permutation importance | `src/cancer_mlops/evaluate.py` |
| **MLOps Governance & Monitoring** | Zero-dependency SQLite `ExperimentTracker`, `ModelRegistry` with promote/rollback, `QualityGate`, `model_card.md`, PSI + KS batch drift + online Z-score alerts | `src/cancer_mlops/tracking.py`, `src/cancer_mlops/monitoring.py` |
| **Cloud & Serving (GCP)** | `StorageBackend` (`LocalStorage` + `GCSStorage` for `gs://`), FastAPI + Pydantic v2, non-root `Dockerfile`, CI workflow, Cloud Build & Cloud Run deploy scripts | `src/cancer_mlops/storage.py`, `src/cancer_mlops/api.py`, `cloud/gcp/` |
| **Agentic AI Workflow** | Repository invariants, prompt recipes, and human approval boundaries for coding/MLOps agents | `AGENTS.md` |

---

## 3. Verified Pipeline Results

All numbers below come directly from `py -m cancer_mlops.pipeline --stage all` on `Cancer_Data.csv` (1,138 rows, SHA-256 prefix `aa47becd63cb`).

### 3.1 Held-Out Test Performance (`cv_tuned` Clinical Thresholds)

| Model | Tuned Threshold | Accuracy | Recall_M | Precision_M | F1 | F2 | ROC-AUC | PR-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hist_gbdt` (Champion) | 0.21 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0012 |
| `random_forest` | 0.46 | 0.9850 | 0.9925 | 0.9706 | 0.9814 | 0.9880 | 0.9987 | 0.9983 | 0.0184 |
| `logistic_l2` | 0.29 | 0.9670 | 0.9774 | 0.9420 | 0.9594 | 0.9701 | 0.9971 | 0.9958 | 0.0217 |
| `logistic_statsmodels` | 0.16 | 0.9580 | 0.9850 | 0.9161 | 0.9493 | 0.9704 | 0.9971 | 0.9958 | 0.0221 |
| `chaid_bonferroni` | 0.10 | 0.8709 | 1.0000 | 0.7557 | 0.8608 | 0.9393 | 0.9760 | 0.9447 | 0.0549 |
| `chaid_unadjusted` | 0.10 | 0.8018 | 1.0000 | 0.6683 | 0.8012 | 0.9097 | 0.9760 | 0.9447 | 0.0557 |

### 3.2 5-Fold Cross-Validation Summary (`cv_tuned`)

- `hist_gbdt`: Accuracy `0.9876`, Recall_M `0.9897`, ROC-AUC `0.9984`, Brier `0.0084`
- `random_forest`: Accuracy `0.9752`, Recall_M `0.9759`, ROC-AUC `0.9947`, Brier `0.0245`
- `logistic_l2`: Accuracy `0.9540`, Recall_M `0.9725`, ROC-AUC `0.9939`, Brier `0.0300`

### 3.3 Unsupervised & Monitoring Diagnostics

- **PCA**: 7 components explain 90% variance; 10 components explain 95% variance (PC1 = 45.09%, PC2 = 17.85%).
- **K-Means (`k=2`, unlabeled)**: Best silhouette `0.3488` at `k=2`; post-hoc alignment yields Train ARI = `0.6627`, Test ARI = `0.6507`.
- **Isolation Forest (5% contamination)**: Flagged train anomalies have `75.6%` malignancy rate vs `34.0%` in normal samples (21/41 anomalies overlap with 1.5x IQR `area_se` outliers).
- **Batch Drift Self-Check (`test` vs `train` baseline)**: `overall_status = "stable"`, `max_psi = 0.078969` (`smoothness_worst`), zero features in moderate/significant drift.

### 3.4 Data Hygiene & Modeling Notes

1. **Duplicate Feature Signatures Across IDs**: Schema auditing confirmed `Cancer_Data.csv` contains 1,138 rows (`id_pat` primary keys) spanning **569 unique feature vectors**. Partitioning by `id_pat` reproduces the course benchmark; however, paired IDs across partitions allow high-capacity ensembles (`hist_gbdt`) to achieve `1.000` test AUC. Real clinical pipelines should deduplicate or group-split by underlying sample signature.
2. **Bonferroni Pruning Effect in CHAID**: Enabling ordinal Bonferroni correction (`chaid_bonferroni`) pruned marginal splits (`area_ratio`, `compactness_mean`) present in `chaid_unadjusted`, raising tuned test accuracy from `0.8018` to `0.8709` and precision from `0.6683` to `0.7557` at `1.000` recall.

---

## 4. Quickstart

### 4.1 Run End-to-End Pipeline, Lint, and Tests (PowerShell)

```powershell
$env:PYTHONPATH = "src"
$env:PYTHONIOENCODING = "utf-8"

# Full lifecycle: audit -> unsupervised -> CV train -> test eval -> quality gate -> registry -> drift
py -m cancer_mlops.pipeline --stage all

# Static lint & unit/integration tests
ruff check src tests
py -m pytest -v
```

### 4.2 Run Local FastAPI Inference Server

```powershell
$env:PYTHONPATH = "src"
uvicorn cancer_mlops.api:app --host 0.0.0.0 --port 8080
```

Endpoints:
- `GET /health` — liveness & loaded model version
- `GET /model/info` — active production lineage, threshold, schema, and metrics
- `POST /predict` — single tumor record scoring with risk tier (`low`/`medium`/`high`), top feature contributions, and online Z-score/range drift warnings
- `POST /predict/batch` — batch inference with batch PSI/KS drift summary

### 4.3 Docker & GCP Cloud Run Deployment

```bash
# Local Docker build & run (after running --stage all so artifacts/ is populated)
docker build -t cancer-mlops-api:latest .
docker run --rm -p 8080:8080 cancer-mlops-api:latest

# Deploy to GCP Artifact Registry + Cloud Run
PROJECT_ID=<your-gcp-project> REGION=europe-west3 ./cloud/gcp/deploy_cloud_run.sh
```

See [`cloud/gcp/ARCHITECTURE.md`](cloud/gcp/ARCHITECTURE.md) for the full GCP service mapping and Vertex AI migration path.

---

## 5. Project Structure

```text
cancer_mlops/
├── .github/workflows/ci.yml        # GitHub Actions CI (ruff + pytest + pipeline smoke)
├── cloud/gcp/
│   ├── ARCHITECTURE.md             # GCP Cloud Run / GCS / Vertex AI blueprint
│   ├── cloudbuild.yaml             # Cloud Build CI/CD pipeline
│   └── deploy_cloud_run.sh         # Artifact Registry + Cloud Run deploy script
├── configs/default.yaml            # Central configuration
├── data/Cancer_Data.csv            # Diagnostic dataset (1,138 records x 32 cols)
├── sql/
│   ├── 01_audit.sql                # DuckDB profiling, missing, and IQR checks
│   └── 02_partition.sql            # Deterministic hash split view
├── src/cancer_mlops/
│   ├── api.py                      # FastAPI inference service
│   ├── chaid.py                    # Hardened CHAID with Bonferroni adjustment
│   ├── config.py                   # Pydantic v2 config & env overrides
│   ├── data.py                     # DuckDB loader, SHA-256 lineage, schema validator
│   ├── evaluate.py                 # Metrics, ROC/PR/calibration, slices, perm importance
│   ├── features.py                 # Leak-free sklearn feature transformers
│   ├── monitoring.py               # PSI, KS, and online Z-score drift detection
│   ├── pipeline.py                 # CLI orchestrator
│   ├── storage.py                  # Local & GCS (gs://) storage abstraction
│   ├── tracking.py                 # SQLite tracker, registry, quality gate, model card
│   ├── train.py                    # StratifiedKFold CV, tuning, threshold optimization
│   └── unsupervised.py             # PCA, K-Means, Isolation Forest analysis
├── tests/                          # Automated pytest suite (12 tests)
├── AGENTS.md                       # Agentic AI engineering workflow & boundaries
├── Dockerfile                      # Non-root Cloud Run container
├── pyproject.toml                  # Package, ruff, and pytest settings
└── requirements.txt                # Runtime & dev dependencies
```
