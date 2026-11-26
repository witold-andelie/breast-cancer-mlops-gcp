"""YAML-driven Pydantic configuration with environment variable overrides."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


class DataConfig(BaseModel):
    uri: str = "data/Cancer_Data.csv"
    id_col: str = "id_pat"
    target_col: str = "diagnosis"
    positive_label: str = "M"
    deduplicate_features: bool = True
    train_pct: int = Field(default=70, ge=10, le=90)
    hash_mod: int = Field(default=100, ge=10)


class FeatureConfig(BaseModel):
    keep_first: list[str] = Field(default_factory=lambda: ["radius_worst"])
    collinearity_threshold: float = Field(default=0.85, gt=0.0, le=1.0)
    max_features: int = Field(default=6, ge=1)
    bin_feature: str = "radius_worst"
    bin_labels: list[str] = Field(default_factory=lambda: ["low", "mid", "high"])


class UnsupervisedConfig(BaseModel):
    pca_variance_threshold: float = Field(default=0.95, gt=0.0, le=1.0)
    kmeans_k_min: int = Field(default=2, ge=2)
    kmeans_k_max: int = Field(default=8, ge=2)
    isolation_contamination: float = Field(default=0.05, gt=0.0, lt=0.5)
    random_state: int = 42


class ChaidConfig(BaseModel):
    n_bins: int = Field(default=10, ge=2)
    alpha_merge: float = Field(default=0.05, gt=0.0, lt=1.0)
    alpha_split: float = Field(default=0.05, gt=0.0, lt=1.0)
    max_depth: int = Field(default=4, ge=1)
    min_child: int = Field(default=40, ge=5)
    bonferroni: bool = True


class TrainingConfig(BaseModel):
    cv_folds: int = Field(default=5, ge=2)
    random_state: int = 42
    recall_target: float = Field(default=0.96, gt=0.0, le=1.0)
    f_beta: float = Field(default=2.0, gt=0.0)
    chaid: ChaidConfig = Field(default_factory=ChaidConfig)


class QualityGateConfig(BaseModel):
    min_recall_m: float = Field(default=0.95, ge=0.0, le=1.0)
    min_roc_auc: float = Field(default=0.98, ge=0.0, le=1.0)
    max_brier_score: float = Field(default=0.06, gt=0.0, le=1.0)


class MonitoringConfig(BaseModel):
    psi_moderate: float = Field(default=0.10, gt=0.0)
    psi_significant: float = Field(default=0.25, gt=0.0)
    z_score_alert: float = Field(default=3.5, gt=0.0)
    n_bins: int = Field(default=10, ge=2)


class ArtifactsConfig(BaseModel):
    root_dir: str = "artifacts"
    db_file: str = "artifacts/mlops.db"
    registry_dir: str = "artifacts/registry"
    reports_dir: str = "artifacts/reports"


class AppConfig(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    unsupervised: UnsupervisedConfig = Field(default_factory=UnsupervisedConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    quality_gate: QualityGateConfig = Field(default_factory=QualityGateConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig)

    def resolve_path(self, path_str: str | Path) -> Path:
        p = Path(path_str)
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()

    def resolved_data_uri(self) -> str:
        uri = self.data.uri
        if uri.startswith(("gs://", "file://")):
            return uri
        return str(self.resolve_path(uri))


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load YAML configuration and apply environment overrides."""
    env_cfg = os.getenv("CANCER_MLOPS_CONFIG")
    cfg_path = Path(path or env_cfg or DEFAULT_CONFIG_PATH)
    if not cfg_path.is_absolute():
        cfg_path = (PROJECT_ROOT / cfg_path).resolve()

    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    cfg = AppConfig.model_validate(raw or {})

    if data_uri := os.getenv("CANCER_MLOPS_DATA_URI"):
        cfg.data.uri = data_uri
    if art_dir := os.getenv("CANCER_MLOPS_ARTIFACTS_DIR"):
        root = Path(art_dir)
        cfg.artifacts.root_dir = str(root)
        cfg.artifacts.db_file = str(root / "mlops.db")
        cfg.artifacts.registry_dir = str(root / "registry")
        cfg.artifacts.reports_dir = str(root / "reports")
    return cfg
