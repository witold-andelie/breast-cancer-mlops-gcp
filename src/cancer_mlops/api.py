"""FastAPI production inference service with Pydantic v2 validation and online drift warnings."""

from __future__ import annotations

import logging
import math
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from cancer_mlops.config import AppConfig, load_config
from cancer_mlops.monitoring import check_single_sample_drift, detect_batch_drift
from cancer_mlops.tracking import ModelRegistry

log = logging.getLogger("cancer_mlops.api")


class TumorFeatures(BaseModel):
    radius_mean: float = Field(..., ge=0.0)
    texture_mean: float = Field(..., ge=0.0)
    perimeter_mean: float = Field(..., ge=0.0)
    area_mean: float = Field(..., ge=0.0)
    smoothness_mean: float = Field(..., ge=0.0)
    compactness_mean: float = Field(..., ge=0.0)
    concavity_mean: float = Field(..., ge=0.0)
    concave_points_mean: float = Field(..., ge=0.0)
    symmetry_mean: float = Field(..., ge=0.0)
    fractal_dimension_mean: float = Field(..., ge=0.0)
    radius_se: float = Field(..., ge=0.0)
    texture_se: float = Field(..., ge=0.0)
    perimeter_se: float = Field(..., ge=0.0)
    area_se: float = Field(..., ge=0.0)
    smoothness_se: float = Field(..., ge=0.0)
    compactness_se: float = Field(..., ge=0.0)
    concavity_se: float = Field(..., ge=0.0)
    concave_points_se: float = Field(..., ge=0.0)
    symmetry_se: float = Field(..., ge=0.0)
    fractal_dimension_se: float = Field(..., ge=0.0)
    radius_worst: float = Field(..., ge=0.0)
    texture_worst: float = Field(..., ge=0.0)
    perimeter_worst: float = Field(..., ge=0.0)
    area_worst: float = Field(..., ge=0.0)
    smoothness_worst: float = Field(..., ge=0.0)
    compactness_worst: float = Field(..., ge=0.0)
    concavity_worst: float = Field(..., ge=0.0)
    concave_points_worst: float = Field(..., ge=0.0)
    symmetry_worst: float = Field(..., ge=0.0)
    fractal_dimension_worst: float = Field(..., ge=0.0)

    @field_validator("*", mode="after")
    @classmethod
    def check_finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("Feature values must be finite numbers.")
        return v


class BatchPredictRequest(BaseModel):
    records: list[TumorFeatures] = Field(..., min_length=1, max_length=2000)


class FeatureContribution(BaseModel):
    feature: str
    contribution: float


class PredictResponse(BaseModel):
    model_name: str
    model_version: int
    probability_malignant: float
    prediction: Literal["M", "B"]
    threshold: float
    risk_tier: Literal["low", "medium", "high"]
    drift_warnings: list[str]
    top_contributions: list[FeatureContribution]
    latency_ms: float


class BatchPredictResponse(BaseModel):
    model_name: str
    model_version: int
    predictions: list[PredictResponse]
    drift_summary: dict[str, Any]


def _risk_tier(prob: float) -> Literal["low", "medium", "high"]:
    if prob < 0.20:
        return "low"
    if prob >= 0.70:
        return "high"
    return "medium"


def _compute_contributions(estimator: Any, row_df: pd.DataFrame) -> list[FeatureContribution]:
    """Compute per-sample linear log-odds contributions or tree importance fallback."""
    if not hasattr(estimator, "named_steps"):
        return []
    steps = estimator.named_steps
    clf = steps.get("clf")
    if clf is None:
        return []

    if "scale" in steps and "prune" in steps:
        x_t = row_df
        for step_name in ("fe", "prune", "scale"):
            if step_name in steps:
                x_t = steps[step_name].transform(x_t)
        cols = list(x_t.columns)
        vals = x_t.iloc[0].to_numpy(dtype=float)

        coefs: np.ndarray | None = None
        if hasattr(clf, "coef_"):
            coefs = np.asarray(clf.coef_[0], dtype=float)
        elif hasattr(clf, "res_"):
            coefs = np.asarray(clf.res_.params[1:], dtype=float)

        if coefs is not None and len(coefs) == len(cols):
            pairs = sorted(
                zip(cols, coefs * vals, strict=True),
                key=lambda kv: abs(kv[1]),
                reverse=True,
            )[:5]
            return [
                FeatureContribution(feature=str(f), contribution=round(float(c), 4))
                for f, c in pairs
            ]

    if hasattr(clf, "feature_importances_") and "fe" in steps:
        cols = list(steps["fe"].get_feature_names_out())
        imps = np.asarray(clf.feature_importances_, dtype=float)
        if len(cols) == len(imps):
            pairs = sorted(zip(cols, imps, strict=True), key=lambda kv: kv[1], reverse=True)[:5]
            return [
                FeatureContribution(feature=str(f), contribution=round(float(c), 4))
                for f, c in pairs
            ]
    return []


def _require_loaded_model(app: FastAPI) -> dict[str, Any]:
    prod = getattr(app.state, "prod", None)
    if prod is None:
        raise HTTPException(
            status_code=503,
            detail="No production model is loaded in registry. Run pipeline first.",
        )
    return prod


def create_app(
    config_path: Path | None = None,
    db_path: Path | None = None,
    registry_dir: Path | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        cfg: AppConfig = load_config(config_path)
        application.state.cfg = cfg
        resolved_db = db_path or cfg.resolve_path(cfg.artifacts.db_file)
        resolved_reg = registry_dir or cfg.resolve_path(cfg.artifacts.registry_dir)
        application.state.prod = None
        if resolved_db.exists():
            try:
                reg = ModelRegistry(resolved_db, resolved_reg)
                application.state.prod = reg.load_production()
                meta = application.state.prod["metadata"]
                log.info(
                    "Loaded production model %s v%s",
                    meta["model_name"],
                    meta["version"],
                )
            except FileNotFoundError:
                log.warning("Registry exists at %s, but no production model promoted.", resolved_db)
        else:
            log.warning("Registry DB not found at %s; API started without model.", resolved_db)
        yield

    application = FastAPI(
        title="Cancer MLOps Inference API",
        version="0.1.0",
        description="Production malignancy prediction API with drift warnings and model lineage.",
        lifespan=lifespan,
    )

    @application.get("/health")
    def health() -> dict[str, Any]:
        prod = getattr(application.state, "prod", None)
        if prod is None:
            return {
                "status": "ok",
                "model_loaded": False,
                "model_name": None,
                "model_version": None,
            }
        meta = prod["metadata"]
        return {
            "status": "ok",
            "model_loaded": True,
            "model_name": meta["model_name"],
            "model_version": meta["version"],
        }

    @application.get("/model/info")
    def model_info() -> dict[str, Any]:
        prod = _require_loaded_model(application)
        meta = prod["metadata"]
        return {
            "model_name": meta["model_name"],
            "version": meta["version"],
            "stage": meta["stage"],
            "run_id": meta["run_id"],
            "created_at": meta["created_at"],
            "threshold": prod["bundle"]["threshold"],
            "schema": meta["schema"],
            "metrics": meta["metrics"],
        }

    @application.post("/predict", response_model=PredictResponse)
    def predict(payload: TumorFeatures) -> PredictResponse:
        t0 = time.perf_counter()
        prod = _require_loaded_model(application)
        bundle = prod["bundle"]
        meta = prod["metadata"]
        cfg: AppConfig = application.state.cfg

        record = payload.model_dump()
        row_df = pd.DataFrame([record])
        est = bundle["estimator"]
        thr = float(bundle["threshold"])
        prob = float(est.predict_proba(row_df)[0, 1])
        pred_label: Literal["M", "B"] = "M" if prob >= thr else "B"

        warnings = check_single_sample_drift(
            record,
            meta.get("baseline_stats", {}),
            z_threshold=cfg.monitoring.z_score_alert,
        )
        contribs = _compute_contributions(est, row_df)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        return PredictResponse(
            model_name=meta["model_name"],
            model_version=int(meta["version"]),
            probability_malignant=round(prob, 6),
            prediction=pred_label,
            threshold=thr,
            risk_tier=_risk_tier(prob),
            drift_warnings=warnings,
            top_contributions=contribs,
            latency_ms=latency_ms,
        )

    @application.post("/predict/batch", response_model=BatchPredictResponse)
    def predict_batch(payload: BatchPredictRequest) -> BatchPredictResponse:
        t0 = time.perf_counter()
        prod = _require_loaded_model(application)
        bundle = prod["bundle"]
        meta = prod["metadata"]
        cfg: AppConfig = application.state.cfg

        records = [r.model_dump() for r in payload.records]
        batch_df = pd.DataFrame(records)
        est = bundle["estimator"]
        thr = float(bundle["threshold"])
        probs = est.predict_proba(batch_df)[:, 1]
        baseline_stats = meta.get("baseline_stats", {})

        preds: list[PredictResponse] = []
        per_item_ms = round(((time.perf_counter() - t0) * 1000.0) / max(len(records), 1), 3)
        for idx, rec in enumerate(records):
            p = float(probs[idx])
            row_df = batch_df.iloc[[idx]]
            preds.append(
                PredictResponse(
                    model_name=meta["model_name"],
                    model_version=int(meta["version"]),
                    probability_malignant=round(p, 6),
                    prediction="M" if p >= thr else "B",
                    threshold=thr,
                    risk_tier=_risk_tier(p),
                    drift_warnings=check_single_sample_drift(
                        rec, baseline_stats, z_threshold=cfg.monitoring.z_score_alert
                    ),
                    top_contributions=_compute_contributions(est, row_df),
                    latency_ms=per_item_ms,
                )
            )

        drift_summary = detect_batch_drift(batch_df, baseline_stats, cfg.monitoring)
        return BatchPredictResponse(
            model_name=meta["model_name"],
            model_version=int(meta["version"]),
            predictions=preds,
            drift_summary=drift_summary,
        )

    return application


app = create_app()
