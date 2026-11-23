"""End-to-end MLOps pipeline CLI entrypoint."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from cancer_mlops.config import AppConfig, load_config
from cancer_mlops.data import load_and_partition
from cancer_mlops.evaluate import evaluate_models_on_test
from cancer_mlops.monitoring import compute_baseline_stats, detect_batch_drift
from cancer_mlops.tracking import (
    ExperimentTracker,
    ModelRegistry,
    QualityGate,
    generate_model_card,
)
from cancer_mlops.train import cross_validate_and_train
from cancer_mlops.unsupervised import run_unsupervised_analysis

log = logging.getLogger("cancer_mlops.pipeline")


def run_pipeline(stage: str = "all", config_path: Path | None = None) -> dict[str, Any]:
    """Execute one pipeline stage or the complete end-to-end MLOps lifecycle."""
    cfg: AppConfig = load_config(config_path)
    reports_dir = cfg.resolve_path(cfg.artifacts.reports_dir)
    db_file = cfg.resolve_path(cfg.artifacts.db_file)
    registry_dir = cfg.resolve_path(cfg.artifacts.registry_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df, audit_summary = load_and_partition(cfg)
    raw_features: list[str] = audit_summary["raw_numeric_features"]
    (reports_dir / "audit_summary.json").write_text(
        json.dumps(audit_summary, indent=2), encoding="utf-8"
    )
    log.info(
        "Audit complete: train=%d, test=%d, sha256=%s",
        len(train_df),
        len(test_df),
        audit_summary["data_sha256"][:12],
    )

    if stage == "audit":
        return {"audit_summary": audit_summary}

    unsup_summary: dict[str, Any] = {}
    if stage in ("unsupervised", "all"):
        unsup_summary = run_unsupervised_analysis(
            train_df=train_df,
            test_df=test_df,
            feature_cols=raw_features,
            cfg=cfg,
            reports_dir=reports_dir,
        )
        log.info(
            "Unsupervised complete: PCA 95%%=%d comps, K=2 Test ARI=%.3f",
            unsup_summary["pca"]["n_components_95pct"],
            unsup_summary["kmeans"]["k2_posthoc_alignment"]["test_ari"],
        )
        if stage == "unsupervised":
            return {"audit_summary": audit_summary, "unsupervised_summary": unsup_summary}

    if stage == "drift":
        registry = ModelRegistry(db_file, registry_dir)
        prod = registry.load_production()
        drift_report = detect_batch_drift(
            test_df[raw_features],
            prod["metadata"]["baseline_stats"],
            cfg.monitoring,
        )
        (reports_dir / "drift_report.json").write_text(
            json.dumps(drift_report, indent=2), encoding="utf-8"
        )
        log.info(
            "Drift check complete: status=%s, max_psi=%.4f",
            drift_report["overall_status"],
            drift_report["max_psi"],
        )
        return {"drift_report": drift_report}

    tracker = ExperimentTracker(db_file)
    run_id = tracker.start_run(
        experiment_name="breast_cancer_mlops",
        config_dict=cfg.model_dump(),
        data_sha256=audit_summary["data_sha256"],
        notes=f"stage={stage}",
    )

    try:
        train_out = cross_validate_and_train(
            train_df=train_df,
            raw_numeric_features=raw_features,
            cfg=cfg,
            reports_dir=reports_dir,
        )
        for row in train_out["cv_rows"]:
            if row["eval_mode"] == "cv_tuned":
                tracker.log_metrics(run_id, "cv_tuned", row["model"], row)

        if stage == "train":
            tracker.finish_run(run_id, status="finished")
            return {"run_id": run_id, "train": train_out}

        champion_name = train_out["champion_name"]
        eval_out = evaluate_models_on_test(
            models=train_out["models"],
            tuned_thresholds=train_out["tuned_thresholds"],
            train_df=train_df,
            test_df=test_df,
            raw_numeric_features=raw_features,
            reports_dir=reports_dir,
            champion_name=champion_name,
            bin_feature=cfg.features.bin_feature,
            bin_labels=tuple(cfg.features.bin_labels),
        )
        for model_name, metrics in eval_out["tuned_metrics_by_model"].items():
            tracker.log_metrics(run_id, "test_tuned", model_name, metrics)

        champ_metrics = eval_out["tuned_metrics_by_model"][champion_name]
        champ_proba = eval_out["test_probas"][champion_name]
        gate = QualityGate(cfg.quality_gate)
        gate_passed, gate_reasons = gate.evaluate(champ_metrics, champ_proba)
        log.info("Champion=%s | QualityGate=%s | %s", champion_name, gate_passed, gate_reasons)

        baseline_stats = compute_baseline_stats(
            train_df, raw_features, n_bins=cfg.monitoring.n_bins
        )
        registry = ModelRegistry(db_file, registry_dir)
        model_stage = "production" if gate_passed else "staging"
        schema_meta = {
            "data_sha256": audit_summary["data_sha256"],
            "raw_numeric_features": raw_features,
            "selected_features": train_out["selected_features"],
            "target_col": cfg.data.target_col,
            "positive_label": cfg.data.positive_label,
        }
        bundle = {
            "estimator": train_out["models"][champion_name],
            "threshold": train_out["tuned_thresholds"][champion_name],
            "model_name": champion_name,
            "raw_numeric_features": raw_features,
            "selected_features": train_out["selected_features"],
        }
        version = registry.register_model(
            model_name=champion_name,
            run_id=run_id,
            bundle=bundle,
            schema=schema_meta,
            metrics=champ_metrics,
            baseline_stats=baseline_stats,
            stage=model_stage,
        )

        if not unsup_summary and (reports_dir / "unsupervised_summary.json").exists():
            unsup_summary = json.loads(
                (reports_dir / "unsupervised_summary.json").read_text(encoding="utf-8")
            )
        if not unsup_summary:
            unsup_summary = run_unsupervised_analysis(
                train_df=train_df,
                test_df=test_df,
                feature_cols=raw_features,
                cfg=cfg,
                reports_dir=reports_dir,
            )

        prod_meta = {
            "model_name": champion_name,
            "version": version,
            "run_id": run_id,
            "stage": model_stage,
            "created_at": pd.Timestamp.utcnow().isoformat(),
            "schema": schema_meta,
            "metrics": champ_metrics,
        }
        card_path = generate_model_card(
            out_path=reports_dir / "model_card.md",
            production_meta=prod_meta,
            cv_df=train_out["cv_df"],
            test_df=eval_out["test_df_summary"],
            slice_df=eval_out["slice_df"],
            unsupervised_summary=unsup_summary,
            audit_summary=audit_summary,
            gate_reasons=gate_reasons,
        )

        drift_report = detect_batch_drift(
            test_df[raw_features], baseline_stats, cfg.monitoring
        )
        drift_path = reports_dir / "drift_report.json"
        drift_path.write_text(json.dumps(drift_report, indent=2), encoding="utf-8")

        for art_name, art_path in (
            ("cv_comparison", reports_dir / "cv_comparison.csv"),
            ("test_comparison", reports_dir / "test_comparison.csv"),
            ("slice_evaluation", reports_dir / "slice_evaluation.csv"),
            ("permutation_importance", reports_dir / "permutation_importance.csv"),
            ("model_card", card_path),
            ("drift_report", drift_path),
        ):
            if art_path.exists():
                tracker.log_artifact(run_id, art_name, art_path)

        tracker.finish_run(run_id, status="finished" if gate_passed else "gate_failed")
        return {
            "run_id": run_id,
            "champion_name": champion_name,
            "version": version,
            "stage": model_stage,
            "gate_passed": gate_passed,
            "gate_reasons": gate_reasons,
            "champ_metrics": champ_metrics,
            "drift_report": drift_report,
        }
    except Exception:
        tracker.finish_run(run_id, status="failed")
        raise


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Cancer MLOps Pipeline Runner")
    parser.add_argument(
        "--stage",
        choices=["audit", "unsupervised", "train", "evaluate", "drift", "all"],
        default="all",
    )
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    run_pipeline(stage=args.stage, config_path=args.config)


if __name__ == "__main__":
    main()
