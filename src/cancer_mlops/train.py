"""Supervised training, StratifiedKFold CV, hyperparameter tuning, and threshold optimization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from statsmodels.stats.outliers_influence import variance_inflation_factor

from cancer_mlops.chaid import ChaidClassifier
from cancer_mlops.config import AppConfig
from cancer_mlops.evaluate import compute_metrics
from cancer_mlops.features import (
    CollinearityPruner,
    DataFrameStandardScaler,
    FeatureEngineer,
    QuantileBinner,
)


class StatsmodelsLogitClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible wrapper around statsmodels.Logit with OR and VIF exports."""

    def __init__(self, maxiter: int = 500) -> None:
        self.maxiter = maxiter

    def fit(self, X: pd.DataFrame, y: Any) -> StatsmodelsLogitClassifier:
        X_df = pd.DataFrame(X).astype(float)
        self.feature_names_ = list(X_df.columns)
        self.classes_ = np.array([0, 1])
        X_const = sm.add_constant(X_df, has_constant="add")
        y_arr = np.asarray(y, dtype=int)
        self.res_ = sm.Logit(y_arr, X_const).fit(disp=0, maxiter=self.maxiter)
        if not self.res_.mle_retvals.get("converged", False):
            raise RuntimeError("Statsmodels Logit failed to converge.")
        self.X_train_df_ = X_df
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_df = pd.DataFrame(X, columns=self.feature_names_).astype(float)
        X_const = sm.add_constant(X_df, has_constant="add")
        pos = np.clip(np.asarray(self.res_.predict(X_const), dtype=float), 0.0, 1.0)
        return np.column_stack([1.0 - pos, pos])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def or_table(self) -> pd.DataFrame:
        table = pd.DataFrame({"B_per_SD": self.res_.params, "p": self.res_.pvalues})
        table["OR_per_SD"] = np.exp(table["B_per_SD"])
        conf = self.res_.conf_int()
        table["OR_lo"] = np.exp(conf[0])
        table["OR_hi"] = np.exp(conf[1])
        return table.round({"B_per_SD": 4, "p": 5, "OR_per_SD": 3, "OR_lo": 3, "OR_hi": 3})

    def vif_table(self) -> pd.DataFrame:
        X_const = sm.add_constant(self.X_train_df_, has_constant="add")
        rows = [
            {"feature": col, "VIF": variance_inflation_factor(X_const.values, i + 1)}
            for i, col in enumerate(self.X_train_df_.columns)
        ]
        return pd.DataFrame(rows).round({"VIF": 2})


def optimize_threshold(
    y_true: np.ndarray,
    oof_proba: np.ndarray,
    recall_target: float = 0.96,
    beta: float = 2.0,
) -> float:
    """Select clinical probability threshold on CV out-of-fold probabilities only."""
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(oof_proba, dtype=float)
    candidates = np.round(np.arange(0.10, 0.91, 0.01), 2)

    records: list[tuple[float, float, float, float]] = []
    for thr in candidates:
        pred = (p_arr >= thr).astype(int)
        rec = float(recall_score(y_arr, pred, zero_division=0))
        prec = float(precision_score(y_arr, pred, zero_division=0))
        fbeta = float(fbeta_score(y_arr, pred, beta=beta, zero_division=0))
        records.append((float(thr), rec, prec, fbeta))

    eligible = [r for r in records if r[1] >= recall_target]
    if eligible:
        best = max(eligible, key=lambda r: (r[3], r[2], -abs(r[0] - 0.5)))
    else:
        best = max(records, key=lambda r: (r[3], r[1], -abs(r[0] - 0.5)))
    return float(best[0])


def _tune_logistic_c(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    raw_numeric_features: list[str],
    cfg: AppConfig,
    cv: StratifiedKFold,
) -> float:
    best_c = 1.0
    best_auc = -1.0
    for c_val in (0.1, 1.0, 10.0):
        pipe = Pipeline(
            [
                ("fe", FeatureEngineer(numeric_cols=raw_numeric_features)),
                (
                    "prune",
                    CollinearityPruner(
                        keep_first=tuple(cfg.features.keep_first),
                        collinearity_threshold=cfg.features.collinearity_threshold,
                        max_features=cfg.features.max_features,
                    ),
                ),
                ("scale", DataFrameStandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=c_val,
                        max_iter=1000,
                        random_state=cfg.training.random_state,
                    ),
                ),
            ]
        )
        oof = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        auc = float(roc_auc_score(y_train, oof))
        if auc > best_auc:
            best_auc = auc
            best_c = c_val
    return best_c


def build_candidate_pipelines(
    raw_numeric_features: list[str],
    cfg: AppConfig,
    best_logistic_c: float,
) -> dict[str, Pipeline]:
    """Construct leak-free end-to-end scikit-learn pipelines for all candidate models."""
    seed = cfg.training.random_state
    keep_first = tuple(cfg.features.keep_first)
    thresh = cfg.features.collinearity_threshold
    max_f = cfg.features.max_features
    bin_labels = tuple(cfg.features.bin_labels)
    ch_cfg = cfg.training.chaid

    def pruner() -> CollinearityPruner:
        return CollinearityPruner(
            keep_first=keep_first,
            collinearity_threshold=thresh,
            max_features=max_f,
        )

    return {
        "logistic_l2": Pipeline(
            [
                ("fe", FeatureEngineer(numeric_cols=raw_numeric_features)),
                ("prune", pruner()),
                ("scale", DataFrameStandardScaler()),
                ("clf", LogisticRegression(C=best_logistic_c, max_iter=1000, random_state=seed)),
            ]
        ),
        "logistic_statsmodels": Pipeline(
            [
                ("fe", FeatureEngineer(numeric_cols=raw_numeric_features)),
                ("prune", pruner()),
                ("scale", DataFrameStandardScaler()),
                ("clf", StatsmodelsLogitClassifier(maxiter=500)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("fe", FeatureEngineer(numeric_cols=raw_numeric_features)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=6,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "hist_gbdt": Pipeline(
            [
                ("fe", FeatureEngineer(numeric_cols=raw_numeric_features)),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_depth=4,
                        learning_rate=0.05,
                        max_iter=150,
                        min_samples_leaf=15,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "chaid_bonferroni": Pipeline(
            [
                ("fe", FeatureEngineer(numeric_cols=raw_numeric_features)),
                ("prune", pruner()),
                (
                    "bin",
                    QuantileBinner(
                        feature=cfg.features.bin_feature,
                        labels=bin_labels,
                        replace=True,
                    ),
                ),
                (
                    "clf",
                    ChaidClassifier(
                        alpha_merge=ch_cfg.alpha_merge,
                        alpha_split=ch_cfg.alpha_split,
                        max_depth=ch_cfg.max_depth,
                        min_child=ch_cfg.min_child,
                        n_bins=ch_cfg.n_bins,
                        bonferroni=True,
                        ordered_labels=bin_labels,
                    ),
                ),
            ]
        ),
        "chaid_unadjusted": Pipeline(
            [
                ("fe", FeatureEngineer(numeric_cols=raw_numeric_features)),
                ("prune", pruner()),
                (
                    "bin",
                    QuantileBinner(
                        feature=cfg.features.bin_feature,
                        labels=bin_labels,
                        replace=True,
                    ),
                ),
                (
                    "clf",
                    ChaidClassifier(
                        alpha_merge=ch_cfg.alpha_merge,
                        alpha_split=ch_cfg.alpha_split,
                        max_depth=ch_cfg.max_depth,
                        min_child=ch_cfg.min_child,
                        n_bins=ch_cfg.n_bins,
                        bonferroni=False,
                        ordered_labels=bin_labels,
                    ),
                ),
            ]
        ),
    }


def cross_validate_and_train(
    train_df: pd.DataFrame,
    raw_numeric_features: list[str],
    cfg: AppConfig,
    reports_dir: Path,
) -> dict[str, Any]:
    """Run leak-free StratifiedKFold CV, tune thresholds on OOF, and fit final models on train."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    X_train = train_df[raw_numeric_features].copy()
    y_train = train_df["y_m"].astype(int).to_numpy()

    cv = StratifiedKFold(
        n_splits=cfg.training.cv_folds,
        shuffle=True,
        random_state=cfg.training.random_state,
    )
    best_c = _tune_logistic_c(X_train, y_train, raw_numeric_features, cfg, cv)
    candidates = build_candidate_pipelines(raw_numeric_features, cfg, best_logistic_c=best_c)

    cv_rows: list[dict[str, Any]] = []
    tuned_thresholds: dict[str, float] = {}
    fitted_models: dict[str, Pipeline] = {}

    for name, template in candidates.items():
        oof_proba = cross_val_predict(
            clone(template),
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
        )[:, 1]
        tuned_thr = optimize_threshold(
            y_train,
            oof_proba,
            recall_target=cfg.training.recall_target,
            beta=cfg.training.f_beta,
        )
        tuned_thresholds[name] = tuned_thr

        m_default = compute_metrics(y_train, oof_proba, threshold=0.50)
        m_tuned = compute_metrics(y_train, oof_proba, threshold=tuned_thr)
        cv_rows.append({"model": name, "eval_mode": "default_0.50", **m_default})
        cv_rows.append({"model": name, "eval_mode": "cv_tuned", **m_tuned})

        fitted = clone(template).fit(X_train, y_train)
        fitted_models[name] = fitted

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(reports_dir / "cv_comparison.csv", index=False)

    # Export interpretability artifacts strictly from train-fitted models
    sm_pipe = fitted_models["logistic_statsmodels"]
    sm_clf: StatsmodelsLogitClassifier = sm_pipe.named_steps["clf"]
    sm_clf.or_table().to_csv(reports_dir / "logistic_or.csv")
    sm_clf.vif_table().to_csv(reports_dir / "logistic_vif.csv", index=False)

    pruner: CollinearityPruner = sm_pipe.named_steps["prune"]
    pruner.ranking_df_.to_csv(reports_dir / "feature_ranking.csv", index=False)
    selected_features = list(pruner.selected_features_)

    chaid_bonf: ChaidClassifier = fitted_models["chaid_bonferroni"].named_steps["clf"]
    (reports_dir / "chaid_rules.txt").write_text("\n".join(chaid_bonf.rules()), encoding="utf-8")

    chaid_unadj: ChaidClassifier = fitted_models["chaid_unadjusted"].named_steps["clf"]
    (reports_dir / "chaid_unadjusted_rules.txt").write_text(
        "\n".join(chaid_unadj.rules()), encoding="utf-8"
    )

    # Select champion strictly from CV tuned results (prefer higher roc_auc, recall, lower brier)
    tuned_cv = [r for r in cv_rows if r["eval_mode"] == "cv_tuned"]
    champion_row = max(
        tuned_cv,
        key=lambda r: (r["roc_auc"], r["recall_M"], -r["brier_score"]),
    )
    champion_name = str(champion_row["model"])

    return {
        "models": fitted_models,
        "tuned_thresholds": tuned_thresholds,
        "cv_rows": cv_rows,
        "cv_df": cv_df,
        "champion_name": champion_name,
        "selected_features": selected_features,
        "best_logistic_c": best_c,
    }
