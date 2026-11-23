"""Leak-free transformers for feature engineering, collinearity pruning, and quantile binning."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

BASE_NUMERIC_FEATURES: tuple[str, ...] = (
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Select base numeric features and add domain feature ``area_ratio``."""

    def __init__(self, numeric_cols: tuple[str, ...] | list[str] | None = None) -> None:
        self.numeric_cols = numeric_cols

    def fit(self, X: pd.DataFrame, y: Any = None) -> FeatureEngineer:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureEngineer requires a pandas DataFrame.")
        if self.numeric_cols is not None:
            self.feature_cols_ = list(self.numeric_cols)
        else:
            self.feature_cols_ = [c for c in BASE_NUMERIC_FEATURES if c in X.columns]
            if not self.feature_cols_:
                self.feature_cols_ = list(X.select_dtypes(include="number").columns)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_cols_)
        out = X[self.feature_cols_].astype(float).copy()
        if "area_worst" in out.columns and "area_mean" in out.columns:
            denom = np.clip(out["area_mean"].to_numpy(dtype=float), 1e-9, None)
            out["area_ratio"] = out["area_worst"].to_numpy(dtype=float) / denom
        return out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        cols = list(self.feature_cols_)
        if "area_worst" in cols and "area_mean" in cols:
            cols.append("area_ratio")
        return np.asarray(cols, dtype=object)


class CollinearityPruner(BaseEstimator, TransformerMixin):
    """Fit target-correlation ranking and greedy collinearity pruning strictly on training data."""

    def __init__(
        self,
        keep_first: tuple[str, ...] | list[str] = ("radius_worst",),
        collinearity_threshold: float = 0.85,
        max_features: int = 6,
    ) -> None:
        self.keep_first = keep_first
        self.collinearity_threshold = collinearity_threshold
        self.max_features = max_features

    def fit(self, X: pd.DataFrame, y: Any) -> CollinearityPruner:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CollinearityPruner requires a pandas DataFrame.")
        y_series = pd.Series(np.asarray(y, dtype=float), index=X.index)

        rows: list[dict[str, Any]] = []
        for col in sorted(X.columns):
            if not pd.api.types.is_numeric_dtype(X[col]):
                continue
            r = float(X[col].astype(float).corr(y_series))
            if np.isnan(r):
                r = 0.0
            rows.append({"feature": col, "r": r, "abs_r": abs(r)})

        ranking = (
            pd.DataFrame(rows).sort_values("abs_r", ascending=False).reset_index(drop=True)
        )
        self.ranking_df_ = ranking

        keep_first_list = [f for f in self.keep_first if f in set(ranking["feature"])]
        ordered = keep_first_list + [
            f for f in ranking["feature"].tolist() if f not in keep_first_list
        ]

        chosen: list[str] = []
        dropped: list[dict[str, Any]] = []
        for feat in ordered:
            if len(chosen) >= self.max_features:
                break
            clash = None
            clash_r = 0.0
            for prev in chosen:
                corr_val = float(X[feat].astype(float).corr(X[prev].astype(float)))
                if np.isnan(corr_val):
                    corr_val = 0.0
                if abs(corr_val) > self.collinearity_threshold:
                    clash = prev
                    clash_r = corr_val
                    break
            if clash is None:
                chosen.append(feat)
            else:
                dropped.append({"feature": feat, "collinear_with": clash, "r": clash_r})

        self.selected_features_ = chosen
        self.dropped_collinear_ = dropped
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_features_].copy()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.selected_features_, dtype=object)


class QuantileBinner(BaseEstimator, TransformerMixin):
    """Fit quantile bin cutpoints strictly on training data with open outer bounds."""

    def __init__(
        self,
        feature: str = "radius_worst",
        labels: tuple[str, ...] | list[str] = ("low", "mid", "high"),
        replace: bool = True,
    ) -> None:
        self.feature = feature
        self.labels = labels
        self.replace = replace

    def fit(self, X: pd.DataFrame, y: Any = None) -> QuantileBinner:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("QuantileBinner requires a pandas DataFrame.")
        if self.feature not in X.columns:
            self.active_ = False
            self.edges_: tuple[float, ...] = ()
            self.cut_edges_: list[float] = []
            self.used_labels_: list[str] = []
            return self

        _, edges = pd.qcut(
            X[self.feature].dropna().astype(float),
            q=len(self.labels),
            retbins=True,
            duplicates="drop",
        )
        self.edges_ = tuple(float(e) for e in edges)
        self.cut_edges_ = [-float("inf"), *self.edges_[1:-1], float("inf")]
        self.used_labels_ = list(self.labels)[: len(self.cut_edges_) - 1]
        self.active_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        if not self.active_:
            return out
        binned = pd.cut(
            out[self.feature].astype(float),
            bins=self.cut_edges_,
            labels=self.used_labels_,
            include_lowest=True,
        ).astype(str)
        bin_col = f"{self.feature}_bin"
        if self.replace:
            cols = [bin_col if c == self.feature else c for c in out.columns]
            out[bin_col] = binned
            out = out[cols]
        else:
            out[bin_col] = binned
        return out


class DataFrameStandardScaler(BaseEstimator, TransformerMixin):
    """Standardize columns to zero mean and unit population SD while preserving DataFrame names."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> DataFrameStandardScaler:
        X_df = pd.DataFrame(X)
        self.feature_names_in_ = list(X_df.columns)
        self.mean_ = X_df.astype(float).mean(axis=0)
        self.sd_ = X_df.astype(float).std(axis=0, ddof=0).replace(0.0, 1.0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = pd.DataFrame(X, columns=self.feature_names_in_)
        return (X_df.astype(float) - self.mean_) / self.sd_

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.feature_names_in_, dtype=object)
