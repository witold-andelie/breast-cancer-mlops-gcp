"""Production-hardened scikit-learn-compatible CHAID decision tree with Bonferroni adjustment."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.base import BaseEstimator, ClassifierMixin


@dataclass(frozen=True)
class Binning:
    """Equal-frequency quantile binning with open outer edges for unseen test values."""

    col: str
    edges: tuple[float, ...]
    na_code: int

    def transform(self, s: pd.Series) -> pd.Series:
        cut_edges = (
            [-float("inf"), *self.edges[1:-1], float("inf")]
            if len(self.edges) >= 2
            else list(self.edges)
        )
        codes = pd.cut(s, bins=cut_edges, include_lowest=True, labels=False).astype(float)
        name = s.name or self.col
        return codes.fillna(float(self.na_code)).astype(int).rename(name)


def make_binning(s: pd.Series, n_bins: int) -> Binning:
    edges_t = pd.qcut(s.dropna(), q=n_bins, retbins=True, duplicates="drop")[1]
    edges = tuple(float(e) for e in edges_t)
    return Binning(col=str(s.name), edges=edges, na_code=len(edges) - 1)


def merge_adjacent(
    board: pd.DataFrame,
    na_code: int,
    alpha: float,
    bonferroni: bool = True,
) -> tuple[list[tuple[tuple[int, ...], pd.Series]], float]:
    """Iteratively merge homogeneous adjacent bins via chi-square tests.

    When ``bonferroni=True``, applies the ordinal Bonferroni multiplier
    ``comb(c - 1, r - 1)`` (capped at 1.0), where ``c`` is the initial number
    of non-empty bins and ``r`` is the merged group count.
    """
    groups: list[tuple[tuple[int, ...], pd.Series]] = [
        ((int(code),), board.loc[code])
        for code in board.index
        if int(code) != int(na_code) and board.loc[code].sum() > 0
    ]
    c_initial = len(groups)

    while len(groups) > 1:
        best_i, best_p = None, -1.0
        for i in range(len(groups) - 1):
            table = pd.concat([groups[i][1], groups[i + 1][1]], axis=1).T
            if (table.sum(axis=1) == 0).any() or (table.sum(axis=0) == 0).any():
                p = 1.0
            else:
                p = float(chi2_contingency(table.to_numpy())[1])
            if p > best_p:
                best_i, best_p = i, p
        if best_i is not None and best_p >= alpha:
            (c1, v1), (c2, v2) = groups[best_i], groups[best_i + 1]
            groups[best_i] = (c1 + c2, v1 + v2)
            groups.pop(best_i + 1)
        else:
            break

    split_p = float("nan")
    r_final = len(groups)
    if r_final >= 2:
        merged_counts = pd.DataFrame(
            [counts.reindex(board.columns).to_dict() for _, counts in groups]
        )
        raw_p = float(chi2_contingency(merged_counts.to_numpy())[1])
        if bonferroni and c_initial > r_final:
            multiplier = math.comb(c_initial - 1, r_final - 1)
            split_p = min(1.0, raw_p * multiplier)
        else:
            split_p = raw_p
    return groups, split_p


@dataclass
class Node:
    n: int
    prob: float
    feature: str | None = None
    split_p: float | None = None
    children: list[tuple[tuple[int, ...], Node]] = field(default_factory=list)


class ChaidClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible CHAID tree with ordinal Bonferroni correction."""

    def __init__(
        self,
        features: list[str] | None = None,
        *,
        alpha_merge: float = 0.05,
        alpha_split: float = 0.05,
        max_depth: int = 4,
        min_child: int = 40,
        n_bins: int = 10,
        bonferroni: bool = True,
        ordered_labels: tuple[str, ...] = ("low", "mid", "high"),
    ) -> None:
        self.features = features
        self.alpha_merge = alpha_merge
        self.alpha_split = alpha_split
        self.max_depth = max_depth
        self.min_child = min_child
        self.n_bins = n_bins
        self.bonferroni = bonferroni
        self.ordered_labels = ordered_labels

    def _binned(self, X: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
        out: dict[str, pd.Series] = {}
        for f in features or self.features_:
            if f in self.bins_:
                out[f] = self.bins_[f].transform(X[f])
            else:
                out[f] = X[f].astype(str).map(self.cat_maps_[f]).fillna(-1).astype(int).rename(f)
        return pd.DataFrame(out, index=X.index)

    def fit(self, X: pd.DataFrame, y: Any) -> ChaidClassifier:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ChaidClassifier requires a pandas DataFrame.")
        y_s = pd.Series(np.asarray(y, dtype=float), index=X.index)
        self.classes_ = np.array([0, 1])
        self.features_ = list(self.features) if self.features is not None else list(X.columns)
        self.bins_: dict[str, Binning] = {}
        self.cat_maps_: dict[str, dict[str, int]] = {}
        self.cat_labels_: dict[str, list[str]] = {}

        for f in self.features_:
            if pd.api.types.is_numeric_dtype(X[f]):
                self.bins_[f] = make_binning(X[f], self.n_bins)
            else:
                cats = list(dict.fromkeys(X[f].astype(str)))
                if set(cats) <= set(self.ordered_labels):
                    cats = [
                        c for c in self.ordered_labels if c in cats
                    ] or list(self.ordered_labels)
                else:
                    cats = sorted(cats)
                self.cat_labels_[f] = cats
                self.cat_maps_[f] = {c: i for i, c in enumerate(cats)}

        self.root_ = self._grow(X, y_s, depth=0)
        return self

    def _grow(self, X: pd.DataFrame, y: pd.Series, depth: int) -> Node:
        leaf = Node(n=len(y), prob=float(y.mean()))
        if depth >= self.max_depth or len(y) < 2 * self.min_child or y.nunique() < 2:
            return leaf

        best: tuple[float, str, list[tuple[tuple[int, ...], pd.Series]]] | None = None
        for f in self.features_:
            codes_f = self._binned(X, features=[f])[f]
            board = pd.crosstab(codes_f, y).reindex(columns=[0.0, 1.0], fill_value=0.0)
            na_code = self.bins_[f].na_code if f in self.bins_ else -1
            groups, p = merge_adjacent(board, na_code, self.alpha_merge, bonferroni=self.bonferroni)
            if len(groups) < 2:
                continue
            sizes = [int(r.sum()) for _, r in groups]
            ok = (p == p) and (p < self.alpha_split) and (min(sizes) >= self.min_child)
            if ok and (best is None or p < best[0]):
                best = (p, f, groups)

        if best is None:
            return leaf

        p, f, groups = best
        node = Node(n=len(y), prob=float(y.mean()), feature=f, split_p=p)
        codes_f = self._binned(X, features=[f])[f]
        for codes, _ in groups:
            mask = codes_f.isin(codes)
            node.children.append((codes, self._grow(X[mask], y[mask], depth + 1)))
        return node

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "root_") or self.root_ is None:
            raise RuntimeError("Call fit() before predict_proba().")
        codes = self._binned(X)
        pos = np.array(
            [self._walk(self.root_, row) for _, row in codes.iterrows()],
            dtype=float,
        )
        pos = np.clip(pos, 0.0, 1.0)
        return np.column_stack([1.0 - pos, pos])

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def _walk(self, node: Node, row: pd.Series) -> float:
        if not node.children or node.feature is None:
            return node.prob
        col_val = row[node.feature]
        for codes, child in node.children:
            if col_val in codes:
                return self._walk(child, row)
        return node.prob

    def rules(self) -> list[str]:
        if not hasattr(self, "root_") or self.root_ is None:
            raise RuntimeError("Call fit() before rules().")
        out: list[str] = []
        self._render(self.root_, depth=0, condition="", out=out)
        return out

    def _bin_edge_text(self, feature: str, codes: tuple[int, ...]) -> str:
        labels = self.cat_labels_.get(feature)
        if labels is not None:
            names = [labels[c] if 0 <= c < len(labels) else str(c) for c in sorted(codes)]
            return " OR ".join(names)
        if feature in self.bins_:
            edges = self.bins_[feature].edges
            na = self.bins_[feature].na_code
            parts = []
            for c in sorted(codes):
                if c == na:
                    parts.append("NA")
                else:
                    parts.append(f"[{edges[c]:.3f}, {edges[c + 1]:.3f})")
            return " OR ".join(parts)
        return f"level={','.join(map(str, codes))}"

    def _render(self, node: Node, depth: int, condition: str, out: list[str]) -> None:
        pad = "    " * depth
        head = f"{condition}n={node.n} P(M)={node.prob:.3f}"
        if node.feature is None:
            out.append(f"{pad}LEAF   {head}")
            return
        p_label = "bonf-p" if self.bonferroni else "raw-p"
        out.append(
            f"{pad}SPLIT  feature={node.feature} ({p_label}={node.split_p:.2e})  {head}"
        )
        for codes, child in node.children:
            text = self._bin_edge_text(node.feature, codes)
            self._render(child, depth + 1, f"{text} -> ", out)
