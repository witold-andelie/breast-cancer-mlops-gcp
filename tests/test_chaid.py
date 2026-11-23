"""Tests for upgraded CHAID binning, out-of-bounds handling, and Bonferroni adjustment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cancer_mlops.chaid import ChaidClassifier, make_binning, merge_adjacent


def test_binning_out_of_bounds_maps_to_extremes_not_na() -> None:
    s = pd.Series(np.linspace(1.0, 100.0, 100), name="x")
    b = make_binning(s, n_bins=5)

    probe = pd.Series([-1e6, 0.5, 50.0, 105.0, 1e6, np.nan], name="x")
    codes = b.transform(probe)

    assert codes.iloc[0] == 0
    assert codes.iloc[1] == 0
    assert codes.iloc[3] == len(b.edges) - 2
    assert codes.iloc[4] == len(b.edges) - 2
    assert codes.iloc[5] == b.na_code


def test_bonferroni_increases_or_equals_raw_p() -> None:
    board = pd.DataFrame(
        {
            0.0: [40, 35, 25, 10, 5],
            1.0: [5, 10, 20, 35, 45],
        },
        index=[0, 1, 2, 3, 4],
    )
    _, p_raw = merge_adjacent(board, na_code=99, alpha=0.05, bonferroni=False)
    _, p_bonf = merge_adjacent(board, na_code=99, alpha=0.05, bonferroni=True)
    assert p_bonf >= p_raw
    assert 0.0 <= p_bonf <= 1.0


def test_chaid_predict_proba_shape_and_bounds() -> None:
    rng = np.random.default_rng(42)
    n = 200
    x1 = np.concatenate([rng.normal(10, 1, n // 2), rng.normal(20, 1, n // 2)])
    x2 = rng.choice(["low", "mid", "high"], size=n)
    y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
    df = pd.DataFrame({"x1": x1, "x2": x2})

    clf = ChaidClassifier(min_child=20, max_depth=3, bonferroni=True).fit(df, y)
    proba = clf.predict_proba(df)
    assert proba.shape == (n, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert ((proba >= 0.0) & (proba <= 1.0)).all()
    assert len(clf.rules()) >= 1
