import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

from jevexp.eval import (INVALID, accuracy, aurc, brier, cluster_bootstrap, coverage_accuracy_curve, ece,
                         macro_f1, reliability_bins)

LABELS = ("negative", "neutral", "positive")


def test_accuracy_and_macro_f1_match_sklearn():
    rng = np.random.default_rng(1)
    y = rng.choice(LABELS, 300)
    p = rng.choice(LABELS, 300)
    assert accuracy(y, p) == pytest.approx(accuracy_score(y, p))
    assert macro_f1(y, p, LABELS) == pytest.approx(f1_score(y, p, labels=list(LABELS), average="macro"))


def test_invalid_counts_as_wrong():
    y = ["negative", "neutral", "positive", "positive"]
    p = ["negative", INVALID, "positive", INVALID]
    assert accuracy(y, p) == 0.5
    # neutral: tp 0 -> F1 0; negative F1 1; positive: tp1 fn1 fp0 -> 2/3
    assert macro_f1(y, p, LABELS) == pytest.approx((1 + 0 + 2 / 3) / 3)


def test_ece_zero_for_perfectly_calibrated_one_hot():
    probs = np.eye(3)[[0, 1, 2, 0]]
    assert ece(probs, [0, 1, 2, 0]) == 0.0
    assert ece(probs, [0, 1, 2, 0], scheme="mass") == 0.0


def test_ece_hand_computed():
    # confidences 0.9 (correct), 0.9 (wrong) in the same bin -> |0.5 - 0.9| = 0.4
    probs = np.array([[0.9, 0.1], [0.9, 0.1]])
    assert ece(probs, [0, 1]) == pytest.approx(0.4)
    # separate bins: 0.6 correct (gap 0.4), 0.95 correct (gap 0.05) -> mean 0.225
    probs = np.array([[0.6, 0.4], [0.95, 0.05]])
    assert ece(probs, [0, 0]) == pytest.approx(0.225)


def test_reliability_bins_right_closed_edges():
    # 15 equal-width bins: 0.6 = 9/15 is an edge, so it closes bin (8/15, 9/15]; 0.61 falls in the next bin.
    probs = np.array([[0.6, 0.4], [0.61, 0.39]])
    bins = reliability_bins(probs, [0, 1])
    assert [c for c, _, _ in bins] == [1, 1]
    # separate bins -> (|1-0.6| + |0-0.61|) / 2 ; merged would give |0.5-0.605| = 0.105
    assert ece(probs, [0, 1]) == pytest.approx(0.505)


def test_brier():
    assert brier(np.eye(3)[[0, 1]], [0, 1]) == 0.0
    assert brier(np.array([[0.5, 0.5]]), [0]) == pytest.approx(0.5)
    assert brier(np.array([[1 / 3] * 3]), [2]) == pytest.approx(2 / 3)


def test_coverage_curve_and_aurc():
    conf = [0.9, 0.8, 0.7, 0.6]
    cov, acc = coverage_accuracy_curve(conf, [1, 1, 0, 0])
    assert cov.tolist() == [0.25, 0.5, 0.75, 1.0]
    assert acc.tolist() == pytest.approx([1, 1, 2 / 3, 0.5])
    assert aurc(conf, [1, 1, 1, 1]) == 0.0
    # confident errors are worst: risks 1, 1, 2/3, 1/2
    assert aurc(conf, [0, 0, 1, 1]) == pytest.approx((1 + 1 + 2 / 3 + 0.5) / 4)


def test_cluster_bootstrap_deterministic_and_brackets_point():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"sentence_id": np.repeat(np.arange(50), 3), "correct": rng.integers(0, 2, 150)})
    fn = lambda d: d["correct"].mean()
    a = cluster_bootstrap(df, fn, n_boot=500, seed=7)
    b = cluster_bootstrap(df, fn, n_boot=500, seed=7)
    assert a == b
    point, lo, hi = a
    assert lo <= point <= hi


def test_cluster_bootstrap_resamples_whole_clusters():
    # every cluster is internally constant, so any resample keeps within-cluster values together
    df = pd.DataFrame({"sentence_id": [0, 0, 1, 1], "v": [1.0, 1.0, 0.0, 0.0]})
    _, lo, hi = cluster_bootstrap(df, lambda d: d["v"].mean(), n_boot=200, seed=1)
    assert {lo, hi} <= {0.0, 0.5, 1.0}
