"""F6 additions to analysis/run_confirmatory.py (K67): test-file prefix, GEE fallback, H3a n*. Synthetic data only."""
import importlib.util

import numpy as np
import pandas as pd
import pytest

from jevexp.eval import macro_f1
from jevexp.paths import ROOT

spec = importlib.util.spec_from_file_location("run_confirmatory", ROOT / "analysis" / "run_confirmatory.py")
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)

LABELS = ("negative", "neutral", "positive")


def test_load_arm_uses_test_full_prefix(monkeypatch):
    seen = []
    monkeypatch.setattr(rc, "read", lambda p: seen.append(p.name))
    rc.load_arm("jev/{split}-A1-noul_en-{d}.parquet", "test", "nt_emot")
    rc.load_arm("jev/{split}-A1-noul_en-{d}.parquet", "valid", "nt_emot")
    assert seen == ["test-full-A1-noul_en-nt_emot.parquet", "valid-A1-noul_en-nt_emot.parquet"]


def test_f1_draws_equal_macro_f1_including_invalid():
    rng = np.random.default_rng(3)
    y = rng.choice(LABELS, 200)
    p = rng.choice(LABELS + ("INVALID",), 200)
    idx = rng.integers(0, 200, size=(25, 200))
    got = rc.f1_draws(y, p, LABELS, idx)
    assert got == pytest.approx([macro_f1(y[i], p[i], LABELS) for i in idx])
    # a label absent from both gold and prediction contributes F1 = 0 (zero_division=0)
    one = np.zeros((1, 4), int) + np.arange(4)
    assert rc.f1_draws(["negative"] * 4, ["negative"] * 4, LABELS, one)[0] == pytest.approx(1 / 3)


def _curve(acc_by_n, n_items=300):
    """Curve whose accuracy at n is acc_by_n[n]; deterministic, identical items across seeds."""
    rng = np.random.default_rng(0)
    gold = rng.choice(LABELS, n_items)
    ids = [f"s{i:04d}" for i in range(n_items)]

    def curve(n, seed):
        r = np.random.default_rng(100 * n + seed)
        wrong = r.random(n_items) > acc_by_n[n]
        pred = np.where(wrong, np.roll(gold, 1), gold)
        return pd.DataFrame({"sentence_id": ids, "gold": gold, "pred": pred})
    return curve


def test_nstar_picks_smallest_n_whose_lower_bound_reaches_target():
    curve = _curve({10: 0.4, 25: 0.5, 50: 0.6, 100: 0.95, 250: 0.97, 500: 0.98})
    ns, rows = rc.nstar(0.8, curve, LABELS, n_boot=300)
    assert ns == 100
    assert (rows["lb"] <= rows["f1_mean"]).all()
    ns, _ = rc.nstar(0.99, curve, LABELS, n_boot=300)
    assert ns == ">500"


def test_nstar_pools_seed_draws():
    """With one bad seed the pooled 2.5th percentile falls far below the mean-based one."""
    good = _curve({n: 0.95 for n in rc.H3A_NS})
    bad = _curve({n: 0.5 for n in rc.H3A_NS})
    mixed = lambda n, s: bad(n, s) if s == 2 else good(n, s)
    _, rows = rc.nstar(0.0, mixed, LABELS, n_boot=300)
    draws_bad = rc.f1_draws(bad(10, 2)["gold"], bad(10, 2)["pred"], LABELS,
                            np.random.default_rng(20260924).integers(0, 300, size=(300, 300)))
    assert rows["lb"].iloc[0] < rows["f1_mean"].iloc[0] - 0.1
    # the bad seed holds 1/3 of the pooled draws, so the pooled 2.5 % point is its 7.5 % point
    assert rows["lb"].iloc[0] == pytest.approx(np.quantile(draws_bad, 0.075), abs=0.02)


def test_spearman_ranks_over_500_last():
    stars = pd.DataFrame({"arm": "B1-LR", "lang": list("abcde"), "jev_f1": [0.5, 0.6, 0.7, 0.8, 0.9],
                          "n_star": [10, 25, 50, 250, ">500"]})
    r = rc.h3a_spearman(stars).iloc[0]
    assert r["rho"] == pytest.approx(1.0) and r["confirmed"]
    stars["n_star"] = [">500", 250, 50, 25, 10]
    assert rc.h3a_spearman(stars).iloc[0]["rho"] == pytest.approx(-1.0)


def test_gee_fallback_recovers_positive_slope():
    rng = np.random.default_rng(5)
    n_sent, langs = 150, 6
    sid = np.repeat([f"s{i}" for i in range(n_sent)], langs)
    ov = rng.random(n_sent * langs)
    eta = -0.5 + 2.5 * (ov - ov.mean()) / ov.std() * 0.5 + np.repeat(rng.normal(0, 0.5, n_sent), langs)
    x = pd.DataFrame({"sentence_id": sid, "overlap_ind": ov,
                      "correct": (rng.random(len(ov)) < 1 / (1 + np.exp(-eta))).astype(int)})
    model, beta, lo, hi, conv = rc.gee_fit(x)
    assert model.startswith("gee(") and conv
    assert lo < beta < hi and lo > 0
