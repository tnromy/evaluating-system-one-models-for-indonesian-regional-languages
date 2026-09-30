"""H3b cascade simulation (analysis/run_cascade.py, K71). Synthetic data only."""
import importlib.util

import numpy as np
import pandas as pd
import pytest

from jevexp.eval import macro_f1
from jevexp.paths import ROOT

spec = importlib.util.spec_from_file_location("run_cascade", ROOT / "analysis" / "run_cascade.py")
rcs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rcs)

LABELS = ("negative", "positive")


def _frames(n=100, seed=0):
    rng = np.random.default_rng(seed)
    gold = rng.choice(LABELS, n)
    primary = pd.DataFrame({"lang": np.repeat(["a", "b"], n // 2), "sentence_id": [f"s{i % (n // 2):03d}" for i in range(n)],
                            "gold": gold, "pred": np.where(rng.random(n) < 0.7, gold, "negative"),
                            "confidence": np.round(rng.random(n), 1), "latency_ms": 300.0, "cost_usd": 1e-5})
    fb = primary[["lang", "sentence_id", "gold"]].assign(pred=gold, cost_usd=1e-4, latency_ms=800.0)
    return primary, fb


def test_extremes_are_primary_and_fallback_and_cost_is_linear():
    p, f = _frames()
    c = rcs.curve(p, [f], LABELS, f["cost_usd"].to_numpy(), f["latency_ms"].to_numpy(), by_lang=False).set_index("rate")
    assert c.loc[0.0, "macro_f1"] == pytest.approx(macro_f1(p["gold"], p["pred"], LABELS))
    assert c.loc[1.0, "macro_f1"] == pytest.approx(1.0)
    extra = c["cost_per_1000_usd"] - c.loc[0.0, "cost_per_1000_usd"]
    assert extra.to_numpy() == pytest.approx(c.index.to_numpy() * 1e-4 * 1000)
    assert c.loc[0.5, "latency_mean_ms"] == pytest.approx(300 + 0.5 * 800)


def test_routing_takes_lowest_confidence_first_and_is_stable():
    p, _ = _frames()
    m = rcs.routed_mask(p, 0.2)
    assert m.sum() == 20
    assert p["confidence"][m].max() <= p["confidence"][~m].min()
    shuffled = p.sample(frac=1, random_state=3).reset_index(drop=True)
    picked = set(map(tuple, p[m][["lang", "sentence_id"]].to_numpy()))
    assert picked == set(map(tuple, shuffled[rcs.routed_mask(shuffled, 0.2)][["lang", "sentence_id"]].to_numpy()))


def test_choose_rate_rule():
    c = pd.DataFrame({"rate": [0.0, 0.05, 0.1, 0.15], "lang": "ALL", "macro_f1": [0.70, 0.74, 0.80, 0.79]})
    assert rcs.choose_rate(c, 0.70, 0.78) == (0.1, True)
    assert rcs.choose_rate(c, 0.70, 0.85) == (0.1, False)  # target never reached -> argmax


def test_multi_seed_fallback_is_averaged():
    p, f = _frames()
    wrong = f.assign(pred=np.where(f["gold"] == "negative", "positive", "negative"))
    c = rcs.curve(p, [f, wrong], LABELS, 0.0, 10.0, by_lang=False).set_index("rate")
    assert c.loc[1.0, "accuracy"] == pytest.approx(0.5)
