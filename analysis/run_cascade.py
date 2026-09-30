"""H3b (secondary): offline confidence-gated cascade Jev A1 -> fallback (docs/metodologi.md §8; spec locked in K71).

The lowest-confidence share r of A1's decisions (per dataset; ties broken by lang, sentence_id) goes to a fallback:
B2 per language, C1 pooled (local, no API cost; 3 seeds averaged) or D1 (LLM comparator, paid per routed item).
  curve.csv          : r = 0, 5, ..., 100 % on test (per dataset and per language), F1/accuracy/cost/latency.
  valid_curve.csv    : the same on the validation pilot items.
  operating_point.csv: r* chosen on validation = smallest r whose cascade F1 >= max(F1 A1, F1 fallback) there
                       (else the r with the highest cascade F1), applied to test; F1 difference to the best single
                       model of the pair with a cluster-bootstrap CI.
Output -> results/cascade_test/
"""
import importlib.util

import numpy as np
import pandas as pd

from jevexp.data import DATASETS
from jevexp.eval import accuracy, cluster_bootstrap, macro_f1
from jevexp.paths import RESULTS, ROOT, RUNS
from jevexp.runners.common import guard_split
from jevexp.schema import read

spec = importlib.util.spec_from_file_location("run_confirmatory", ROOT / "analysis" / "run_confirmatory.py")
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)

KEY = ["lang", "sentence_id"]
RATES = np.round(np.arange(0, 1.0001, 0.05), 2)
SEED = 20260924


def route_order(primary):
    """Row positions of `primary` from lowest to highest confidence (stable on lang, sentence_id)."""
    p = primary.reset_index(drop=True)
    return np.lexsort((p["sentence_id"].to_numpy(), p["lang"].to_numpy(), p["confidence"].to_numpy()))


def routed_mask(primary, rate):
    order = route_order(primary)
    m = np.zeros(len(primary), bool)
    m[order[:int(round(rate * len(primary)))]] = True
    return m


def cascade_preds(primary, fallback, mask):
    """Predictions of the cascade for one fallback seed (fallback aligned row-by-row with primary)."""
    return np.where(mask, fallback["pred"].to_numpy(), primary["pred"].to_numpy())


def align(primary, fallbacks):
    """Sort primary by KEY and keep only items every fallback seed has; fallbacks re-ordered to match."""
    keys = primary[KEY]
    for f in fallbacks:
        keys = keys.merge(f[KEY], on=KEY)
    keys = keys.sort_values(KEY).reset_index(drop=True)
    p = keys.merge(primary, on=KEY, how="left")
    return p, [keys.merge(f, on=KEY, how="left") for f in fallbacks]


def point(primary, fallbacks, mask, labels, fb_cost, fb_latency):
    """Metrics of one cascade setting; F1/accuracy averaged over fallback seeds."""
    gold = primary["gold"].to_numpy()
    preds = [cascade_preds(primary, f, mask) for f in fallbacks]
    lat = primary["latency_ms"].to_numpy(float) + np.where(mask, fb_latency, 0.0)
    cost = primary["cost_usd"].fillna(0).to_numpy(float) + np.where(mask, fb_cost, 0.0)
    return {"n": len(primary), "routed": int(mask.sum()),
            "macro_f1": float(np.mean([macro_f1(gold, p, labels) for p in preds])),
            "accuracy": float(np.mean([accuracy(gold, p) for p in preds])),
            "cost_per_1000_usd": float(cost.mean() * 1000), "latency_mean_ms": float(lat.mean()),
            "latency_p95_ms": float(np.quantile(lat, 0.95))}


def curve(primary, fallbacks, labels, fb_cost, fb_latency, by_lang=True):
    rows = []
    conf = primary["confidence"].to_numpy()
    order = route_order(primary)
    for r in RATES:
        mask = routed_mask(primary, r)
        k = int(mask.sum())
        tau = float(conf[order[k - 1]]) if k else np.nan
        rows.append({"rate": r, "lang": "ALL", "tau": tau, **point(primary, fallbacks, mask, labels, fb_cost, fb_latency)})
        if by_lang:
            for lang in primary["lang"].unique():
                sel = (primary["lang"] == lang).to_numpy()
                sub = [f[sel].reset_index(drop=True) for f in fallbacks]
                rows.append({"rate": r, "lang": lang, "tau": tau,
                             **point(primary[sel].reset_index(drop=True), sub, mask[sel], labels,
                                     fb_cost[sel] if np.ndim(fb_cost) else fb_cost,
                                     fb_latency[sel] if np.ndim(fb_latency) else fb_latency)})
    return pd.DataFrame(rows)


def choose_rate(valid_curve, f1_primary, f1_fallback):
    """r* = smallest rate whose cascade F1 reaches the better single model on validation; else the argmax."""
    c = valid_curve[valid_curve["lang"] == "ALL"].sort_values("rate")
    target = max(f1_primary, f1_fallback)
    hit = c[c["macro_f1"] >= target]
    return (float(hit["rate"].iloc[0]), True) if len(hit) else (float(c.loc[c["macro_f1"].idxmax(), "rate"]), False)


def local_latency(arm):
    lat = pd.read_csv(RESULTS / "latency_local.csv")
    lat = lat[(lat["arm"] == arm) & lat["p50_ms"].notna()]
    row = lat[lat["device"] == "cuda"] if (lat["device"] == "cuda").any() else lat
    return float(row["mean_ms"].iloc[0]), str(row["device"].iloc[0])


def seeds(pattern):
    return [read(RUNS / pattern.format(s=s)) for s in (0, 1, 2)]


def fallbacks(d, split):
    """name -> (list of seed frames, per-item API cost or 0, per-item latency, latency source)."""
    langs = list(DATASETS[d]["parallel_langs"])
    sub = "" if split == "test" else "valid/"
    b2 = [pd.concat([read(RUNS / f"B2/{sub}b2-{d}-{lang}-s{s}-nfull.parquet") for lang in langs]) for s in (0, 1, 2)]
    c1 = seeds(f"C1/{sub}c1-{d}-pooled-s{{s}}-nfull.parquet")
    d1 = read(RUNS / (f"deepseek/test-full-D1-deepseek-flash-nothink-{d}.parquet" if split == "test"
                      else f"deepseek/valid-D1-deepseek-flash-nothink-{d}.parquet"))
    (b2_ms, b2_dev), (c1_ms, c1_dev) = local_latency("B2"), local_latency("C1")
    return {"B2-per_lang": (b2, "local", b2_ms, f"laptop {b2_dev}"),
            "C1-pooled": (c1, "local", c1_ms, f"laptop {c1_dev}"),
            "D1": ([d1], "api", None, "measured per call")}


def build(d, split, name, fb):
    frames, kind, ms, _ = fb
    primary, frames = align(rc.load_arm("jev/{split}-A1-noul_en-{d}.parquet", split, d), frames)
    if kind == "api":
        cost, lat = frames[0]["cost_usd"].fillna(0).to_numpy(float), frames[0]["latency_ms"].to_numpy(float)
    else:
        cost, lat = 0.0, ms
    return primary, frames, cost, lat


def diff_ci(primary, frames, mask, labels, best):
    """Cascade F1 minus the best single model of the pair (seed-averaged), cluster bootstrap over sentence_id."""
    x = primary[["sentence_id", "gold"]].copy()
    for i, f in enumerate(frames):
        x[f"c{i}"] = cascade_preds(primary, f, mask)
        x[f"b{i}"] = primary["pred"].to_numpy() if best == "A1" else f["pred"].to_numpy()

    def fn(s):
        return float(np.mean([macro_f1(s["gold"], s[f"c{i}"], labels) - macro_f1(s["gold"], s[f"b{i}"], labels)
                              for i in range(len(frames))]))
    return cluster_bootstrap(x, fn, n_boot=2000, seed=SEED)


def main():
    guard_split("test")
    out = RESULTS / "cascade_test"
    out.mkdir(parents=True, exist_ok=True)
    curves, vcurves, ops = [], [], []
    for d in DATASETS:
        labels = DATASETS[d]["labels"]
        fb_test, fb_valid = fallbacks(d, "test"), fallbacks(d, "valid")
        for name in fb_test:
            vp, vf, vcost, vlat = build(d, "valid", name, fb_valid[name])
            vc = curve(vp, vf, labels, vcost, vlat, by_lang=False).assign(dataset=d, fallback=name)
            vcurves.append(vc)
            f1_a1_v = macro_f1(vp["gold"], vp["pred"], labels)
            f1_fb_v = float(np.mean([macro_f1(vp["gold"], f["pred"], labels) for f in vf]))
            r_star, reached = choose_rate(vc, f1_a1_v, f1_fb_v)

            tp, tf, tcost, tlat = build(d, "test", name, fb_test[name])
            tc = curve(tp, tf, labels, tcost, tlat).assign(dataset=d, fallback=name)
            curves.append(tc)
            allc = tc[tc["lang"] == "ALL"].set_index("rate")
            f1_a1 = macro_f1(tp["gold"], tp["pred"], labels)
            f1_fb = float(np.mean([macro_f1(tp["gold"], f["pred"], labels) for f in tf]))
            # cross-checks: r = 0 is A1 alone, r = 100 % is the fallback alone
            assert abs(allc.loc[0.0, "macro_f1"] - f1_a1) < 1e-12 and abs(allc.loc[1.0, "macro_f1"] - f1_fb) < 1e-12
            best = "A1" if f1_a1 >= f1_fb else name
            mask = routed_mask(tp, r_star)
            p, lo, hi = diff_ci(tp, tf, mask, labels, best)
            at = allc.loc[r_star]
            ops.append({"dataset": d, "fallback": name, "fallback_latency_source": fb_test[name][3],
                        "r_star": r_star, "reached_target_on_valid": reached, "valid_f1_a1": f1_a1_v,
                        "valid_f1_fallback": f1_fb_v, "valid_f1_cascade": vc.set_index("rate").loc[r_star, "macro_f1"],
                        "test_f1_a1": f1_a1, "test_f1_fallback": f1_fb, "test_f1_cascade": at["macro_f1"],
                        "best_single": best, "f1_diff_cascade_minus_best": p, "ci_lo": lo, "ci_hi": hi,
                        "equivalent_or_better": lo <= 0 <= hi or lo > 0,
                        "cost_per_1000_usd": at["cost_per_1000_usd"],
                        "cost_per_1000_usd_a1": allc.loc[0.0, "cost_per_1000_usd"],
                        "cost_per_1000_usd_all_routed": allc.loc[1.0, "cost_per_1000_usd"],
                        "latency_mean_ms": at["latency_mean_ms"], "latency_p95_ms": at["latency_p95_ms"]})
            print(ops[-1])
    for name, df in [("curve", pd.concat(curves)), ("valid_curve", pd.concat(vcurves)),
                     ("operating_point", pd.DataFrame(ops))]:
        df.to_csv(out / f"{name}.csv", index=False)


if __name__ == "__main__":
    main()
