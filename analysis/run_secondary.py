"""Secondary analyses of prereg-v1 (hypotheses.md "Secondary"; docs/metodologi.md §8). Test split only.

Written before any test metric was computed (K67). Reported with CIs and Holm within family; never used to claim
confirmation. H3b (cascade) and exploratory analyses are not part of this script.

  metrics_all_arms : every arm x dataset x language (+ ALL): macro-F1, accuracy, INVALID rate, ECE (width, mass),
                     Brier, AURC, latency, cost per 1,000 decisions. Multi-seed trained arms: mean over seeds.
  f1_family        : A1 vs B1-LR (LR-BoW, full training data), McNemar per language, Holm over 26 tests.
  f2_family        : A1 vs D1, McNemar per language, Holm over 26 tests.
  f2_calibration   : A1-choice vs D1 (both model probabilities), delta ECE / Brier, cluster bootstrap.
  h1c              : A3 - A1 per language with NLLB chrF (descriptive).
  ladder           : S-NX / S-NT / S-NE cost-accuracy table; McNemar A1 vs each LLM (dataset level), Holm.
  rq4_min          : noul_en vs choice_en vs choice_id on the subsets; Cohen's kappa between formulations.

Outputs -> results/secondary_test/
"""
import argparse
import importlib.util
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from jevexp.data import DATASETS
from jevexp.eval import INVALID, accuracy, aurc, brier, cluster_bootstrap, ece, macro_f1
from jevexp.paths import RESULTS, ROOT, RUNS
from jevexp.runners.common import guard_split, items
from jevexp.schema import read

spec = importlib.util.spec_from_file_location("run_confirmatory", ROOT / "analysis" / "run_confirmatory.py")
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)

SPLIT = "test"
SEED = 20260924
KEY = ["lang", "sentence_id"]


def latest(pattern):
    return read(sorted(RUNS.glob(pattern))[-1])


def seeds(pattern):
    """Concatenate the three seeds of a trained arm, tagged by `seed_run`."""
    return pd.concat([read(RUNS / pattern.format(s=s)).assign(seed_run=s) for s in (0, 1, 2)], ignore_index=True)


def arms(d):
    """Every test arm of dataset d: name -> (frame, scope). Loaded lazily by the callers that need them."""
    langs = list(DATASETS[d]["parallel_langs"])
    b1 = latest(f"B1/B1-{d}-*.parquet")
    b1 = b1[b1["lang"].isin(langs)]  # B1 was also trained on the non-parallel NT languages
    out = {
        "A1": (rc.load_arm("jev/{split}-A1-noul_en-{d}.parquet", SPLIT, d), "full"),
        "A1-choice": (rc.load_arm("jev/{split}-A1-choice_en-{d}.parquet", SPLIT, d), "full"),
        "A3": (rc.load_arm("jev/{split}-A3-noul_en-{d}.parquet", SPLIT, d), "full"),
        "D1": (read(RUNS / f"deepseek/test-full-D1-deepseek-flash-nothink-{d}.parquet"), "full"),
        "RQ4-choice_id": (read(RUNS / f"jev/test-subset-A1-choice_id-{d}.parquet"), "subset"),
        "C2": (read(RUNS / f"C2/c2-{d}-zeroshot.parquet"), "full" if d == "nusax_senti" else "subset"),
    }
    if d != "nusax_senti":
        out["A4"] = (rc.load_arm("jev/{split}-A4-noul_en-{d}.parquet", SPLIT, d), "full")
        out["A4-choice"] = (rc.load_arm("jev/{split}-A4-choice_en-{d}.parquet", SPLIT, d), "full")
    for cond in sorted(b1["condition"].unique()):
        clf, feat = (kv.split("=")[1] for kv in cond.split(";"))
        out[f"B1-{clf.upper()}-{feat}"] = (b1[b1["condition"] == cond], "full")
    for arm in ("B2", "B3", "C1"):  # regimes per metodologi §4 (e.g. C1 on NusaTranslation is pooled only)
        if (RUNS / arm / f"{arm.lower()}-{d}-{langs[0]}-s0-nfull.parquet").exists():
            per_lang = [seeds(f"{arm}/{arm.lower()}-{d}-{lang}-s{{s}}-nfull.parquet") for lang in langs]
            out[f"{arm}-per_lang"] = (pd.concat(per_lang, ignore_index=True), "full")
        if (RUNS / arm / f"{arm.lower()}-{d}-pooled-s0-nfull.parquet").exists():
            out[f"{arm}-pooled"] = (seeds(f"{arm}/{arm.lower()}-{d}-pooled-s{{s}}-nfull.parquet"), "full")
    for sub in ("claude", "codex"):
        for f in sorted((RUNS / sub).glob(f"test-subset-D*-{d}.parquet")):
            out[f.name.removeprefix("test-subset-").removesuffix(f"-{d}.parquet")] = (read(f), "subset")
    return out


def subset_keys(d):
    return pd.DataFrame([(i["lang"], i["sentence_id"]) for i in items(d, SPLIT, "subset")], columns=KEY)


def metrics(df, labels):
    """One row of metrics; multi-seed frames are averaged over `seed_run`."""
    if "seed_run" in df and df["seed_run"].nunique() > 1:
        rows = pd.DataFrame([metrics(g.drop(columns="seed_run"), labels) for _, g in df.groupby("seed_run")])
        num = rows.select_dtypes("number").mean()
        return {**num.to_dict(), "confidence_type": rows["confidence_type"].iloc[0], "n_seeds": len(rows)}
    y_idx = df["gold"].map({lab: i for i, lab in enumerate(labels)}).to_numpy()
    r = {"n": len(df), "macro_f1": macro_f1(df["gold"], df["pred"], labels), "accuracy": accuracy(df["gold"], df["pred"]),
         "invalid_rate": float((df["pred"] == INVALID).mean()), "confidence_type": df["confidence_type"].iloc[0],
         "n_seeds": 1}
    has = df["probs"].notna().to_numpy()
    if has.any():
        probs = np.stack(df["probs"][has].to_numpy())
        r.update(n_calib=int(has.sum()), ece=ece(probs, y_idx[has]), ece_mass=ece(probs, y_idx[has], scheme="mass"),
                 brier=brier(probs, y_idx[has]),
                 aurc=aurc(df["confidence"][has].to_numpy(), (df["pred"] == df["gold"])[has].to_numpy()))
    if "latency_ms" in df and df["latency_ms"].notna().any():
        r.update(latency_p50_ms=float(df["latency_ms"].median()), latency_p95_ms=float(df["latency_ms"].quantile(0.95)))
    if "cost_usd" in df and df["cost_usd"].notna().any():
        r["cost_per_1000_usd"] = float(df["cost_usd"].fillna(0).sum() / len(df) * 1000)
    return r


def metrics_table(all_arms):
    rows = []
    for d, arm_map in all_arms.items():
        labels = DATASETS[d]["labels"]
        for arm, (df, scope) in arm_map.items():
            for lang, g in [("ALL", df), *df.groupby("lang")]:
                rows.append({"dataset": d, "arm": arm, "scope": scope, "lang": lang, **metrics(g, labels)})
    return pd.DataFrame(rows)


def paired(a, b, name_b, family, n_boot):
    """McNemar per (dataset, language) of A1 vs b, effect = accuracy difference with bootstrap CI over sentences."""
    rows = []
    for (d, lang), g in a.merge(b, on=["dataset", *KEY], suffixes=("_a", "_b")).groupby(["dataset", "lang"]):
        ca, cb = g["pred_a"] == g["gold_a"], g["pred_b"] == g["gold_b"]
        b01, b10, p, diff = rc.mcnemar(ca, cb)
        x = pd.DataFrame({"sentence_id": g["sentence_id"], "d": ca.astype(float) - cb.astype(float)}).reset_index(drop=True)
        _, lo, hi = cluster_bootstrap(x, lambda s: s["d"].mean(), n_boot=n_boot, seed=SEED)
        rows.append({"family": family, "dataset": d, "lang": lang, "comparator": name_b, "n": len(g), "a1_only": b01,
                     "comparator_only": b10, "acc_diff_a1_minus_comparator": diff, "ci_lo": lo, "ci_hi": hi, "p": p})
    out = pd.DataFrame(rows)
    out["p_holm"] = rc.holm(out["p"])
    return out


def f2_calibration(all_arms, n_boot):
    rows = []
    for d, arm_map in all_arms.items():
        labels = DATASETS[d]["labels"]
        m = arm_map["A1-choice"][0].merge(arm_map["D1"][0], on=KEY, suffixes=("_j", "_d"))
        m = m[m["probs_j"].notna() & m["probs_d"].notna()].reset_index(drop=True)
        y = m["gold_j"].map({lab: i for i, lab in enumerate(labels)}).to_numpy()
        m["y"] = y
        for name, fn in [("ECE", ece), ("Brier", brier)]:
            def delta(x, fn=fn):
                return fn(np.stack(x["probs_j"].to_numpy()), x["y"].to_numpy()) - \
                    fn(np.stack(x["probs_d"].to_numpy()), x["y"].to_numpy())
            point, lo, hi = cluster_bootstrap(m, delta, n_boot=n_boot, seed=SEED)
            rows.append({"dataset": d, "metric": f"delta_{name}_A1choice_minus_D1", "n": len(m), "point": point,
                         "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def h1c(all_arms):
    rows = []
    for d, arm_map in all_arms.items():
        labels = DATASETS[d]["labels"]
        chrf = pd.read_csv(RUNS / "nllb" / f"{d}_test-full_stats.csv").set_index("lang")["chrf"]
        m = arm_map["A3"][0].merge(arm_map["A1"][0], on=KEY, suffixes=("_a3", "_a1"))
        for lang, g in m.groupby("lang"):
            b01, b10, p, diff = rc.mcnemar(g["pred_a3"] == g["gold_a3"], g["pred_a1"] == g["gold_a1"])
            rows.append({"dataset": d, "lang": lang, "n": len(g), "nllb_chrf": chrf.get(lang, np.nan),
                         "acc_diff_a3_minus_a1": diff,
                         "f1_diff_a3_minus_a1": macro_f1(g["gold_a3"], g["pred_a3"], labels)
                         - macro_f1(g["gold_a1"], g["pred_a1"], labels), "p_uncorrected": p})
    return pd.DataFrame(rows)


LLM = ("A1", "A1-choice", "D1")


def ladder(all_arms):
    """Every arm on the fixed subsets (identical items); McNemar A1 vs each LLM arm, pooled over languages."""
    rows, tests = [], []
    for d, arm_map in all_arms.items():
        labels = DATASETS[d]["labels"]
        keys = subset_keys(d)
        a1 = arm_map["A1"][0].merge(keys, on=KEY)
        for arm, (df, _) in arm_map.items():
            s = df.merge(keys, on=KEY)
            if s.empty:
                continue
            rows.append({"dataset": d, "arm": arm, **metrics(s, labels)})
            if arm != "A1" and (arm in LLM or arm.startswith("D")):
                m = a1.merge(s, on=KEY, suffixes=("_a", "_b"))
                b01, b10, p, diff = rc.mcnemar(m["pred_a"] == m["gold_a"], m["pred_b"] == m["gold_b"])
                tests.append({"dataset": d, "comparator": arm, "n": len(m), "a1_only": b01, "comparator_only": b10,
                              "acc_diff_a1_minus_comparator": diff, "p": p})
    tests = pd.DataFrame(tests)
    tests["p_holm"] = rc.holm(tests["p"])
    return pd.DataFrame(rows), tests


def rq4_min(all_arms):
    rows = []
    forms = {"noul_en": "A1", "choice_en": "A1-choice", "choice_id": "RQ4-choice_id"}
    for d, arm_map in all_arms.items():
        labels = DATASETS[d]["labels"]
        keys = subset_keys(d)
        f = {k: arm_map[a][0].merge(keys, on=KEY).sort_values(KEY).reset_index(drop=True) for k, a in forms.items()}
        for k, g in f.items():
            rows.append({"dataset": d, "formulation": k, "n": len(g), "macro_f1": macro_f1(g["gold"], g["pred"], labels),
                         "accuracy": accuracy(g["gold"], g["pred"])})
        for x, y in combinations(f, 2):
            assert (f[x][KEY].to_numpy() == f[y][KEY].to_numpy()).all()
            rows.append({"dataset": d, "formulation": f"kappa {x} vs {y}", "n": len(f[x]),
                         "kappa": cohen_kappa_score(f[x]["pred"], f[y]["pred"])})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    guard_split(SPLIT)
    out = RESULTS / "secondary_test"
    out.mkdir(parents=True, exist_ok=True)
    all_arms = {d: arms(d) for d in DATASETS}
    cat = lambda name: pd.concat([m[name][0].assign(dataset=d) for d, m in all_arms.items()], ignore_index=True)
    lad, lad_tests = ladder(all_arms)
    results = [("metrics_all_arms", metrics_table(all_arms)),
               ("f1_family", paired(cat("A1"), cat("B1-LR-bow"), "B1-LR-bow", "F1", a.n_boot)),
               ("f2_family", paired(cat("A1"), cat("D1"), "D1", "F2", a.n_boot)),
               ("f2_calibration", f2_calibration(all_arms, a.n_boot)),
               ("h1c", h1c(all_arms)), ("ladder", lad), ("ladder_mcnemar", lad_tests), ("rq4_min", rq4_min(all_arms))]
    for name, df in results:
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"\n== {name} ({len(df)} rows) -> {out / name}.csv")


if __name__ == "__main__":
    main()
