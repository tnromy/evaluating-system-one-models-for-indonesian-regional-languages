"""Confirmatory analyses of prereg-v1 (hypotheses.md, analysis_plan.md).

  --split valid : DRY RUN on the F2 pilot files, only to prove the pipeline works (results not used).
  --split test  : the real analysis, after all MVP arms are collected (analysis_plan.md §7).

Post-tag additions (K67, written before any test metric was computed): test file prefix `test-full` (K50),
the pre-declared GEE fallback of H1a, and H3a (n*, pooled 3 seeds x B bootstrap draws).

Outputs -> results/confirmatory_<split>/
"""
import argparse
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

from jevexp.data import DATASETS
from jevexp.eval import aurc, cluster_bootstrap, ece, macro_f1
from jevexp.paths import RESULTS, ROOT, RUNS
from jevexp.runners.common import run_tag
from jevexp.schema import read

REGIONAL_EXCLUDE = {"ind", "eng"}


def load_arm(pattern, split, dataset):
    """Full-test files carry the `test-full` prefix since K50; the valid pilot files keep `valid`."""
    return read(RUNS / pattern.format(split=run_tag(split, "full"), d=dataset))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def mcnemar(a, b):
    """a, b: boolean correctness aligned on the same items. Exact two-sided test."""
    b01, b10 = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
    return b01, b10, p, float(a.mean() - b.mean())


def a4_frame(split, dataset, a1):
    if dataset == "nusax_senti":
        ind = a1[a1["lang"] == "ind"][["sentence_id", "pred", "gold"]]
    else:
        ind = load_arm("jev/{split}-A4-noul_en-{d}.parquet", split, dataset)[["sentence_id", "pred", "gold"]]
    reg = a1[~a1["lang"].isin(REGIONAL_EXCLUDE)][["sentence_id", "lang"]]
    return reg.merge(ind, on="sentence_id")


def h1b(split, datasets):
    rows = []
    for d in datasets:
        a1 = load_arm("jev/{split}-A1-noul_en-{d}.parquet", split, d)
        a4 = a4_frame(split, d, a1)
        m = a4.merge(a1, on=["sentence_id", "lang"], suffixes=("_a4", "_a1"))
        for lang, g in m.groupby("lang"):
            b01, b10, p, diff = mcnemar(g["pred_a4"] == g["gold_a4"], g["pred_a1"] == g["gold_a1"])
            rows.append({"dataset": d, "lang": lang, "n": len(g), "a4_only": b01, "a1_only": b10,
                         "acc_diff_a4_minus_a1": diff, "p": p})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"])
    return out


def h2(split, datasets, n_boot):
    rows = []
    for d in datasets:
        labels = DATASETS[d]["labels"]
        df = load_arm("jev/{split}-A1-choice_en-{d}.parquet", split, d)
        if d != "nusax_senti":  # NT has no `ind` language: the reference is Jev choice_en on the human Indonesian (A4)
            df = pd.concat([df, load_arm("jev/{split}-A4-choice_en-{d}.parquet", split, d)], ignore_index=True)
        df = df[df["probs"].notna()].reset_index(drop=True)
        df["y"] = df["gold"].map({lab: i for i, lab in enumerate(labels)})
        df["correct"] = (df["pred"] == df["gold"]).astype(float)
        df["regional"] = ~df["lang"].isin(REGIONAL_EXCLUDE)

        def delta(fn):
            def f(x):
                r, i = x[x["regional"]], x[x["lang"] == "ind"]
                return fn(r) - fn(i)
            return f

        e = lambda x: ece(np.stack(x["probs"].to_numpy()), x["y"].to_numpy())
        a = lambda x: aurc(x["confidence"].to_numpy(), x["correct"].to_numpy())
        over = lambda x: float(x["confidence"].mean() - x["correct"].mean())
        for name, fn in [("ECE", e), ("AURC", a), ("overconfidence", over)]:
            point, lo, hi = cluster_bootstrap(df, delta(fn), n_boot=n_boot, seed=20260924)
            rows.append({"dataset": d, "metric": f"delta_{name}_regional_minus_ind", "point": point, "ci_lo": lo,
                         "ci_hi": hi, "confirmed": lo > 0})
    return pd.DataFrame(rows)


def h1a(split, datasets, out_dir):
    ov = pd.read_parquet(ROOT / "features" / "sentence_overlap.parquet")
    frames = []
    for d in datasets:
        a1 = load_arm("jev/{split}-A1-noul_en-{d}.parquet", split, d)
        a1 = a1[~a1["lang"].isin(REGIONAL_EXCLUDE)]
        m = a1.merge(ov[(ov["dataset"] == d) & (ov["split"] == split)], on=["dataset", "lang", "sentence_id"])
        frames.append(m.assign(correct=(m["pred"] == m["gold"]).astype(int))[
            ["dataset", "lang", "sentence_id", "correct", "overlap_ind"]])
    inp, res = out_dir / "h1a_input.csv", out_dir / "h1a_glmm.csv"
    pd.concat(frames).to_csv(inp, index=False)
    r = subprocess.run(["Rscript", str(ROOT / "analysis" / "glmm.R"), str(inp), str(res)], capture_output=True, text=True)
    print(r.stdout.strip(), r.stderr.strip()[-500:])
    out = pd.read_csv(res)
    data = pd.read_csv(inp, dtype={"sentence_id": str})
    for i in out.index[out["model"] == "FALLBACK_GEE"]:
        out.loc[i, ["model", "beta", "ci_lo", "ci_hi", "converged"]] = gee_fit(data[data["dataset"] == out.at[i, "dataset"]])
    return out


def gee_fit(x):
    """Pre-declared fallback 2 (analysis_plan §4): binomial GEE, exchangeable, clusters = sentence_id, robust SE."""
    import statsmodels.api as sm
    x = x.assign(overlap_z=(x["overlap_ind"] - x["overlap_ind"].mean()) / x["overlap_ind"].std())
    x = x.sort_values("sentence_id", kind="stable")
    fit = sm.GEE.from_formula("correct ~ overlap_z", groups="sentence_id", data=x, family=sm.families.Binomial(),
                              cov_struct=sm.cov_struct.Exchangeable()).fit()
    lo, hi = fit.conf_int().loc["overlap_z"]
    return ["gee(correct ~ overlap_z, exchangeable | sentence_id)", fit.params["overlap_z"], lo, hi, bool(fit.converged)]


H3A_NS = (10, 25, 50, 100, 250, 500)
H3A_ARMS = {"B1-LR": "primary", "B2": "secondary", "C1": "secondary"}


def f1_draws(y, p, labels, idx):
    """Macro-F1 of every bootstrap draw (rows of idx), identical to eval.macro_f1 (labels only, zero_division=0)."""
    y, p = np.asarray(y)[idx], np.asarray(p)[idx]
    f1 = []
    for lab in labels:
        tp = ((y == lab) & (p == lab)).sum(1)
        denom = 2 * tp + ((y != lab) & (p == lab)).sum(1) + ((y == lab) & (p != lab)).sum(1)
        f1.append(np.divide(2 * tp, denom, out=np.zeros(len(tp)), where=denom > 0))
    return np.mean(f1, axis=0)


def curve_preds(arm, lang, n, seed, b1=None):
    """Test predictions (sentence_id-sorted) of one learning-curve run on NusaX, language `lang`."""
    if arm == "B1-LR":
        df = b1[(b1["condition"] == f"clf=lr;feat=bow;n={n};seed={seed}") & (b1["lang"] == lang)]
    else:
        df = read(RUNS / arm / f"{arm.lower()}-nusax_senti-{lang}-s{seed}-n{n}.parquet")
    return df.sort_values("sentence_id")[["sentence_id", "gold", "pred"]].reset_index(drop=True)


def nstar(target, curve, labels, n_boot=2000, seed=20260924, seeds=(0, 1, 2)):
    """H3a n*: smallest n whose lower 95 % bound reaches `target` (Jev A1 macro-F1), else '>500'.

    curve(n, seed) -> frame (sentence_id, gold, pred) with the same sentences in the same order for every seed.
    Bound = 2.5th percentile of the pooled draws (len(seeds) x n_boot); the resample indices are shared by the seeds.
    """
    rows = []
    for n in H3A_NS:
        frames = [curve(n, s) for s in seeds]
        ids = frames[0]["sentence_id"].to_numpy()
        assert all((f["sentence_id"].to_numpy() == ids).all() for f in frames), "seeds must cover the same items"
        idx = np.random.default_rng(seed).integers(0, len(ids), size=(n_boot, len(ids)))
        draws = np.concatenate([f1_draws(f["gold"], f["pred"], labels, idx) for f in frames])
        rows.append({"n": n, "f1_mean": float(np.mean([macro_f1(f["gold"], f["pred"], labels) for f in frames])),
                     "lb": float(np.quantile(draws, 0.025)), "target": target})
    rows = pd.DataFrame(rows)
    hit = rows[rows["lb"] >= rows["target"]]
    return (int(hit["n"].iloc[0]) if len(hit) else ">500"), rows


def h3a(split, n_boot):
    """H3a on NusaX regional languages. Learning curves exist on test only."""
    d = "nusax_senti"
    labels = DATASETS[d]["labels"]
    a1 = load_arm("jev/{split}-A1-noul_en-{d}.parquet", split, d)
    b1 = read(sorted(RUNS.glob("B1/B1curve-nusax_senti-*.parquet"))[-1])
    langs = [lang for lang in DATASETS[d]["parallel_langs"] if lang not in REGIONAL_EXCLUDE]
    curves, stars = [], []
    for arm in H3A_ARMS:
        for lang in langs:
            g = a1[a1["lang"] == lang]
            target = macro_f1(g["gold"], g["pred"], labels)
            ns, rows = nstar(target, lambda n, s: curve_preds(arm, lang, n, s, b1), labels, n_boot=n_boot)
            curves.append(rows.assign(arm=arm, lang=lang))
            stars.append({"arm": arm, "role": H3A_ARMS[arm], "lang": lang, "jev_f1": target, "n_star": ns})
    stars = pd.DataFrame(stars)
    return stars, pd.concat(curves, ignore_index=True), h3a_spearman(stars)


def h3a_spearman(stars):
    """Spearman rho(n*, Jev F1) > 0, one-sided; '>500' ranks above every observed n (ties averaged)."""
    rows = []
    for arm, g in stars.groupby("arm", sort=False):
        ns = g["n_star"].map(lambda v: np.inf if v == ">500" else float(v))
        rho, p = spearmanr(ns.rank(), g["jev_f1"], alternative="greater")
        rows.append({"arm": arm, "role": H3A_ARMS.get(arm, ""), "n_lang": len(g), "rho": rho, "p_one_sided": p,
                     "confirmed": bool(p < 0.05 and rho > 0)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["valid", "test"], required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    if a.split == "test":
        from jevexp.runners.common import guard_split
        guard_split("test")
    out = RESULTS / f"confirmatory_{a.split}"
    out.mkdir(parents=True, exist_ok=True)
    ds = list(DATASETS)
    results = [("h1a_glmm", h1a(a.split, ds, out)), ("h1b_mcnemar", h1b(a.split, ds)),
               ("h2_bootstrap", h2(a.split, ds, a.n_boot))]
    if a.split == "test":  # learning curves were predicted on test only
        stars, curves, rho = h3a(a.split, a.n_boot)
        results += [("h3a_nstar", stars), ("h3a_curves", curves), ("h3a_spearman", rho)]
    for name, df in results:
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"\n== {name}\n{df.round(4).to_string(index=False)}")


if __name__ == "__main__":
    main()
