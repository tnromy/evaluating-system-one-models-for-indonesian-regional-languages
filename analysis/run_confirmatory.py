"""Confirmatory analyses of prereg-v1 (hypotheses.md, analysis_plan.md).

  --split valid : DRY RUN on the F2 pilot files, only to prove the pipeline works (results not used).
  --split test  : the real analysis, after all MVP arms are collected (analysis_plan.md §7).

Outputs -> results/confirmatory_<split>/
"""
import argparse
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from jevexp.data import DATASETS
from jevexp.eval import aurc, cluster_bootstrap, ece
from jevexp.paths import RESULTS, ROOT, RUNS
from jevexp.schema import read

REGIONAL_EXCLUDE = {"ind", "eng"}


def load_arm(pattern, split, dataset):
    return read(RUNS / pattern.format(split=split, d=dataset))


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
    return pd.read_csv(res)


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
    for name, df in [("h1a_glmm", h1a(a.split, ds, out)), ("h1b_mcnemar", h1b(a.split, ds)),
                     ("h2_bootstrap", h2(a.split, ds, a.n_boot))]:
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"\n== {name}\n{df.round(4).to_string(index=False)}")


if __name__ == "__main__":
    main()
