"""Power analysis for prereg-v1 (K25), sized from VALIDATION pilot data only.

McNemar (exact, two-sided): for n paired items with discordance rate d and true accuracy difference D,
b ~ Bin(m, (d+D)/(2d)) given m ~ Bin(n, d) discordant pairs. Power is computed exactly by summing over m.
Minimum detectable difference (MDD) = smallest D with power >= 0.8 at alpha = 0.05 and at the worst Holm
alpha (0.05 / family size).

Run:  .venv/bin/python analysis/power.py   -> prereg/power_mcnemar.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import binom

from jevexp.paths import ROOT, RUNS
from jevexp.schema import read

TEST_N = {"nusax_senti": 400, "nt_senti": 1200, "nt_emot": 1000}
FAMILY = 26
PAIRS = {  # (arm A, arm B) from the pilot parquet files: discordance proxy for the F-families
    "A1_vs_D1": ("jev/valid-A1-noul_en-{d}.parquet", "deepseek/valid-D1-deepseek-flash-nothink-{d}.parquet"),
    "A4_vs_A1": ("jev/valid-A4-noul_en-{d}.parquet", "jev/valid-A1-noul_en-{d}.parquet"),
}


def mcnemar_crit(m, alpha):
    """Smallest k such that the exact two-sided test on m discordant pairs rejects when max(b, c) >= k."""
    for k in range(m // 2, m + 1):
        if 2 * binom.sf(k - 1, m, 0.5) <= alpha:
            return k
    return m + 1


def power(n, d, diff, alpha):
    if diff <= 0 or diff > d:
        return 0.0
    p = (d + diff) / (2 * d)
    total = 0.0
    for m in range(1, n + 1):
        pm = binom.pmf(m, n, d)
        if pm < 1e-12:
            continue
        k = mcnemar_crit(m, alpha)
        if k <= m:
            total += pm * (binom.sf(k - 1, m, p) + binom.cdf(m - k, m, p))
    return total


def mdd(n, d, alpha, target=0.8):
    for diff in np.arange(0.005, d + 1e-9, 0.005):
        if power(n, d, diff, alpha) >= target:
            return round(float(diff), 3)
    return np.nan


def discordance(pair, dataset):
    if pair == "A4_vs_A1" and dataset == "nusax_senti":  # NusaX A4 = Jev on the `ind` language itself
        b = read(RUNS / PAIRS[pair][1].format(d=dataset))
        a = b[b["lang"] == "ind"]
    else:
        a, b = (read(RUNS / p.format(d=dataset)) for p in PAIRS[pair])
    if pair == "A4_vs_A1":  # A4 is one row per sentence (lang='ind'); compare against every regional language
        a = a.drop(columns="lang").merge(b[["sentence_id", "lang"]], on="sentence_id")
        b = b[b["lang"] != "ind"]
    m = a.merge(b, on=["sentence_id", "lang"], suffixes=("_a", "_b"))
    m = m[~m["lang"].isin(["ind", "eng"])]
    ca, cb = m["pred_a"] == m["gold_a"], m["pred_b"] == m["gold_b"]
    return float((ca != cb).mean()), float(ca.mean() - cb.mean()), len(m)


def main():
    rows = []
    for pair in PAIRS:
        for dataset, n in TEST_N.items():
            try:
                d, obs, n_pilot = discordance(pair, dataset)
            except FileNotFoundError:
                continue
            rows.append({"pair": pair, "dataset": dataset, "pilot_items": n_pilot, "pilot_discordance": round(d, 3),
                         "pilot_diff": round(obs, 3), "test_n_per_lang": n,
                         "mdd_alpha_0.05": mdd(n, d, 0.05), f"mdd_alpha_holm_{FAMILY}": mdd(n, d, 0.05 / FAMILY)})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "prereg" / "power_mcnemar.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
