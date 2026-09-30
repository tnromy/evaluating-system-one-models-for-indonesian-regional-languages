"""EXPLORATORY analyses (hypotheses.md "Exploratory"; labelled exploratory in the paper; spec in K71).

  lang_features.csv / lang_features_rho.csv : per regional language, Jev A1 macro-F1 and A1 minus B2 / B1-LR
      against genetic / geographic distance to Indonesian, CC100 / FLORES coverage, NLLB chrF (Spearman, descriptive).
  emotion_confusion.csv / emotion_recall.csv : NT-emot confusion (row-normalised by gold) and per-emotion recall
      for A1, D1, B2 per language (3 seeds pooled), C1 pooled (3 seeds pooled); overall and per language.
  reliability_by_lang.csv / reliability_bins.csv : ECE (width, mass), overconfidence per language, and reliability
      bins, for A1-choice and D1 (both model probabilities).
Output -> results/exploratory_test/
"""
import importlib.util

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from jevexp.data import DATASETS
from jevexp.eval import ece, reliability_bins
from jevexp.paths import RESULTS, ROOT, RUNS
from jevexp.runners.common import guard_split
from jevexp.schema import read

spec = importlib.util.spec_from_file_location("run_confirmatory", ROOT / "analysis" / "run_confirmatory.py")
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)

FEATURES = ["dist_genetic", "dist_geographic", "in_xlmr_cc100", "in_nllb_flores200", "nllb_chrf"]


def lang_features():
    m = pd.read_csv(RESULTS / "secondary_test" / "metrics_all_arms.csv")
    m = m[(m["lang"] != "ALL") & ~m["lang"].isin(rc.REGIONAL_EXCLUDE)]
    f1 = m.pivot_table(index=["dataset", "lang"], columns="arm", values="macro_f1")
    t = pd.DataFrame({"jev_a1_f1": f1["A1"], "a1_minus_b2": f1["A1"] - f1["B2-per_lang"],
                      "a1_minus_b1lr": f1["A1"] - f1["B1-LR-bow"]}).reset_index()
    feats = pd.read_csv(ROOT / "features" / "lang_features.csv")
    chrf = pd.concat([pd.read_csv(RUNS / "nllb" / f"{d}_test-full_stats.csv").assign(dataset=d) for d in DATASETS])
    t = t.merge(feats, on="lang", how="left").merge(chrf[["dataset", "lang", "chrf"]].rename(columns={"chrf": "nllb_chrf"}),
                                                     on=["dataset", "lang"], how="left")
    rows = []
    for d, g in [("ALL", t), *t.groupby("dataset")]:
        for target in ("jev_a1_f1", "a1_minus_b2", "a1_minus_b1lr"):
            for feat in FEATURES:
                x = g[[feat, target]].dropna()
                x = x.assign(**{feat: x[feat].astype(float)})
                rho, p = spearmanr(x[feat], x[target]) if len(x) >= 4 and x[feat].nunique() > 1 else (np.nan, np.nan)
                rows.append({"dataset": d, "target": target, "feature": feat, "n": len(x), "rho": rho,
                             "p_uncorrected": p})
    return t, pd.DataFrame(rows)


def seeds(pattern):
    return pd.concat([read(RUNS / pattern.format(s=s)) for s in (0, 1, 2)], ignore_index=True)


def emotions():
    d = "nt_emot"
    labels = list(DATASETS[d]["labels"])
    langs = list(DATASETS[d]["parallel_langs"])
    arms = {"A1": rc.load_arm("jev/{split}-A1-noul_en-{d}.parquet", "test", d),
            "D1": read(RUNS / f"deepseek/test-full-D1-deepseek-flash-nothink-{d}.parquet"),
            "B2-per_lang": pd.concat([seeds(f"B2/b2-{d}-{lang}-s{{s}}-nfull.parquet") for lang in langs],
                                     ignore_index=True),
            "C1-pooled": seeds(f"C1/c1-{d}-pooled-s{{s}}-nfull.parquet")}
    conf, rec = [], []
    for arm, df in arms.items():
        for lang, g in [("ALL", df), *df.groupby("lang")]:
            ct = pd.crosstab(g["gold"], g["pred"]).reindex(index=labels, columns=labels + ["INVALID"], fill_value=0)
            norm = ct.div(ct.sum(axis=1), axis=0)
            for gold in labels:
                rec.append({"arm": arm, "lang": lang, "emotion": gold, "n": int(ct.loc[gold].sum()),
                            "recall": float(norm.loc[gold, gold])})
                for pred in labels:
                    if pred != gold:
                        conf.append({"arm": arm, "lang": lang, "gold": gold, "pred": pred,
                                     "share_of_gold": float(norm.loc[gold, pred]), "count": int(ct.loc[gold, pred])})
    return pd.DataFrame(conf), pd.DataFrame(rec)


def reliability():
    rows, bins = [], []
    for d in DATASETS:
        labels = DATASETS[d]["labels"]
        arms = {"A1-choice": rc.load_arm("jev/{split}-A1-choice_en-{d}.parquet", "test", d),
                "D1": read(RUNS / f"deepseek/test-full-D1-deepseek-flash-nothink-{d}.parquet")}
        if d != "nusax_senti":
            arms["A1-choice"] = pd.concat([arms["A1-choice"],
                                           rc.load_arm("jev/{split}-A4-choice_en-{d}.parquet", "test", d)])
        for arm, df in arms.items():
            df = df[df["probs"].notna()]
            regional = df[df["lang"] != "ind"] if d != "nusax_senti" else df  # ALL = D1's languages (no A4 `ind`)
            for lang, g in [("ALL", regional), *df.groupby("lang")]:
                probs = np.stack(g["probs"].to_numpy())
                y = g["gold"].map({lab: i for i, lab in enumerate(labels)}).to_numpy()
                correct = (g["pred"] == g["gold"]).to_numpy()
                rows.append({"dataset": d, "arm": arm, "lang": lang, "n": len(g), "ece": ece(probs, y),
                             "ece_mass": ece(probs, y, scheme="mass"), "accuracy": float(correct.mean()),
                             "mean_confidence": float(probs.max(1).mean()),
                             "overconfidence": float(probs.max(1).mean() - correct.mean())})
                for b, (cnt, cf, acc) in enumerate(reliability_bins(probs, y)):
                    bins.append({"dataset": d, "arm": arm, "lang": lang, "bin": b, "count": cnt,
                                 "mean_confidence": cf, "accuracy": acc})
    return pd.DataFrame(rows), pd.DataFrame(bins)


def main():
    guard_split("test")
    out = RESULTS / "exploratory_test"
    out.mkdir(parents=True, exist_ok=True)
    feats, rho = lang_features()
    conf, rec = emotions()
    rel, bins = reliability()
    for name, df in [("lang_features", feats), ("lang_features_rho", rho), ("emotion_confusion", conf),
                     ("emotion_recall", rec), ("reliability_by_lang", rel), ("reliability_bins", bins)]:
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"== {name} ({len(df)} rows)")


if __name__ == "__main__":
    main()
