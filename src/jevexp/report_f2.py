"""F2 pilot summaries (validation only): per-language macro-F1 per run, formulation selection rule, ops stats.

CLI:  python -m jevexp.report_f2
"""
import numpy as np
import pandas as pd

from .data import DATASETS
from .eval import INVALID, accuracy, ece, macro_f1
from .paths import RESULTS, RUNS
from .schema import read

OUT = RESULTS / "f2_pilot"
SELECTION_MARGIN = 1.0  # prereg/pilot_rules.md


def per_lang(df):
    labels = DATASETS[df["dataset"].iloc[0]]["labels"]
    rows = []
    for lang, g in df.groupby("lang"):
        row = {"lang": lang, "n": len(g), "macro_f1": 100 * macro_f1(g["gold"], g["pred"], labels),
               "acc": 100 * accuracy(g["gold"], g["pred"]), "invalid_pct": 100 * (g["pred"] == INVALID).mean()}
        ok = g[g["probs"].notna()]
        if len(ok):
            probs = np.stack(ok["probs"].to_numpy())
            y = ok["gold"].map({lab: i for i, lab in enumerate(labels)}).to_numpy()
            row["ece"] = 100 * ece(probs, y)
        rows.append(row)
    return pd.DataFrame(rows)


def ops(df):
    lat = df["latency_ms"].dropna()
    return {"n": len(df), "invalid_pct": 100 * (df["pred"] == INVALID).mean(),
            "latency_p50_ms": lat.median(), "latency_p95_ms": lat.quantile(0.95),
            "cost_usd": df["cost_usd"].fillna(0).sum() if "cost_usd" in df else np.nan,
            "cost_per_1000_usd": 1000 * df["cost_usd"].fillna(0).mean() if "cost_usd" in df else np.nan,
            "versions": ",".join(sorted(df["model_version"].dropna().unique())) if "model_version" in df else ""}


def select_formulation(scores):
    """scores: {formulation: mean macro-F1}. Rule from prereg/pilot_rules.md."""
    best = max(scores, key=scores.get)
    if best != "choice_en" and scores[best] - scores["choice_en"] <= SELECTION_MARGIN:
        return "choice_en"
    return best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary, tables = [], {}
    for path in sorted(RUNS.glob("*/valid-*.parquet")):
        df = read(path)
        pl = per_lang(df)
        tables[path.stem] = pl
        summary.append({"run": path.stem, "arm": df["arm"].iloc[0], "condition": df["condition"].iloc[0],
                        "dataset": df["dataset"].iloc[0], "mean_macro_f1": pl["macro_f1"].mean(),
                        "mean_ece": pl["ece"].mean() if "ece" in pl else np.nan, **ops(df)})
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "summary.csv", index=False)
    wide = pd.concat({k: v.set_index("lang")["macro_f1"] for k, v in tables.items()}, axis=1).round(1)
    wide.to_csv(OUT / "macro_f1_per_lang.csv")
    form = summary[(summary.arm == "A1") & (summary.dataset == "nusax_senti")]
    scores = {c.split("=")[1]: s for c, s in zip(form["condition"], form["mean_macro_f1"])}
    if {"choice_en", "choice_id", "noul_en"} <= set(scores):
        print("formulation scores (NusaX valid, mean macro-F1):", {k: round(v, 2) for k, v in scores.items()})
        print("SELECTED:", select_formulation(scores))
    print(summary.drop(columns=["versions"]).round(2).to_string(index=False))
    return summary, wide


if __name__ == "__main__":
    main()
