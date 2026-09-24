"""F1 gate report: our B1 replication vs NusaX Table 3 (docs/metodologi.md §5, K20, K32).

Variants per classifier:
  bow, tfidf       feature fixed (as the official notebook runs them)
  select_valid     feature chosen by validation accuracy (paper Table 9 lists feature as a hyperparameter)
Gate: mean |ΔF1| over 12 languages <= 2.0 points on `select_valid`.

CLI:  python -m jevexp.report_f1
"""
import pandas as pd

from .nusax_reference import LANGS, TABLE3
from .paths import RESULTS

GATE = 2.0
OUT = RESULTS / "f1_baseline"


def variants(scores):
    """Rows (lang, clf, variant, f1, feature) for the three variants."""
    rows = []
    for (lang, clf), g in scores.groupby(["lang", "clf"]):
        for _, r in g.iterrows():
            rows.append((lang, clf, r["feature"], 100 * r["test_macro_f1"], r["feature"]))
        best = g.sort_values(["valid_acc", "feature"], ascending=[False, True]).iloc[0]  # tie -> bow
        rows.append((lang, clf, "select_valid", 100 * best["test_macro_f1"], best["feature"]))
    return pd.DataFrame(rows, columns=["lang", "clf", "variant", "f1", "feature"])


def compare(scores):
    v = variants(scores)
    v["table3"] = [TABLE3[c][lang] for lang, c in zip(v["lang"], v["clf"])]
    v["delta"] = v["f1"] - v["table3"]
    summary = (v.groupby(["clf", "variant"])["delta"]
                .agg(mean_abs=lambda d: d.abs().mean(), mean=lambda d: d.mean(), max_abs=lambda d: d.abs().max())
                .reset_index())
    summary["gate"] = summary["mean_abs"] <= GATE
    return v, summary


def _markdown(v, summary):
    lines = ["# F1 — B1 replication vs NusaX Table 3 (macro-F1 %, test)", ""]
    lines += ["## Gate (mean |ΔF1| ≤ 2.0 on `select_valid`)", "",
              "| clf | variant | mean \\|Δ\\| | mean Δ | max \\|Δ\\| | pass |", "|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        lines.append(f"| {r.clf} | {r.variant} | {r.mean_abs:.2f} | {r['mean']:+.2f} | {r.max_abs:.2f} | "
                     f"{'✔' if r.gate else '✘'} |")
    for variant in ["select_valid", "bow", "tfidf"]:
        lines += ["", f"## Per language — `{variant}` (ours / Table 3)", "",
                  "| clf | " + " | ".join(LANGS) + " | avg |", "|---|" + "---|" * (len(LANGS) + 1)]
        for clf in ["nb", "svm", "lr"]:
            g = v[(v.clf == clf) & (v.variant == variant)].set_index("lang")
            cells = [f"{g.loc[lang, 'f1']:.1f} / {g.loc[lang, 'table3']:.1f}" for lang in LANGS]
            lines.append(f"| {clf} | " + " | ".join(cells) +
                         f" | {g['f1'].mean():.1f} / {g['table3'].mean():.1f} |")
    return "\n".join(lines) + "\n"


def main():
    scores = pd.read_csv(OUT / "scores_nusax_senti.csv")
    v, summary = compare(scores)
    v.to_csv(OUT / "comparison_nusax.csv", index=False)
    (OUT / "comparison.md").write_text(_markdown(v, summary))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
