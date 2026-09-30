"""Manuscript figures from the F6 result CSVs only (no analysis is recomputed). Output -> results/figures/{png,pdf}.

Palette: first three categorical slots of the dataviz reference palette (validated all-pairs, light mode); each
series also has its own marker shape (secondary encoding: aqua is < 3:1 on the surface, and print is grayscale).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from jevexp.paths import RESULTS  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]
DS_NAME = {"nusax_senti": "NusaX-senti", "nt_senti": "NusaTranslation-senti", "nt_emot": "NusaTranslation-emot"}
OUT = RESULTS / "figures"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "font.size": 8.5, "axes.titlesize": 9, "legend.frameon": False,
    "lines.linewidth": 1.5, "lines.markersize": 5.5, "lines.markeredgewidth": 1.0, "pdf.fonttype": 42,
})


def mark(i):
    return {"color": SERIES[i], "marker": MARKERS[i], "markeredgecolor": SURFACE}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_learning_curves():
    c = pd.read_csv(RESULTS / "confirmatory_test" / "h3a_curves.csv")
    c = c[~((c["arm"] == "C1") & (c["n"] <= 50))]  # C1 did not train at n <= 50 (K68)
    langs = sorted(c["lang"].unique())
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.4), sharex=True, sharey=True)
    for ax, lang in zip(axes.flat, langs):
        g = c[c["lang"] == lang]
        for i, arm in enumerate(["B1-LR", "B2", "C1"]):
            a = g[g["arm"] == arm].sort_values("n")
            ax.plot(a["n"], a["f1_mean"], label=arm, **mark(i))
            if arm == "B1-LR":
                ax.fill_between(a["n"], a["lb"], a["f1_mean"], color=SERIES[0], alpha=0.12, linewidth=0)
        ax.axhline(g["target"].iloc[0], color=INK2, linestyle=(0, (4, 3)), linewidth=1.2)
        ax.set_title(lang)
        ax.set_xscale("log")
        ax.set_xticks([10, 25, 50, 100, 250, 500], ["10", "25", "50", "100", "250", "500"], rotation=45, fontsize=7)
        ax.minorticks_off()
    for ax in axes[:, 0]:
        ax.set_ylabel("macro-F1")
    fig.supxlabel("training labels per language (n, log scale)", fontsize=8.5, y=0.02)
    handles = [plt.Line2D([], [], label=l, **mark(i)) for i, l in enumerate(["B1-LR (primary)", "B2 IndoBERT", "C1 Laya-ft"])]
    handles.append(plt.Line2D([], [], color=INK2, linestyle=(0, (4, 3)), label="Jev A1 (zero-shot)"))
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04))
    fig.text(0.5, -0.04, "Shaded: B1-LR mean down to the pooled 95 % lower bound (3 seeds x 2,000 draws). "
             "C1 points at n <= 50 omitted (no effective training, K68).", ha="center", color=INK2, fontsize=7.5)
    save(fig, "fig1_learning_curves_h3a")


def fig_forest():
    h1b = pd.read_csv(RESULTS / "confirmatory_test" / "h1b_mcnemar.csv")
    f1 = pd.read_csv(RESULTS / "secondary_test" / "f1_family.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 6.2), sharey=False)
    for ax, df, col, title, lo, hi in [
            (a1, h1b, "acc_diff_a4_minus_a1", "H1b: A4 (human Indonesian) minus A1", None, None),
            (a2, f1, "acc_diff_a1_minus_comparator", "F1 family: Jev A1 minus B1-LR", "ci_lo", "ci_hi")]:
        df = df.sort_values(["dataset", "lang"], key=lambda s: s.map({d: i for i, d in enumerate(DS_NAME)})
                            if s.name == "dataset" else s).reset_index(drop=True)
        y = np.arange(len(df))[::-1]
        for i, d in enumerate(DS_NAME):
            m = (df["dataset"] == d).to_numpy()
            sig = (df["p_holm"] < 0.05).to_numpy()
            if lo:
                ax.hlines(y[m], df[lo][m] * 100, df[hi][m] * 100, color=SERIES[i], linewidth=1.5)
            ax.plot(df[col][m & sig] * 100, y[m & sig], linestyle="none", **mark(i))
            ax.plot(df[col][m & ~sig] * 100, y[m & ~sig], linestyle="none", marker=MARKERS[i],
                    markerfacecolor=SURFACE, markeredgecolor=SERIES[i], markeredgewidth=1.3)
        ax.axvline(0, color=INK2, linewidth=1)
        ax.set_yticks(y, [f"{DS_NAME[d].replace('NusaTranslation', 'NT')} · {l}" for d, l in zip(df["dataset"], df["lang"])],
                      fontsize=7)
        ax.set_xlabel("accuracy difference (points)")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    fig.text(0.5, -0.02, "Filled marker: Holm-adjusted p < 0.05 (exact McNemar); hollow: not significant. "
             "Right panel bars: 95 % cluster-bootstrap CI.", ha="center", color=INK2, fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig2_forest_h1b_f1")


def fig_frontier():
    lad = pd.read_csv(RESULTS / "secondary_test" / "ladder.csv")
    lad = lad[lad["cost_per_1000_usd"].notna()]
    names = {"A1": "Jev", "A1-choice": None, "RQ4-choice_id": None, "A3": None, "D1": "DeepSeek flash",
             "D3-claude-haiku-4-5-20251001": "Claude Haiku 4.5", "D3-claude-sonnet-5": "Claude Sonnet 5",
             "D3-claude-opus-5-5": "Claude Opus 5.5"}
    lad = lad[lad["arm"].map(names).notna()]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), sharey=False)
    for ax, (d, g) in zip(axes, [(d, lad[lad["dataset"] == d]) for d in DS_NAME]):
        for _, r in g.iterrows():
            i = 0 if r["arm"] == "A1" else (1 if r["arm"] == "D1" else 2)
            ax.plot(r["cost_per_1000_usd"], r["macro_f1"], linestyle="none", **mark(i))
            ax.annotate(names[r["arm"]], (r["cost_per_1000_usd"], r["macro_f1"]), textcoords="offset points",
                        xytext=(6, -3), fontsize=7, color=INK)
        ax.set_xscale("log")
        ax.set_xlim(0.008, 40)
        ax.margins(y=0.08)
        ax.set_title(DS_NAME[d] + " (subset)")
        ax.set_xlabel("API cost per 1,000 decisions (USD, log)")
    axes[0].set_ylabel("macro-F1")
    fig.text(0.5, -0.06, "Claude cost = API-equivalent cost reported by Claude Code. Codex models report no cost "
             "and are omitted here (see table).", ha="center", color=INK2, fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig3_cost_frontier")


def fig_cascade():
    c = pd.read_csv(RESULTS / "cascade_test" / "curve.csv")
    op = pd.read_csv(RESULTS / "cascade_test" / "operating_point.csv")
    c = c[c["lang"] == "ALL"]
    fbs = ["B2-per_lang", "C1-pooled", "D1"]
    label = {"B2-per_lang": "→ B2 IndoBERT (local)", "C1-pooled": "→ C1 Laya-ft (local)", "D1": "→ DeepSeek D1 (API)"}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
    for ax, d in zip(axes, DS_NAME):
        for i, fb in enumerate(fbs):
            g = c[(c["dataset"] == d) & (c["fallback"] == fb)].sort_values("rate")
            ax.plot(g["rate"] * 100, g["macro_f1"], label=label[fb], markevery=4, **mark(i))
            o = op[(op["dataset"] == d) & (op["fallback"] == fb)].iloc[0]
            ax.plot(o["r_star"] * 100, o["test_f1_cascade"], marker="D", markersize=8, color=SERIES[i],
                    markeredgecolor=INK, markeredgewidth=1.0, linestyle="none")
        ax.set_title(DS_NAME[d])
        ax.set_xlabel("share of decisions routed to fallback (%)")
    axes[0].set_ylabel("macro-F1")
    h, lab = axes[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], marker="D", color=INK2, markeredgecolor=INK, linestyle="none", markersize=7))
    lab.append("operating point chosen on validation")
    fig.legend(h, lab, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08))
    fig.text(0.5, -0.06, "Jev A1 answers every item; its lowest-confidence share is re-answered by the fallback. "
             "Local fallbacks: 3 seeds averaged.", ha="center", color=INK2, fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig4_cascade_h3b")


def fig_reliability():
    b = pd.read_csv(RESULTS / "exploratory_test" / "reliability_bins.csv")
    rel = pd.read_csv(RESULTS / "exploratory_test" / "reliability_by_lang.csv")
    b = b[b["lang"] == "ALL"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for ax, d in zip(axes, DS_NAME):
        ax.plot([0, 1], [0, 1], color=INK2, linestyle=(0, (4, 3)), linewidth=1)
        for i, arm in enumerate(["A1-choice", "D1"]):
            g = b[(b["dataset"] == d) & (b["arm"] == arm)]
            g = g[g["count"] >= 10]  # bins with < 10 items are too noisy to draw
            e = rel[(rel["dataset"] == d) & (rel["arm"] == arm) & (rel["lang"] == "ALL")]["ece"].iloc[0]
            ax.plot(g["mean_confidence"], g["accuracy"], label=f"{'Jev choice' if i == 0 else 'DeepSeek D1'} "
                    f"(ECE {e:.3f})", **mark(i))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(DS_NAME[d])
        ax.set_xlabel("mean top-class confidence")
        ax.legend(loc="upper left", fontsize=7)
    axes[0].set_ylabel("accuracy")
    fig.text(0.5, -0.04, "15 equal-width bins; bins with fewer than 10 items not drawn. Dashed: perfect calibration. "
             "Same items for both models (NusaX: 12 languages; NT: 7 regional).",
             ha="center", color=INK2, fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig5_reliability")


def main():
    for f in (fig_learning_curves, fig_forest, fig_frontier, fig_cascade, fig_reliability):
        f()
        print("ok", f.__name__)


if __name__ == "__main__":
    main()
