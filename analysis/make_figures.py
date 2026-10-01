"""Manuscript figures from the F6 result CSVs only (no analysis is recomputed). Output -> results/figures/{png,pdf}.
`--lang id` writes the Indonesian version (text in the figures, decimal comma) to results/figures_id/ for the
non-submitted Indonesian manuscript v1/id (K76); `--lang en` (default) is the submission version.
No in-figure footnotes: the manuscript captions carry the explanations (journal 010 F7).

Palette: first three categorical slots of the dataviz reference palette (validated all-pairs, light mode); each
series also has its own marker shape (secondary encoding: aqua is < 3:1 on the surface, and print is grayscale).
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import ScalarFormatter  # noqa: E402

from jevexp.paths import RESULTS  # noqa: E402

SURFACE, INK, INK2, GRID = "#ffffff", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]
DS_NAME = {"nusax_senti": "NusaX-senti", "nt_senti": "NusaTranslation-senti", "nt_emot": "NusaTranslation-emot"}
OUT = RESULTS / "figures"
LANG = "en"
T = {
    "en": {"lc_x": "training labels per language (n, log scale)", "lc_b1": "B1-LR (primary)", "lc_b2": "B2 IndoBERT",
           "lc_c1": "C1 Laya-ft", "lc_jev": "Jev A1 (zero-shot)",
           "fo_left": "Indonesian source minus regional text (A4 − A1)",
           "fo_right": "Jev minus classical baseline (A1 − B1-LR)", "fo_x": "accuracy difference (points)",
           "fr_title": " (subset)", "fr_x": "API cost per 1,000 decisions (USD, log)", "fr_strip": "GPT\n(cost n/r)",
           "ca_b2": "→ B2 IndoBERT (local)", "ca_c1": "→ C1 Laya-ft (local)", "ca_d1": "→ DeepSeek D1 (API)",
           "ca_x": "share of decisions routed to fallback (%)", "ca_op": "operating point chosen on validation",
           "re_jev": "Jev choice", "re_d1": "DeepSeek D1", "re_x": "mean top-class confidence", "re_y": "accuracy"},
    "id": {"lc_x": "jumlah label latih per bahasa (n, skala log)", "lc_b1": "B1-LR (primer)", "lc_b2": "B2 IndoBERT",
           "lc_c1": "C1 Laya-ft", "lc_jev": "Jev A1 (zero-shot)",
           "fo_left": "Kalimat sumber Indonesia dikurangi teks daerah (A4 − A1)",
           "fo_right": "Jev dikurangi baseline klasik (A1 − B1-LR)", "fo_x": "selisih akurasi (poin)",
           "fr_title": " (subset)", "fr_x": "biaya API per 1.000 keputusan (USD, log)", "fr_strip": "GPT\n(biaya n/a)",
           "ca_b2": "→ B2 IndoBERT (lokal)", "ca_c1": "→ C1 Laya-ft (lokal)", "ca_d1": "→ DeepSeek D1 (API)",
           "ca_x": "porsi keputusan yang di-routing ke fallback (%)", "ca_op": "titik operasi dipilih di validation",
           "re_jev": "Jev choice", "re_d1": "DeepSeek D1", "re_x": "rata-rata confidence kelas teratas",
           "re_y": "akurasi"},
}


def t(key):
    return T[LANG][key]


def num(x, d):
    """Number text in the figure language (decimal comma for id)."""
    s = f"{x:.{d}f}"
    return s.replace(".", ",") if LANG == "id" else s


def localise(ax):
    """Decimal comma on linear axes for the Indonesian figures (log axes keep 10^k labels)."""
    if LANG != "id":
        return
    if ax.get_yscale() == "linear":
        ax.yaxis.set_major_formatter(CommaFormatter())
    if ax.get_xscale() == "linear":
        ax.xaxis.set_major_formatter(CommaFormatter())


class CommaFormatter(ScalarFormatter):
    """matplotlib's default tick labels (same number of decimals on every tick), with a decimal comma."""

    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace(".", ",")

    def format_ticks(self, values):
        return [v.replace(".", ",") for v in super().format_ticks(values)]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "font.size": 8.5, "axes.titlesize": 9, "legend.frameon": False,
    "lines.linewidth": 1.5, "lines.markersize": 5.5, "lines.markeredgewidth": 1.0, "pdf.fonttype": 42,
    # Manuscript house style shared with journals 002/003 (SNCS): Times New Roman, 300 dpi.
    "font.family": "Times New Roman", "mathtext.fontset": "stix",
})
DPI = 300


def mark(i):
    return {"color": SERIES[i], "marker": MARKERS[i], "markeredgecolor": SURFACE}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=DPI, bbox_inches="tight")
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
        localise(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("macro-F1")
    fig.supxlabel(t("lc_x"), fontsize=8.5, y=0.02)
    handles = [plt.Line2D([], [], label=l, **mark(i)) for i, l in enumerate([t("lc_b1"), t("lc_b2"), t("lc_c1")])]
    handles.append(plt.Line2D([], [], color=INK2, linestyle=(0, (4, 3)), label=t("lc_jev")))
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04))
    save(fig, "fig1_learning_curves_h3a")


def fig_forest():
    h1b = pd.read_csv(RESULTS / "confirmatory_test" / "h1b_mcnemar.csv")
    f1 = pd.read_csv(RESULTS / "secondary_test" / "f1_family.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 6.2), sharey=False)
    for ax, df, col, title, lo, hi in [
            (a1, h1b, "acc_diff_a4_minus_a1", t("fo_left"), None, None),
            (a2, f1, "acc_diff_a1_minus_comparator", t("fo_right"), "ci_lo", "ci_hi")]:
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
        ax.set_xlabel(t("fo_x"))
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        localise(ax)
    fig.tight_layout()
    save(fig, "fig2_forest_h1b_f1")


def fig_frontier():
    """Cost–accuracy on the fixed subsets. Models without a reported cost (GPT via Codex CLI) sit in a narrow
    strip to the right of each panel, on the same y axis, instead of being dropped."""
    lad = pd.read_csv(RESULTS / "secondary_test" / "ladder.csv")
    priced = {"A1": "Jev", "D1": "DeepSeek V4.1 Flash", "D3-claude-haiku-4-5-20251001": "Claude Haiku 4.5",
              "D3-claude-sonnet-5": "Claude Sonnet 5", "D3-claude-opus-5-5": "Claude Opus 5.5"}
    unpriced = {"D4-gpt-6-luna": "GPT-6 Luna", "D4-gpt-6-sol": "GPT-6 Sol", "D4-gpt-5.6-terra": "GPT-5.6 Terra"}
    fig, axes = plt.subplots(1, 6, figsize=(10, 3.4), gridspec_kw={"width_ratios": [4, 1.5, 4, 1.5, 4, 1.5]})
    for k, d in enumerate(DS_NAME):
        ax, strip = axes[2 * k], axes[2 * k + 1]
        g = lad[lad["dataset"] == d]
        for arm, name in priced.items():
            r = g[g["arm"] == arm].iloc[0]
            i = 0 if arm == "A1" else (1 if arm == "D1" else 2)
            ax.plot(r["cost_per_1000_usd"], r["macro_f1"], linestyle="none", **mark(i))
            ax.annotate(name, (r["cost_per_1000_usd"], r["macro_f1"]), textcoords="offset points",
                        xytext=(5, -3), fontsize=6.5, color=INK)
        for j, (arm, name) in enumerate(unpriced.items()):
            r = g[g["arm"] == arm].iloc[0]
            strip.plot(0, r["macro_f1"], linestyle="none", marker="D", color=INK2, markeredgecolor=SURFACE)
            strip.annotate(name, (0, r["macro_f1"]), textcoords="offset points",
                           xytext=(5, -3), fontsize=6.5, color=INK)
        ys = g[g["arm"].isin(list(priced) + list(unpriced))]["macro_f1"]
        lo, hi = ys.min(), ys.max()
        pad = (hi - lo) * 0.08
        for a in (ax, strip):
            a.set_ylim(lo - pad, hi + pad)
        ax.set_xscale("log")
        ax.set_xlim(0.008, 40)
        ax.set_title(DS_NAME[d] + t("fr_title"))
        ax.set_xlabel(t("fr_x"))
        localise(ax)
        strip.set_xlim(-0.3, 2.2)
        strip.set_xticks([0], [t("fr_strip")], fontsize=7)
        strip.tick_params(axis="y", labelleft=False)
        strip.grid(axis="x", visible=False)
        strip.spines["left"].set_linestyle((0, (2, 2)))
    axes[0].set_ylabel("macro-F1")
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig3_cost_frontier")


def fig_cascade():
    c = pd.read_csv(RESULTS / "cascade_test" / "curve.csv")
    op = pd.read_csv(RESULTS / "cascade_test" / "operating_point.csv")
    c = c[c["lang"] == "ALL"]
    fbs = ["B2-per_lang", "C1-pooled", "D1"]
    label = {"B2-per_lang": t("ca_b2"), "C1-pooled": t("ca_c1"), "D1": t("ca_d1")}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
    for ax, d in zip(axes, DS_NAME):
        for i, fb in enumerate(fbs):
            g = c[(c["dataset"] == d) & (c["fallback"] == fb)].sort_values("rate")
            ax.plot(g["rate"] * 100, g["macro_f1"], label=label[fb], markevery=4, **mark(i))
            o = op[(op["dataset"] == d) & (op["fallback"] == fb)].iloc[0]
            ax.plot(o["r_star"] * 100, o["test_f1_cascade"], marker="D", markersize=8, color=SERIES[i],
                    markeredgecolor=INK, markeredgewidth=1.0, linestyle="none")
        ax.set_title(DS_NAME[d])
        ax.set_xlabel(t("ca_x"))
        localise(ax)
    axes[0].set_ylabel("macro-F1")
    h, lab = axes[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], marker="D", color=INK2, markeredgecolor=INK, linestyle="none", markersize=7))
    lab.append(t("ca_op"))
    fig.legend(h, lab, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08))
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
            ax.plot(g["mean_confidence"], g["accuracy"], label=f"{t('re_jev') if i == 0 else t('re_d1')} "
                    f"(ECE {num(e, 3)})", **mark(i))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(DS_NAME[d])
        ax.set_xlabel(t("re_x"))
        ax.legend(loc="upper left", fontsize=7)
        localise(ax)
    axes[0].set_ylabel(t("re_y"))
    fig.tight_layout()
    save(fig, "fig5_reliability")


def main():
    global LANG, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=sorted(T), default="en")
    LANG = ap.parse_args().lang
    OUT = RESULTS / ("figures" if LANG == "en" else f"figures_{LANG}")
    for f in (fig_learning_curves, fig_forest, fig_frontier, fig_cascade, fig_reliability):
        f()
        print("ok", f.__name__)


if __name__ == "__main__":
    main()
