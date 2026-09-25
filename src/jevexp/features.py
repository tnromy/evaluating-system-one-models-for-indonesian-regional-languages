"""Covariates for H1 (docs/metodologi.md §8; prereg/hypotheses.md).

Sentence level:  overlap_ind = sentence chrF between the regional sentence and its human Indonesian parallel.
Language level:  lang2vec genetic ('fam') and geographic ('geo') cosine distance to Indonesian; pretraining
                 coverage indicators (XLM-R CC-100, J16; NLLB FLORES-200, J55); NLLB chrF on validation (F2).
Joshi et al. classes (J18) and mmBERT coverage (J06) are not used: per-language assignments are not verifiable
from the papers themselves.

CLI:  python -m jevexp.features
"""
import warnings

import numpy as np
import pandas as pd
from sacrebleu.metrics import CHRF

from .data import DATASETS, SPLITS, ind_source, load
from .paths import ROOT, RUNS

FEATURES = ROOT / "features"
# NusaWrites uses the collective code `btk` (Batak); its texts are Toba Batak, so URIEL features come from `bbc`.
URIEL_CODE = {"btk": "bbc"}
CC100 = {"ind", "jav", "sun", "eng"}                                     # XLM-R pretraining languages (J16 Table 6)
FLORES200 = {"ace", "ban", "bjn", "bug", "jav", "min", "sun", "ind", "eng"}  # NLLB-200 languages (J55)


def sentence_overlap():
    chrf = CHRF()
    rows = []
    for dataset in DATASETS:
        ref = ind_source(dataset)
        df = load(dataset, langs=DATASETS[dataset]["parallel_langs"])
        for r in df.itertuples():
            src = ref.get(r.sentence_id)
            rows.append({"dataset": dataset, "split": r.split, "lang": r.lang, "sentence_id": r.sentence_id,
                         "overlap_ind": chrf.sentence_score(r.text, [src]).score if src is not None else np.nan})
    return pd.DataFrame(rows)


def _cosine_distance(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def language_features():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import lang2vec.lang2vec as l2v
    langs = sorted({lang for d in DATASETS.values() for lang in d["parallel_langs"]})
    codes = {lang: URIEL_CODE.get(lang, lang) for lang in langs}
    feats = {fs: l2v.get_features(sorted(set(codes.values())), fs) for fs in ("fam", "geo")}
    nllb = pd.concat([pd.read_csv(p) for p in RUNS.glob("nllb/*_valid_stats.csv")]) if RUNS.exists() else None
    rows = []
    for lang in langs:
        c = codes[lang]
        row = {"lang": lang, "uriel_code": c,
               "dist_genetic": _cosine_distance(feats["fam"][c], feats["fam"]["ind"]),
               "dist_geographic": _cosine_distance(feats["geo"][c], feats["geo"]["ind"]),
               "in_xlmr_cc100": lang in CC100, "in_nllb_flores200": lang in FLORES200}
        if nllb is not None and lang in set(nllb["lang"]):
            row["nllb_chrf_valid"] = nllb.loc[nllb["lang"] == lang, "chrf"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    FEATURES.mkdir(parents=True, exist_ok=True)
    s = sentence_overlap()
    s.to_parquet(FEATURES / "sentence_overlap.parquet", index=False)
    lf = language_features()
    lf.to_csv(FEATURES / "lang_features.csv", index=False)
    print(s.groupby(["dataset", "lang"])["overlap_ind"].mean().round(1).unstack(0).to_string())
    print(lf.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
