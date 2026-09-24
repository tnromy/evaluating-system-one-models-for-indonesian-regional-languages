"""Pinned download, uniform loading and fixed subsets for the three datasets.

Datasets (docs/metodologi.md §3):
  nusax_senti  NusaX sentiment, 12 languages, 3 labels
  nt_senti     NusaTranslation sentiment (NusaWrites), 2 labels
  nt_emot      NusaTranslation emotion (NusaWrites), 5 labels

CLI:  python -m jevexp.data {fetch,summary,subsets}
"""
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import requests

from .paths import MANIFEST, RAW, SUBSETS

NUSAX_REPO, NUSAX_COMMIT = "IndoNLP/nusax", "75b0be5d982348332509f296e10d9bf9cb66d60f"
NUSAWRITES_REPO, NUSAWRITES_COMMIT = "IndoNLP/nusa-writes", "b337f511185cd44c9c57ff906ecef1af6165adbc"

NUSAX_FOLDERS = {
    "ace": "acehnese", "ban": "balinese", "bbc": "toba_batak", "bjn": "banjarese",
    "bug": "buginese", "eng": "english", "ind": "indonesian", "jav": "javanese",
    "mad": "madurese", "min": "minangkabau", "nij": "ngaju", "sun": "sundanese",
}
NT_LANGS = ["abs", "bew", "bhp", "btk", "jav", "mad", "mak", "min", "mui", "rej", "sun"]
# Only these 7 share ids (fully parallel, 1,200 senti / 1,000 emot test items). abs, bhp, mui, rej are
# smaller independent samples that overlap only partly, so they are excluded from paired analyses.
NT_PARALLEL = ["bew", "btk", "jav", "mad", "mak", "min", "sun"]
SPLITS = ("train", "valid", "test")

DATASETS = {
    "nusax_senti": {"langs": sorted(NUSAX_FOLDERS), "parallel_langs": sorted(NUSAX_FOLDERS),
                    "labels": ("negative", "neutral", "positive")},
    "nt_senti": {"langs": NT_LANGS, "parallel_langs": NT_PARALLEL, "labels": ("negative", "positive")},
    "nt_emot": {"langs": NT_LANGS, "parallel_langs": NT_PARALLEL,
                "labels": ("anger", "fear", "happy", "love", "sadness")},
}

SUBSET_SEED = 20260924
SUBSET_SIZES = {"nusax_senti": 100, "nt_senti": 200, "nt_emot": 200}


def _sources(dataset, lang, split):
    """Return (url, local relative path) for one CSV at the pinned commit."""
    if dataset == "nusax_senti":
        rel = f"datasets/sentiment/{NUSAX_FOLDERS[lang]}/{split}.csv"
        return f"https://raw.githubusercontent.com/{NUSAX_REPO}/{NUSAX_COMMIT}/{rel}", f"nusax/{rel}"
    task = {"nt_senti": "senti", "nt_emot": "emot"}[dataset]
    rel = f"data/nusa_kalimat-{task}-{lang}-{split}.csv"
    return f"https://raw.githubusercontent.com/{NUSAWRITES_REPO}/{NUSAWRITES_COMMIT}/{rel}", f"nusawrites/{rel}"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_manifest():
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def fetch():
    """Download every CSV at the pinned commit; verify against / write MANIFEST.json.

    If a file is already listed in the manifest, its hash must match (reproducibility guard).
    """
    manifest = _read_manifest()
    for dataset, spec in DATASETS.items():
        for lang in spec["langs"]:
            for split in SPLITS:
                url, rel = _sources(dataset, lang, split)
                path = RAW / rel
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    r = requests.get(url, timeout=60)
                    r.raise_for_status()
                    path.write_bytes(r.content)
                digest = _sha256(path)
                known = manifest.get(rel)
                if known and known["sha256"] != digest:
                    raise RuntimeError(f"hash mismatch for {rel}: manifest {known['sha256']} vs file {digest}")
                n_rows = len(pd.read_csv(path, keep_default_na=False))
                manifest[rel] = {"url": url, "sha256": digest, "rows": n_rows}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(dict(sorted(manifest.items())), indent=1) + "\n")
    return manifest


def load(dataset, split=None, langs=None):
    """Uniform frame: dataset, lang, split, sentence_id, text, label (canonical lowercase)."""
    spec = DATASETS[dataset]
    frames = []
    for lang in langs or spec["langs"]:
        for sp in [split] if split else SPLITS:
            path = RAW / _sources(dataset, lang, sp)[1]
            if not path.exists():
                raise FileNotFoundError(f"{path} missing; run `python -m jevexp.data fetch`")
            df = pd.read_csv(path, keep_default_na=False, dtype={"id": str})
            frames.append(pd.DataFrame({
                "dataset": dataset, "lang": lang, "split": sp,
                "sentence_id": df["id"].str.strip(),
                "text": df["text"],
                "label": df["label"].str.strip().str.lower(),
            }))
    out = pd.concat(frames, ignore_index=True)
    unknown = set(out["label"]) - set(spec["labels"])
    if unknown:
        raise ValueError(f"{dataset}: labels outside {spec['labels']}: {sorted(unknown)}")
    return out


def parallel_report(df):
    """Per split: ids shared by all languages, and ids whose gold label differs across languages."""
    report = {}
    for sp, g in df.groupby("split"):
        id_sets = g.groupby("lang")["sentence_id"].apply(set)
        common = set.intersection(*id_sets)
        n_labels = g[g["sentence_id"].isin(common)].groupby("sentence_id")["label"].nunique()
        report[sp] = {
            "per_lang": {lang: len(ids) for lang, ids in id_sets.items()},
            "common_ids": len(common),
            "label_conflicts": int((n_labels > 1).sum()),
        }
    return report


def make_subsets(seed=SUBSET_SEED, sizes=SUBSET_SIZES):
    """Fixed test subsets (S-NX, S-NT, S-NE): proportional-stratified by label over sentence_id,
    drawn only from ids present in every language with a consistent label, so parallelism holds."""
    rng = np.random.default_rng(seed)
    subsets = {}
    for dataset, n in sizes.items():
        test = load(dataset, split="test", langs=DATASETS[dataset]["parallel_langs"])
        id_sets = test.groupby("lang")["sentence_id"].apply(set)
        common = set.intersection(*id_sets)
        per_id = test[test["sentence_id"].isin(common)].groupby("sentence_id")["label"].agg(set)
        gold = per_id[per_id.apply(len) == 1].apply(lambda s: next(iter(s))).sort_index()
        counts = gold.value_counts().sort_index()
        # largest-remainder allocation of n across labels
        exact = counts / counts.sum() * n
        alloc = np.floor(exact).astype(int)
        for lab in (exact - alloc).sort_values(ascending=False).index[: n - alloc.sum()]:
            alloc[lab] += 1
        chosen = []
        for lab, k in alloc.items():
            pool = gold.index[gold == lab].to_numpy()
            chosen += sorted(rng.choice(pool, size=k, replace=False).tolist())
        subsets[dataset] = {
            "seed": seed, "n": n, "split": "test", "langs": DATASETS[dataset]["parallel_langs"],
            "label_counts": {lab: int(k) for lab, k in alloc.items()},
            "sentence_ids": sorted(chosen),
        }
    SUBSETS.parent.mkdir(parents=True, exist_ok=True)
    SUBSETS.write_text(json.dumps(subsets, indent=1) + "\n")
    return subsets


def summary():
    for dataset in DATASETS:
        df = load(dataset)
        print(f"\n== {dataset}")
        print(df.pivot_table(index="lang", columns="split", values="sentence_id", aggfunc="count")[list(SPLITS)])
        par = df[df["lang"].isin(DATASETS[dataset]["parallel_langs"])]
        test = par[par["split"] == "test"]
        print("parallel langs:", DATASETS[dataset]["parallel_langs"])
        print("test label counts identical across parallel langs:",
              test.groupby("lang")["label"].value_counts().unstack().nunique().eq(1).all())
        print(test[test["lang"] == test["lang"].iloc[0]]["label"].value_counts().to_dict())
        print("parallel report:", json.dumps(parallel_report(par)))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m jevexp.data")
    ap.add_argument("cmd", choices=["fetch", "summary", "subsets"])
    cmd = ap.parse_args(argv).cmd
    if cmd == "fetch":
        print(f"{len(fetch())} files in {MANIFEST}")
    elif cmd == "summary":
        summary()
    else:
        for name, s in make_subsets().items():
            print(name, s["n"], s["label_counts"])


if __name__ == "__main__":
    sys.exit(main())
