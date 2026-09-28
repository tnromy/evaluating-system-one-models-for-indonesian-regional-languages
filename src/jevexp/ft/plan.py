"""F5 run plan (docs/metodologi.md §4, K60/K61): every fine-tuning run, frozen in `splits/f5_plan.json`.

Stages (user order "core first"):
  1  grid (B2/B3 on NusaX ind valid) -> B2/B3 full data -> C1 full data + C2 zero-shot
  2  NusaX learning curves (B2, C1; B1 runs on the laptop from the same train subsets)
  3  NusaTranslation learning curves (B2, C1)

Learning-curve train subsets are drawn once per (dataset, seed) over sentence ids that are parallel with a
consistent gold label, label-stratified with at least one item per label, and nested (n=10 is inside n=25 ...).
The same ids are used for every language and for B1/B2/C1, so curves are paired (K61).

CLI:  python -m jevexp.ft.plan            # (re)write splits/f5_plan.json and print the run counts
"""
import hashlib
import json
from collections import Counter

import numpy as np

from ..data import DATASETS, _consistent_gold, load
from ..paths import SPLITS

PLAN = SPLITS / "f5_plan.json"
PLAN_SEED = 20260928
SEEDS = [0, 1, 2]
GRID_LR = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]  # NusaX paper Table 10
GRID_BS = [4, 8, 16, 32]
GRID_LANG = "ind"
CURVE_N = {"nusax_senti": [10, 25, 50, 100, 250, 500], "nt_senti": [10, 25, 50, 100, 250, 500, 1000],
           "nt_emot": [10, 25, 50, 100, 250, 500, 1000]}
C2_SCOPE = {"nusax_senti": "full", "nt_senti": "subset", "nt_emot": "subset"}  # metodologi §4 table

MODELS = {  # arm -> (HF repo, pinned revision)
    "B2": ("indobenchmark/indobert-base-p1", "c2cd0b51ddce6580eb35263b39b0a1e5fb0a39e2"),
    "B3": ("FacebookAI/xlm-roberta-base", "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"),
    "C1": ("convaiinnovations/laya-multilingual", "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67"),
}
MODELS["C2"] = MODELS["C1"]
EXPORT = {("B2", "nusax_senti", "pooled", 0), ("B3", "nusax_senti", "pooled", 0), ("C1", "nusax_senti", "pooled", 0)}


def _alloc(counts, n):
    """At least one per label, the rest proportional by largest remainder."""
    labels = sorted(counts)
    base = {lab: 1 for lab in labels}
    rest = n - len(labels)
    total = sum(counts.values())
    exact = {lab: counts[lab] / total * rest for lab in labels}
    for lab in labels:
        base[lab] += int(np.floor(exact[lab]))
    left = n - sum(base.values())
    for lab in sorted(labels, key=lambda lab: exact[lab] - np.floor(exact[lab]), reverse=True)[:left]:
        base[lab] += 1
    return base


def curve_subsets(dataset, gold, seed):
    """{n: sorted ids} nested over n for one (dataset, seed). `gold`: Series sentence_id -> label."""
    rng = np.random.default_rng([PLAN_SEED, seed, sorted(DATASETS).index(dataset)])
    order = {lab: rng.permutation(sorted(gold.index[gold == lab])).tolist() for lab in sorted(set(gold))}
    counts = Counter(gold)
    out, prev = {}, set()
    for n in CURVE_N[dataset]:
        alloc = _alloc(counts, n)
        ids = {i for lab, k in alloc.items() for i in order[lab][:k]}
        if not prev <= ids:
            raise AssertionError(f"{dataset} seed {seed}: subset n={n} is not nested")
        out[n] = sorted(ids)
        prev = ids
    return out


def run_id(arm, dataset, train_lang, seed, n):
    return f"{arm.lower()}-{dataset}-{train_lang}-s{seed}-n{n or 'full'}"


def build():
    runs, curves = [], {}

    def add(stage, **r):
        r.setdefault("n", None)
        r["run_id"] = run_id(r["arm"], r["dataset"], r["train_lang"], r["seed"], r["n"])
        r["export"] = (r["arm"], r["dataset"], r["train_lang"], r["seed"]) in EXPORT and r["n"] is None
        runs.append({"stage": stage, "kind": "train", **r})

    # 1a grid
    for arm in ("B2", "B3"):
        for lr in GRID_LR:
            for bs in GRID_BS:
                runs.append({"stage": 1, "kind": "grid", "run_id": f"grid-{arm.lower()}-lr{lr:g}-bs{bs}",
                             "arm": arm, "dataset": "nusax_senti", "train_lang": GRID_LANG, "seed": 0,
                             "n": None, "lr": lr, "bs": bs, "export": False})
    # 1b/1c B2, B3 full data: NusaX per language + pooled, NT per language
    for dataset in DATASETS:
        langs = DATASETS[dataset]["parallel_langs"] + (["pooled"] if dataset == "nusax_senti" else [])
        for arm in ("B2", "B3"):
            for lang in langs:
                for seed in SEEDS:
                    add(1, arm=arm, dataset=dataset, train_lang=lang, seed=seed)
    # 1d C1 (pooled first: it is the primary regime), then C1 per language on NusaX, then C2 zero-shot
    for dataset in DATASETS:
        for seed in SEEDS:
            add(1, arm="C1", dataset=dataset, train_lang="pooled", seed=seed)
    for lang in DATASETS["nusax_senti"]["parallel_langs"]:
        for seed in SEEDS:
            add(1, arm="C1", dataset="nusax_senti", train_lang=lang, seed=seed)
    for dataset in DATASETS:
        runs.append({"stage": 1, "kind": "zeroshot", "run_id": f"c2-{dataset}-zeroshot", "arm": "C2",
                     "dataset": dataset, "train_lang": "none", "seed": 0, "n": 0,
                     "eval_scope": C2_SCOPE[dataset], "export": False})
    # 2/3 learning curves (B2 and C1 per language; subsets shared with B1)
    for dataset in DATASETS:
        stage = 2 if dataset == "nusax_senti" else 3
        train = load(dataset, split="train", langs=DATASETS[dataset]["parallel_langs"])
        gold = _consistent_gold(train)
        curves[dataset] = {str(seed): {str(n): ids for n, ids in curve_subsets(dataset, gold, seed).items()}
                           for seed in SEEDS}
        for n in CURVE_N[dataset]:
            for seed in SEEDS:
                for lang in DATASETS[dataset]["parallel_langs"]:
                    for arm in ("B2", "C1"):
                        add(stage, arm=arm, dataset=dataset, train_lang=lang, seed=seed, n=n)
    ids = [r["run_id"] for r in runs]
    assert len(ids) == len(set(ids)), "duplicate run ids"
    return {"version": 1, "seed": PLAN_SEED, "models": MODELS, "grid": {"lr": GRID_LR, "bs": GRID_BS,
            "lang": GRID_LANG}, "curve_n": CURVE_N, "curves": curves, "runs": runs}


def dumps(plan):
    return json.dumps(plan, indent=1, sort_keys=True) + "\n"


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_plan(path=PLAN):
    text = path.read_text()
    return json.loads(text), sha(text)


def main():
    plan = build()
    PLAN.write_text(dumps(plan))
    c = Counter((r["stage"], r["kind"], r["arm"]) for r in plan["runs"])
    for k in sorted(c):
        print(*k, c[k])
    print("total", len(plan["runs"]), "sha", sha(dumps(plan))[:12])


if __name__ == "__main__":
    main()
