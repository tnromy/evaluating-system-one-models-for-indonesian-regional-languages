"""B1: classical baselines, replicating the official NusaX notebook
(IndoNLP/nusax@75b0be5 code/notebook/sentiment_analysis_nusax_classical.ipynb):

  nltk.word_tokenize -> Count/Tfidf vectorizer fit on train only -> GridSearchCV with a predefined
  train/valid split (default scoring = accuracy, refit on train+valid) -> macro-F1 on test.

Test data is read here by design (K31): the protocol is fully pre-specified, nothing is tuned on test.

CLI:  python -m jevexp.runners.baseline --dataset {nusax_senti,nt_senti,nt_emot,all}
      python -m jevexp.runners.baseline --dataset nusax_senti --curves   # H3a learning curves, B1-LR (LR-BoW)
Needs NLTK data `punkt_tab` (nltk.download('punkt_tab')).
"""
import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
from nltk import word_tokenize
from scipy.sparse import vstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from ..data import DATASETS, load
from ..paths import RESULTS, RUNS
from ..schema import write

CLASSIFIERS = {
    "nb": (MultinomialNB, {"alpha": np.linspace(0.001, 1, 50)}),
    "svm": (SVC, {"C": [0.01, 0.1, 1, 10, 100], "kernel": ["rbf", "linear"]}),
    "lr": (LogisticRegression, {"C": np.linspace(0.001, 10, 100)}),
}
FEATURES = {"bow": CountVectorizer, "tfidf": TfidfVectorizer}


def tokenize(texts):
    return [" ".join(word_tokenize(t)) for t in texts]


def fit_select(train, valid, test, clf, feature, n_jobs=None):
    """One notebook run. Frames have columns text, label. Returns dict with fitted search + test outputs."""
    xtr, xva, xte = tokenize(train["text"]), tokenize(valid["text"]), tokenize(test["text"])
    vec = FEATURES[feature]().fit(xtr)
    xtr, xva, xte = vec.transform(xtr), vec.transform(xva), vec.transform(xte)
    ytr, yva = list(train["label"]), list(valid["label"])
    model, grid = CLASSIFIERS[clf]
    ps = PredefinedSplit([-1] * len(ytr) + [0] * len(yva))
    t0 = time.perf_counter()
    search = GridSearchCV(model(), grid, cv=ps, n_jobs=n_jobs).fit(vstack([xtr, xva]), ytr + yva)
    train_time = time.perf_counter() - t0
    est = search.best_estimator_
    pred = est.predict(xte)
    probs = est.predict_proba(xte) if hasattr(est, "predict_proba") else None
    return {"search": search, "classes": list(est.classes_), "pred": pred, "probs": probs,
            "train_time_s": train_time, "train_n": len(ytr) + len(yva)}


def _prediction_frame(run_id, dataset, lang, test, out, clf, feature):
    labels = list(DATASETS[dataset]["labels"])
    probs = None
    if out["probs"] is not None:
        order = [out["classes"].index(lab) for lab in labels]
        probs = out["probs"][:, order]
    return pd.DataFrame({
        "run_id": run_id, "arm": "B1", "condition": f"clf={clf};feat={feature}",
        "dataset": dataset, "split": "test", "lang": lang,
        "sentence_id": test["sentence_id"].to_numpy(), "gold": test["label"].to_numpy(), "pred": out["pred"],
        "confidence_type": "model_prob" if probs is not None else "none",
        "model_id": f"sklearn-{sklearn.__version__}:{clf}",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "probs": list(probs) if probs is not None else None,
        "confidence": probs.max(axis=1) if probs is not None else np.nan,
        "train_n": out["train_n"], "train_time_s": out["train_time_s"],
    })


def run(dataset):
    """All languages x classifiers x features for one dataset; writes runs/B1 and returns score rows."""
    run_id = f"B1-{dataset}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
    df = load(dataset)
    labels = list(DATASETS[dataset]["labels"])
    rows, frames, params = [], [], {}
    for lang in DATASETS[dataset]["langs"]:
        sub = df[df["lang"] == lang]
        train, valid, test = (sub[sub["split"] == s] for s in ("train", "valid", "test"))
        for clf in CLASSIFIERS:
            for feature in FEATURES:
                out = fit_select(train, valid, test, clf, feature)
                frames.append(_prediction_frame(run_id, dataset, lang, test, out, clf, feature))
                s = out["search"]
                params[f"{lang}|{clf}|{feature}"] = {k: (float(v) if isinstance(v, np.floating) else v)
                                                      for k, v in s.best_params_.items()}
                rows.append({"dataset": dataset, "lang": lang, "clf": clf, "feature": feature,
                             "valid_acc": s.best_score_,
                             "test_macro_f1": f1_score(test["label"], out["pred"], labels=labels, average="macro"),
                             "train_time_s": out["train_time_s"]})
        print(f"{dataset} {lang} done")
    write(pd.concat(frames, ignore_index=True), RUNS / "B1" / f"{run_id}.parquet")
    return rows, params


def run_curves(dataset, clf="lr", feature="bow"):
    """H3a (analysis_plan §6): per language, the frozen F5 train subsets (splits/f5_plan.json, shared with B2/C1),
    hyperparameters re-selected on the full validation split with the official protocol for every subset."""
    from ..ft.plan import load_plan
    plan, plan_sha = load_plan()
    run_id = f"B1curve-{dataset}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
    df = load(dataset, langs=DATASETS[dataset]["parallel_langs"])
    frames, rows = [], []
    for lang in DATASETS[dataset]["parallel_langs"]:
        sub = df[df["lang"] == lang]
        train, valid, test = (sub[sub["split"] == s] for s in ("train", "valid", "test"))
        for seed, by_n in plan["curves"][dataset].items():
            for n, ids in by_n.items():
                tr = train[train["sentence_id"].isin(set(ids))]
                assert len(tr) == int(n)
                out = fit_select(tr, valid, test, clf, feature, n_jobs=-1)
                f = _prediction_frame(run_id, dataset, lang, test, out, clf, feature)
                f["condition"] = f"clf={clf};feat={feature};n={n};seed={seed}"
                f["seed"] = int(seed)
                frames.append(f)
                rows.append({"dataset": dataset, "lang": lang, "seed": int(seed), "n": int(n),
                             "valid_acc": out["search"].best_score_,
                             "best_params": json.dumps({k: float(v) if isinstance(v, np.floating) else v
                                                        for k, v in out["search"].best_params_.items()}),
                             "train_time_s": out["train_time_s"], "plan_sha": plan_sha})
        print(f"{dataset} {lang} curves done", flush=True)
    write(pd.concat(frames, ignore_index=True), RUNS / "B1" / f"{run_id}.parquet")
    return rows  # no test metric here: learning curves are analysed in F6 (analysis_plan §7)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m jevexp.runners.baseline")
    ap.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    ap.add_argument("--curves", action="store_true", help="H3a learning curves (LR-BoW) on the F5 subsets")
    args = ap.parse_args(argv)
    if args.curves:
        out_dir = RESULTS / "f5_b1_curves"
        out_dir.mkdir(parents=True, exist_ok=True)
        for dataset in DATASETS if args.dataset == "all" else [args.dataset]:
            pd.DataFrame(run_curves(dataset)).to_csv(out_dir / f"selection_{dataset}.csv", index=False)
        return
    out_dir = RESULTS / "f1_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS if args.dataset == "all" else [args.dataset]:
        rows, params = run(dataset)
        pd.DataFrame(rows).to_csv(out_dir / f"scores_{dataset}.csv", index=False)
        (out_dir / f"selected_params_{dataset}.json").write_text(json.dumps(params, indent=1) + "\n")


if __name__ == "__main__":
    main()
