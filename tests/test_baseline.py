import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from jevexp.data import load
from jevexp.paths import MANIFEST
from jevexp.report_f1 import compare
from jevexp.runners.baseline import _prediction_frame, fit_select, tokenize
from jevexp.schema import validate


def _toy(n, seed):
    rng = np.random.default_rng(seed)
    words = {"positive": "bagus enak senang", "negative": "buruk jelek kecewa", "neutral": "biasa saja ada"}
    labels = rng.choice(list(words), n)
    return pd.DataFrame({"sentence_id": [str(i) for i in range(n)], "label": labels,
                         "text": [f"{words[lab]} hotel , ." for lab in labels]})


def test_tokenize_matches_notebook():
    assert tokenize(["Iya bener, deknen lagi jaga warung."]) == ["Iya bener , deknen lagi jaga warung ."]


@pytest.mark.parametrize("clf", ["nb", "svm", "lr"])
def test_fit_select_and_schema(clf):
    train, valid, test = _toy(60, 0), _toy(20, 1), _toy(30, 2)
    out = fit_select(train, valid, test, clf, "bow")
    assert (out["pred"] == test["label"].to_numpy()).mean() == 1.0
    frame = _prediction_frame("t", "nusax_senti", "ind", test, out, clf, "bow")
    validate(frame)
    if clf == "svm":
        assert (frame["confidence_type"] == "none").all()
    else:
        probs = np.stack(frame["probs"].to_numpy())
        # column order follows DATASETS labels (negative, neutral, positive)
        assert np.allclose(probs.sum(axis=1), 1)
        assert (np.array(["negative", "neutral", "positive"])[probs.argmax(1)] == frame["pred"]).all()


def test_report_select_valid_picks_best_validation():
    scores = pd.DataFrame([
        dict(lang=lang, clf=clf, feature=feat, valid_acc=acc, test_macro_f1=f1)
        for lang in ["ace", "ban", "bbc", "bjn", "bug", "eng", "ind", "jav", "mad", "min", "nij", "sun"]
        for clf in ["nb", "svm", "lr"]
        for feat, acc, f1 in [("bow", 0.8, 0.70), ("tfidf", 0.9, 0.75)]
    ])
    v, summary = compare(scores)
    sel = v[v.variant == "select_valid"]
    assert (sel["feature"] == "tfidf").all() and np.allclose(sel["f1"], 75.0)


@pytest.mark.data
@pytest.mark.skipif(not MANIFEST.exists(), reason="data not fetched")
def test_anchor_matches_official_notebook_output():
    """Official notebook output: indonesian, NB, BoW -> test macro-F1 0.7311052022751223."""
    d = load("nusax_senti", langs=["ind"])
    train, valid, test = (d[d.split == s] for s in ("train", "valid", "test"))
    out = fit_select(train, valid, test, "nb", "bow")
    assert f1_score(test["label"], out["pred"], average="macro") == pytest.approx(0.7311052022751223, abs=5e-3)
