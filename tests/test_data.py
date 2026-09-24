"""Gate F0: the pinned data matches docs/metodologi.md §3. Needs `python -m jevexp.data fetch`."""
import json

import pytest

from jevexp.data import DATASETS, NT_PARALLEL, SUBSET_SIZES, load, parallel_report
from jevexp.paths import MANIFEST, SUBSETS

pytestmark = [pytest.mark.data,
              pytest.mark.skipif(not MANIFEST.exists(), reason="data not fetched")]


def test_manifest_covers_all_files():
    manifest = json.loads(MANIFEST.read_text())
    n_files = sum(len(s["langs"]) for s in DATASETS.values()) * 3
    assert len(manifest) == n_files
    assert all(len(v["sha256"]) == 64 for v in manifest.values())


def test_nusax_shape_and_parallelism():
    df = load("nusax_senti")
    sizes = df.groupby(["lang", "split"]).size().unstack()
    assert len(sizes) == 12
    assert (sizes["train"] == 500).all() and (sizes["valid"] == 100).all() and (sizes["test"] == 400).all()
    rep = parallel_report(df)
    assert rep["test"]["common_ids"] == 400 and rep["test"]["label_conflicts"] == 0
    test_ind = df[(df.lang == "ind") & (df.split == "test")]
    assert test_ind["label"].value_counts().to_dict() == {"negative": 153, "positive": 151, "neutral": 96}


@pytest.mark.parametrize("dataset,n_test,dist", [
    ("nt_senti", 1200, {"negative": 847, "positive": 353}),
    ("nt_emot", 1000, {"anger": 257, "happy": 230, "sadness": 210, "fear": 161, "love": 142}),
])
def test_nusatranslation_parallel_core(dataset, n_test, dist):
    df = load(dataset, split="test", langs=NT_PARALLEL)
    rep = parallel_report(df)["test"]
    assert rep["common_ids"] == n_test and rep["label_conflicts"] == 0
    assert set(rep["per_lang"].values()) == {n_test}
    assert df[df.lang == "jav"]["label"].value_counts().to_dict() == dist


def test_subsets_fixed_balanced_and_parallel():
    subsets = json.loads(SUBSETS.read_text())
    for dataset, n in SUBSET_SIZES.items():
        s = subsets[dataset]
        assert s["n"] == n == len(s["sentence_ids"]) == len(set(s["sentence_ids"]))
        assert sum(s["label_counts"].values()) == n
        test = load(dataset, split="test", langs=s["langs"])
        picked = test[test["sentence_id"].isin(s["sentence_ids"])]
        # every chosen id exists in every language of the subset
        assert (picked.groupby("lang").size() == n).all()
        assert picked[picked.lang == s["langs"][0]]["label"].value_counts().to_dict() == \
            {k: v for k, v in s["label_counts"].items() if v}
