import pandas as pd
import pytest

from jevexp.paths import ROOT

FEAT = ROOT / "features"
pytestmark = pytest.mark.skipif(not (FEAT / "sentence_overlap.parquet").exists(), reason="features not built")


def test_overlap_ind_is_identity_for_indonesian_and_complete():
    s = pd.read_parquet(FEAT / "sentence_overlap.parquet")
    ind = s[(s["dataset"] == "nusax_senti") & (s["lang"] == "ind")]
    assert (ind["overlap_ind"].round(6) == 100).all()
    assert s["overlap_ind"].notna().all()


def test_language_features_cover_all_parallel_languages():
    from jevexp.data import DATASETS
    lf = pd.read_csv(FEAT / "lang_features.csv")
    langs = {lang for d in DATASETS.values() for lang in d["parallel_langs"]}
    assert set(lf["lang"]) == langs
    assert lf.loc[lf["lang"] == "ind", "dist_genetic"].iloc[0] == pytest.approx(0, abs=1e-9)
