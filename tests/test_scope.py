import pytest

from jevexp.paths import MANIFEST
from jevexp.runners.common import items, run_tag

pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="data not fetched")


@pytest.mark.parametrize("dataset,full,subset,pilot", [
    ("nusax_senti", 4800, 1200, 1200), ("nt_senti", 8400, 1400, 700), ("nt_emot", 7000, 1400, 700)])
def test_scope_sizes(dataset, full, subset, pilot):
    assert len(items(dataset, "test", "full")) == full
    assert len(items(dataset, "test", "subset")) == subset
    assert len(items(dataset, "valid")) == pilot


def test_test_split_requires_explicit_scope():
    with pytest.raises(ValueError):
        items("nusax_senti", "test")
    with pytest.raises(ValueError):
        items("nusax_senti", "valid", "full")
    assert run_tag("valid", None) == "valid" and run_tag("test", "full") == "test-full"
