import pandas as pd
import pytest

from jevexp.schema import read, validate, write


def _row(**over):
    row = dict(run_id="r1", arm="A1", condition="main", dataset="nusax_senti", split="test", lang="jav",
               sentence_id="411", gold="positive", pred="positive", confidence_type="model_prob",
               model_id="jev-1.13.0", timestamp_utc="2026-09-24T00:00:00Z",
               probs=[0.1, 0.2, 0.7], confidence=0.7)
    row.update(over)
    return row


def test_valid_frame_roundtrip(tmp_path):
    df = pd.DataFrame([_row(), _row(sentence_id="412", pred="INVALID", probs=None, confidence=None,
                                    confidence_type="none")])
    write(df, tmp_path / "x.parquet")
    assert len(read(tmp_path / "x.parquet")) == 2


@pytest.mark.parametrize("bad", [
    dict(pred="angry"),                    # label outside set
    dict(gold="neutral ", pred="neutral"),  # non-canonical gold
    dict(probs=[0.5, 0.2, 0.2]),           # does not sum to 1
    dict(probs=[0.5, 0.5]),                # wrong number of classes
    dict(confidence_type="guess"),
    dict(extra_column=1),
])
def test_rejects_violations(bad):
    with pytest.raises(ValueError):
        validate(pd.DataFrame([_row(**bad)]))


def test_rejects_missing_required_and_duplicates():
    with pytest.raises(ValueError):
        validate(pd.DataFrame([_row()]).drop(columns=["model_id"]))
    with pytest.raises(ValueError):
        validate(pd.DataFrame([_row(), _row()]))
