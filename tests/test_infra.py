import pytest

from jevexp.runners.common import RawLog, is_infra_text, run_parallel


def test_infra_text():
    assert is_infra_text("Claude AI usage limit reached|1790000000")
    assert is_infra_text("API Error: 529 Overloaded")
    assert not is_infra_text('{"choice": "positive"}')


def _items(n):
    return [{"dataset": "d", "split": "test", "lang": "x", "sentence_id": str(i)} for i in range(n)]


def test_infra_records_are_not_logged_and_are_retried(tmp_path, monkeypatch):
    monkeypatch.setattr("jevexp.runners.common.RUNS", tmp_path)
    log = RawLog("arm", "r")
    log.path = tmp_path / "r.jsonl"
    flaky = {"3"}
    fn = lambda it: {"_infra": True} if it["sentence_id"] in flaky else {"ok": 1}
    run_parallel(_items(6), fn, log, workers=1)
    assert len(log.records()) == 5 and (tmp_path / "r.infra.jsonl").exists()
    flaky.clear()
    run_parallel(_items(6), fn, log, workers=1)  # resume retries the infra item
    assert len(log.records()) == 6 and len(log.done_keys()) == 6


def test_stops_after_consecutive_infra_failures(tmp_path, monkeypatch):
    monkeypatch.setattr("jevexp.runners.common.RUNS", tmp_path)
    log = RawLog("arm", "r2")
    log.path = tmp_path / "r2.jsonl"
    with pytest.raises(SystemExit):
        run_parallel(_items(20), lambda it: {"_infra": True}, log, workers=1, max_infra=5)
    assert log.records() == []
