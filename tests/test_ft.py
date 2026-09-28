"""F5 plumbing: store, learning-curve subsets, resumable training, idempotent worker (no downloads)."""
import numpy as np
import pandas as pd
import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

from jevexp.ft import hf_clf, worker as W
from jevexp.ft.common import Checkpoint
from jevexp.ft.plan import CURVE_N, _alloc, build, curve_subsets
from jevexp.ft.store import LocalStore
from jevexp.schema import validate


def test_local_store_roundtrip(tmp_path):
    s = LocalStore(tmp_path)
    assert not s.exists("a/b.json") and s.get_json("a/b.json", {}) == {}
    s.put_json("a/b.json", {"x": 1})
    s.put_bytes("a/c/d.bin", b"xyz")
    assert s.get_json("a/b.json") == {"x": 1}
    assert s.list("a/") == ["a/b.json", "a/c/d.bin"]
    s.delete("a/c")
    assert s.list("a/") == ["a/b.json"]


def test_alloc_min_one_and_total():
    counts = {"anger": 733, "fear": 427, "happy": 697, "love": 441, "sadness": 702}
    for n in (10, 25, 1000):
        a = _alloc(counts, n)
        assert sum(a.values()) == n and min(a.values()) >= 1


def test_curve_subsets_nested_stratified_deterministic():
    rng = np.random.default_rng(0)
    gold = pd.Series(rng.choice(["negative", "neutral", "positive"], 500, p=[.4, .25, .35]),
                     index=[str(i) for i in range(500)])
    a, b = curve_subsets("nusax_senti", gold, 1), curve_subsets("nusax_senti", gold, 1)
    assert a == b
    assert curve_subsets("nusax_senti", gold, 2) != a
    prev = set()
    for n in CURVE_N["nusax_senti"]:
        ids = set(a[n])
        assert len(ids) == n and prev <= ids and set(gold[list(ids)]) == set(gold)
        prev = ids


@pytest.mark.data
def test_plan_shape():
    plan = build()
    runs = plan["runs"]
    assert len({r["run_id"] for r in runs}) == len(runs)
    assert sum(r["kind"] == "grid" for r in runs) == 40
    assert sum(r["export"] for r in runs) == 3
    stage1 = [r for r in runs if r["stage"] == 1 and r["kind"] == "train"]
    assert {r["arm"] for r in stage1} == {"B2", "B3", "C1"}
    # curves: B2 and C1 on the same subset ids
    assert set(plan["curves"]) == {"nusax_senti", "nt_senti", "nt_emot"}


# ------------------------------------------------------------------ resumable training with a tiny model
@pytest.fixture
def tiny(monkeypatch, tmp_path):
    words = "bagus jelek biasa sangat tidak film makanan enak buruk".split()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words
    (tmp_path / "vocab.txt").write_text("\n".join(vocab))
    tok = BertTokenizerFast(vocab_file=str(tmp_path / "vocab.txt"))

    def load_model(arm, n_labels, ctx):
        cfg = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                         intermediate_size=32, num_labels=n_labels)
        return tok, BertForSequenceClassification(cfg)
    monkeypatch.setattr(hf_clf, "load_model", load_model)
    rng = np.random.default_rng(0)
    labs = ["negative", "neutral", "positive"]

    def frame(n, split):
        y = rng.choice(labs, n)
        text = [{"negative": "jelek buruk", "neutral": "biasa film", "positive": "bagus enak"}[v] + " sangat"
                for v in y]
        return pd.DataFrame({"dataset": "nusax_senti", "lang": "ind", "split": split,
                             "sentence_id": [str(i) for i in range(n)], "text": text, "label": y})
    return frame(40, "train"), frame(12, "valid"), frame(12, "test")


class Stop(Exception):
    pass


def _ctx(store, run_id, tmp_path, stop_after=None, max_epochs=4):
    seen = []

    def heartbeat(msg):
        seen.append(msg)
        if stop_after and len(seen) == stop_after:
            raise Stop()
    return {"ckpt": Checkpoint(store, run_id, tmp_path / "work", every_s=0), "heartbeat": heartbeat,
            "log": lambda m: seen.append("LOG " + m), "max_epochs": max_epochs}, seen


def test_hf_train_resumes_after_kill(tiny, tmp_path):
    store = LocalStore(tmp_path / "store")
    run = {"run_id": "b2-x", "arm": "B2", "dataset": "nusax_senti", "seed": 0}
    # heartbeat fires before the epoch checkpoint: kill during the heartbeat of epoch 3 -> checkpoint of epoch 2
    ctx, seen = _ctx(store, "b2-x", tmp_path, stop_after=3)
    with pytest.raises(Stop):
        hf_clf.train(run, tiny, 1e-3, 8, ctx)
    assert store.exists("runs/b2-x/ckpt/latest.pt")
    ctx2, seen2 = _ctx(store, "b2-x", tmp_path)
    res = hf_clf.train(run, tiny, 1e-3, 8, ctx2)
    assert any("resumed at epoch 2" in m for m in seen2)
    assert [h["epoch"] for h in res["history"]] == list(range(1, res["epochs"] + 1))
    assert res["test_probs"].shape == (12, 3)


def test_hf_train_resume_equals_uninterrupted(tiny, tmp_path):
    run = {"run_id": "b2-y", "arm": "B2", "dataset": "nusax_senti", "seed": 1}
    a = hf_clf.train(run, tiny, 1e-3, 8, _ctx(LocalStore(tmp_path / "s1"), "b2-y", tmp_path)[0])
    store = LocalStore(tmp_path / "s2")
    with pytest.raises(Stop):
        hf_clf.train(run, tiny, 1e-3, 8, _ctx(store, "b2-y", tmp_path, stop_after=3)[0])
    b = hf_clf.train(run, tiny, 1e-3, 8, _ctx(store, "b2-y", tmp_path)[0])
    assert a["history"] == b["history"]
    np.testing.assert_allclose(a["test_probs"], b["test_probs"], atol=1e-6)


# ------------------------------------------------------------------ worker idempotence
@pytest.mark.data
def test_worker_skips_done_and_records_errors(tmp_path, monkeypatch):
    calls = []

    def fake_train(run, data, lr, bs, ctx):
        calls.append(run["run_id"])
        if run["run_id"].endswith("bs32") and "lr1e-06" in run["run_id"]:
            raise RuntimeError("boom")
        n_valid = len(data[1])
        return {"valid_probs": np.full((n_valid, 3), 1 / 3), "test_probs": None, "valid_f1": 0.5,
                "best_epoch": 1, "epochs": 1, "history": [], "train_time_s": 0.1, "train_n": len(data[0])}
    monkeypatch.setattr(W.hf_clf, "train", fake_train)
    store = LocalStore(tmp_path / "store")
    w = W.Worker(store, work=tmp_path / "work", stage=1, log=lambda m: None)
    w.plan["runs"] = [r for r in w.plan["runs"] if r["kind"] == "grid" and r["arm"] == "B2"]
    w.check_plan()
    w.run()
    assert len(calls) == 20
    st = w.status()
    assert st["done"].sum() == 19 and st["failed"].sum() == 1
    w2 = W.Worker(store, work=tmp_path / "work", stage=1, log=lambda m: None)
    w2.plan["runs"] = w.plan["runs"]
    w2.run()
    assert len(calls) == 21  # only the failed run is retried (attempt 2)
    w2.run()
    assert len(calls) == 21  # MAX_ATTEMPTS reached: skipped
    cfg = w2.best_config("B2")
    assert cfg is None  # one grid cell never finished -> no winner
    out = w2.collect(tmp_path / "output")
    assert out.exists()


def test_worker_refuses_other_code(tmp_path):
    store = LocalStore(tmp_path / "store")
    store.put_json("plan/active.json", {"plan_sha": "x", "code_sha": "y", "since": "t", "history": []})
    w = W.Worker(store, work=tmp_path / "work", log=lambda m: None)
    with pytest.raises(RuntimeError):
        w.check_plan()
    w.check_plan(allow_new_code=True)
    assert store.get_json("plan/active.json")["history"][0]["code_sha"] == "y"


def test_pred_frame_valid(tiny):
    from jevexp.ft.common import pred_frame
    _, valid, _ = tiny
    run = {"run_id": "b2-z", "arm": "B2", "dataset": "nusax_senti", "seed": 0, "kind": "train",
           "train_lang": "ind", "n": None}
    p = torch.softmax(torch.randn(len(valid), 3), -1).numpy()
    validate(pred_frame(run, "valid", valid, p, "m", "v", 40, 1.0))


def test_hf_train_with_worker_default_ctx(tiny, tmp_path):
    """The worker passes max_epochs=None (Colab default): training must fall back to MAX_EPOCHS, not crash."""
    store = LocalStore(tmp_path / "store")
    w = W.Worker(store, work=tmp_path / "work", log=lambda m: None)
    ctx = w._ctx("b2-w")
    assert ctx["max_epochs"] is None
    run = {"run_id": "b2-w", "arm": "B2", "dataset": "nusax_senti", "seed": 0}
    res = hf_clf.train(run, tiny, 1e-3, 8, ctx)
    assert 1 <= res["epochs"] <= hf_clf.MAX_EPOCHS and res["test_probs"].shape == (12, 3)


# ------------------------------------------------------------------ progress + auto stages (K63)
def _mini_plan():
    mk = lambda stage, rid, kind="train", arm="C1", ds="nusax_senti", lang="pooled", n=None: {
        "stage": stage, "kind": kind, "run_id": rid, "arm": arm, "dataset": ds, "train_lang": lang, "seed": 0,
        "n": n, "export": False}
    return {"runs": [mk(1, "a1"), mk(1, "a2"), mk(1, "a3"), mk(2, "b1", arm="B2", lang="jav", n=10),
                     mk(3, "c1", arm="C1", ds="nt_emot", lang="jav", n=10)]}


def test_progress_counts_and_eta():
    from jevexp.ft.progress import PRIOR_S, Progress
    import tempfile
    store = LocalStore(tempfile.mkdtemp())
    store.put_json("runs/a1/done.json", {})
    store.put_json("runs/a1/meta.json", {"wall_s": 100.0})
    store.put_json("runs/a2/error.json", {"attempts": 2})
    store.put_json("runs/b1/error.json", {"attempts": 1})
    text, d = Progress(store, _mini_plan()).report(stage=1)
    s1, s2, s3 = d["stages"][1], d["stages"][2], d["stages"][3]
    assert (s1["done"], s1["total"], s1["failed"], s1["retry"]) == (1, 3, 1, 0)
    assert s1["eta_h"] == round(100 / 3600, 2) and not s1["prior"]  # a3 estimated from a1's group mean
    assert s2["retry"] == 1 and s2["prior"] and s2["eta_h"] == round(PRIOR_S[("curve", "B2")] / 3600, 2)
    assert "Stage 1" in text and "*" in text and "gagal permanen" in text


def test_run_all_advances_stages_and_collects(tmp_path, monkeypatch):
    store = LocalStore(tmp_path / "store")
    w = W.Worker(store, work=tmp_path / "work", out_dir=tmp_path / "output", log=lambda m: None)
    w.plan = _mini_plan()
    w.progress.plan = w.plan
    seen, collected = [], []

    def fake_one(r, cfg):
        seen.append((w.stage, r["run_id"]))
        store.put_json(f"runs/{r['run_id']}/meta.json", {"wall_s": 1.0})
        store.put_json(f"runs/{r['run_id']}/done.json", {})
    monkeypatch.setattr(w, "_one", fake_one)
    monkeypatch.setattr(w, "best_config", lambda arm: {"lr": 1e-5, "bs": 8})
    monkeypatch.setattr(w, "collect", lambda out: collected.append(w.stage))
    w.run_all(1, auto_next=True)
    assert seen == [(1, "a1"), (1, "a2"), (1, "a3"), (2, "b1"), (3, "c1")]
    assert collected == [1, 2, 3]
    assert "Total sisa semua stage" in store.get_bytes("status/progress.txt").decode()
    assert (tmp_path / "output" / "progress.txt").exists()
    assert store.get_json("status/progress.json")["stages"]["3"]["done"] == 1


def test_heartbeat_prints_status_periodically(tmp_path):
    logs = []
    w = W.Worker(LocalStore(tmp_path / "s"), work=tmp_path / "w", status_every_s=0, log=logs.append)
    w.plan = _mini_plan()
    w.progress.plan = w.plan
    w.current = {"run_id": "a1", "started": 0.0, "msg": None}
    ctx = w._ctx("a1")
    ctx["heartbeat"]("epoch 1/4 loss 0.3")
    assert any(m.startswith("==== STATUS F5") for m in logs)
    assert any("Sedang: a1 — epoch 1/4 loss 0.3" in m for m in logs)


def test_eta_scales_other_dataset_by_train_size():
    """NT runs without history borrow the same arm's NusaX per-language duration x train-size ratio, marked prior."""
    from jevexp.ft.progress import Progress
    import tempfile
    mk = lambda rid, ds: {"stage": 1, "kind": "train", "run_id": rid, "arm": "B2", "dataset": ds,
                          "train_lang": "jav", "seed": 0, "n": None, "export": False}
    store = LocalStore(tempfile.mkdtemp())
    store.put_json("runs/x/done.json", {})
    store.put_json("runs/x/meta.json", {"wall_s": 50.0})
    _, d = Progress(store, {"runs": [mk("x", "nusax_senti"), mk("y", "nt_senti")]}).report(stage=1)
    assert d["stages"][1]["eta_h"] == round(50 * 3400 / 500 / 3600, 2) and d["stages"][1]["prior"]
