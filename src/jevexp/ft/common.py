"""Shared F5 plumbing: run data, prediction frames, resumable checkpoints."""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..data import DATASETS, load
from ..paths import SUBSETS
from ..runners.common import guard_split

PATIENCE = 3       # NusaX Table 10: early stopping 3
MAX_EPOCHS = 100


def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_langs(run):
    ds = run["dataset"]
    if run["kind"] == "zeroshot" or run["train_lang"] == "pooled":
        return list(DATASETS[ds]["parallel_langs"])
    return [run["train_lang"]]


def run_data(run, plan):
    """(train, valid, test) frames for a run. Grid runs have no test; zero-shot runs have no train.

    Validation is always the full validation split of the training languages (also on learning curves,
    as for B1); test is only read for final predictions, never for selection.
    """
    ds, langs = run["dataset"], run_langs(run)
    if run["kind"] == "zeroshot":
        guard_split("test")
        test = load(ds, split="test", langs=langs)
        if run["eval_scope"] == "subset":
            ids = set(json.loads(SUBSETS.read_text())[ds]["sentence_ids"])
            test = test[test["sentence_id"].isin(ids)]
        return None, None, test.reset_index(drop=True)
    train = load(ds, split="train", langs=langs)
    if run["n"]:
        ids = set(plan["curves"][ds][str(run["seed"])][str(run["n"])])
        train = train[train["sentence_id"].isin(ids)]
        assert len(train) == run["n"] * len(langs), f"{run['run_id']}: {len(train)} train rows"
    valid = load(ds, split="valid", langs=langs)
    test = None
    if run["kind"] != "grid":
        guard_split("test")
        test = load(ds, split="test", langs=langs)
    return train.reset_index(drop=True), valid.reset_index(drop=True), \
        None if test is None else test.reset_index(drop=True)


def condition(run):
    if run["kind"] == "zeroshot":
        return "zeroshot"
    regime = "pooled" if run["train_lang"] == "pooled" else "per_lang"
    return f"regime={regime};n={run['n'] or 'full'}"


def pred_frame(run, split, df, probs, model_id, model_version, train_n, train_time_s):
    labels = list(DATASETS[run["dataset"]]["labels"])
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return pd.DataFrame({
        "run_id": run["run_id"], "arm": run["arm"], "condition": condition(run),
        "dataset": run["dataset"], "split": split, "lang": df["lang"].to_numpy(),
        "sentence_id": df["sentence_id"].to_numpy(), "gold": df["label"].to_numpy(),
        "pred": [labels[i] for i in probs.argmax(axis=1)],
        "confidence_type": "model_prob", "model_id": model_id, "timestamp_utc": now_utc(),
        "probs": list(probs), "confidence": probs.max(axis=1), "model_version": model_version,
        "train_n": int(train_n), "seed": int(run["seed"]), "train_time_s": float(train_time_s),
        "gpu": gpu_name(),
    })


def rng_state():
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(s):
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


class Checkpoint:
    """Rolling `runs/<id>/ckpt/latest.pt` in the store, written at most every `every_s` seconds (at epoch ends).

    Uploading a full optimizer state costs time, so short runs never checkpoint; a run killed before its
    first checkpoint simply restarts, losing at most `every_s` of work.
    """

    def __init__(self, store, run_id, work, every_s=900, on_save=None):
        self.store, self.key = store, f"runs/{run_id}/ckpt/latest.pt"
        self.local = Path(work) / "ckpt" / f"{run_id}.pt"
        self.every_s, self.on_save = every_s, on_save
        self.last = time.time()

    def load(self):
        if not self.store.exists(self.key):
            return None
        self.store.get_file(self.key, self.local)
        state = torch.load(self.local, map_location="cpu", weights_only=False)
        self.local.unlink()
        return state

    def maybe_save(self, make_state, force=False):
        if not force and time.time() - self.last < self.every_s:
            return False
        self.local.parent.mkdir(parents=True, exist_ok=True)
        torch.save(make_state(), self.local)
        self.store.put_file(self.key, self.local)
        self.local.unlink()
        self.last = time.time()
        if self.on_save:
            self.on_save()
        return True

    def clear(self):
        if self.store.exists(self.key):
            self.store.delete(self.key)


def hf_kwargs(ctx):
    kw = {}
    if ctx.get("cache_dir"):
        kw["cache_dir"] = ctx["cache_dir"]
    if ctx.get("hf_token"):
        kw["token"] = ctx["hf_token"]
    return kw


def env_versions():
    import transformers
    out = {"torch": torch.__version__, "transformers": transformers.__version__,
           "cuda": torch.version.cuda, "python": os.sys.version.split()[0]}
    try:
        import laya
        out["laya"] = laya.__version__
    except ImportError:
        pass
    return out
