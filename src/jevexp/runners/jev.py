"""Jev arms via OpenRouter System One API (K34): A1 (source text), A3 (NLLB -> ind), A4 (human Indonesian).

CLI:  python -m jevexp.runners.jev --arm A1 --formulation choice_en --dataset nusax_senti --split valid
"""
import argparse
import os

import pandas as pd

from ..data import ind_source
from ..parsing import jev_answer
from ..paths import RUNS
from ..questions import FORMULATIONS, to_jev
from ..schema import write
from .common import RawLog, frame, items, load_env, now_utc, post_json, run_parallel

URL = "https://openrouter.ai/api/v1/systemone"
MODEL = "jev-1.13"


def _texts(arm, dataset, todo):
    """Replace item text according to the arm; A4 collapses to one item per sentence_id (lang='ind')."""
    if arm == "A1":
        return todo
    if arm == "A4":
        src = ind_source(dataset)
        seen, out = set(), []
        for it in todo:
            if it["sentence_id"] not in seen:
                seen.add(it["sentence_id"])
                out.append({**it, "lang": "ind", "text": src[it["sentence_id"]]})
        return out
    if arm == "A3":
        mt = pd.read_parquet(RUNS / "nllb" / f"{dataset}_{todo[0]['split']}.parquet")
        mt = {(r.lang, r.sentence_id): r.mt for r in mt.itertuples()}
        return [{**it, "text": mt[(it["lang"], it["sentence_id"])]} for it in todo if (it["lang"], it["sentence_id"]) in mt]
    raise ValueError(arm)


def run(arm, formulation, dataset, split, workers=8):
    load_env()
    key = os.environ["OPENROUTER_API_KEY"]
    run_id = f"{split}-{arm}-{formulation}-{dataset}"
    log = RawLog("jev", run_id)
    todo = _texts(arm, dataset, items(dataset, split))

    def call(it):
        body = {"model": MODEL, **to_jev(dataset, formulation, it["text"])}
        res = post_json(URL, key, body)
        return {"dataset": it["dataset"], "split": it["split"], "lang": it["lang"],
                "sentence_id": it["sentence_id"], "gold": it["gold"], "timestamp_utc": now_utc(), **res}

    records = run_parallel(todo, call, log, workers=workers)
    rows = []
    for r in records:
        resp = r["response"] if r["status"] == 200 else {}
        pred, probs, conf = jev_answer(resp, dataset, formulation)
        usage = resp.get("usage") or {}
        rows.append({"dataset": r["dataset"], "split": r["split"], "lang": r["lang"], "sentence_id": r["sentence_id"],
                     "gold": r["gold"], "pred": pred, "probs": probs, "confidence": conf,
                     "confidence_type": "model_prob" if probs is not None else "none",
                     "model_id": MODEL, "model_version": resp.get("model"), "timestamp_utc": r["timestamp_utc"],
                     "latency_ms": r["latency_ms"], "input_tokens": usage.get("input_tokens"),
                     "output_tokens": usage.get("output_tokens"), "cost_usd": usage.get("cost"),
                     "raw_response_path": str(log.path.relative_to(RUNS.parent)), "client_location": "ID"})
    df = frame(rows, run_id=run_id, arm=arm, condition=f"form={formulation}")
    write(df, RUNS / "jev" / f"{run_id}.parquet")
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m jevexp.runners.jev")
    ap.add_argument("--arm", choices=["A1", "A3", "A4"], required=True)
    ap.add_argument("--formulation", choices=list(FORMULATIONS), default="choice_en")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", choices=["valid", "test"], default="valid")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    df = run(a.arm, a.formulation, a.dataset, a.split, a.workers)
    print(f"{len(df)} rows, INVALID {(df['pred'] == 'INVALID').mean():.2%}, "
          f"cost ${df['cost_usd'].fillna(0).sum():.5f}, versions {sorted(df['model_version'].dropna().unique())}")


if __name__ == "__main__":
    main()
