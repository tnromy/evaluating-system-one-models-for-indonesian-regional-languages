"""DeepSeek arms, direct API (not OpenRouter; K34-K36).

D1 = deepseek-flash, thinking disabled, temperature 1, logprobs -> label distribution (prediction = argmax).
D2 = deepseek-flash thinking / deepseek-v4-pro non-thinking (subsets).
JSON invalid or label outside the options -> INVALID (K35). If no label token carries logprobs, the JSON
`choice` is used and the fallback is counted.

CLI:  python -m jevexp.runners.deepseek --dataset nusax_senti --model deepseek-flash [--thinking] [--limit N]
"""
import argparse
import os

import requests

from ..data import DATASETS
from ..eval import INVALID
from ..parsing import llm_json, logprob_label_dist
from ..paths import RUNS
from ..questions import to_llm_messages
from ..schema import write
from .common import RawLog, frame, items, load_env, now_utc, post_json, run_parallel, run_tag

URL = "https://api.deepseek.com/chat/completions"
# USD per 1M tokens at PEAK rates (conservative upper bound; off-peak is half). api-docs.deepseek.com, 2026-09-25.
PRICES = {"deepseek-flash": {"hit": 0.006, "miss": 0.3, "out": 1.2},
          "deepseek-v4-pro": {"hit": 0.044, "miss": 1.32, "out": 3.96}}


def cost_upper(model, usage):
    p = PRICES[model]
    hit = usage.get("prompt_cache_hit_tokens", 0) or 0
    miss = usage.get("prompt_cache_miss_tokens", usage.get("prompt_tokens", 0)) or 0
    return (hit * p["hit"] + miss * p["miss"] + (usage.get("completion_tokens", 0) or 0) * p["out"]) / 1e6


def balance(key):
    r = requests.get("https://api.deepseek.com/user/balance", headers={"Authorization": f"Bearer {key}"}, timeout=30)
    return float(r.json()["balance_infos"][0]["total_balance"])


def body(dataset, text, model, thinking):
    b = {"model": model, "messages": to_llm_messages(dataset, text), "response_format": {"type": "json_object"},
         "logprobs": True, "top_logprobs": 20, "thinking": {"type": "enabled" if thinking else "disabled"}}
    if not thinking:
        b["temperature"] = 1.0
    return b


def parse(resp, labels):
    """-> pred, probs, confidence_type, used_fallback"""
    ch = ((resp or {}).get("choices") or [{}])[0]
    choice, _verbalized = llm_json((ch.get("message") or {}).get("content"), labels)
    if choice == INVALID:
        return INVALID, None, "none", False
    dist = logprob_label_dist(ch, labels)
    if dist is None:
        return choice, None, "none", True
    return labels[max(range(len(labels)), key=lambda i: dist[i])], dist, "logprob", False


def run(dataset, split, model, thinking, limit=None, workers=8, scope=None):
    load_env()
    key = os.environ["DEEPSEEK_API_KEY"]
    arm = "D2" if (thinking or model != "deepseek-flash") else "D1"
    tag = f"{model}-{'think' if thinking else 'nothink'}"
    run_id = f"{run_tag(split, scope)}-{arm}-{tag}-{dataset}" + (f"-n{limit}" if limit else "")
    log = RawLog("deepseek", run_id)
    todo = items(dataset, split, scope)
    if limit:  # first `limit` sentence_ids (all languages), deterministic
        keep = sorted({it["sentence_id"] for it in todo})[:limit]
        todo = [it for it in todo if it["sentence_id"] in keep]
    labels = list(DATASETS[dataset]["labels"])
    before = balance(key)

    def call(it):
        res = post_json(URL, key, body(dataset, it["text"], model, thinking), timeout=180 if thinking else 60)
        return {"dataset": it["dataset"], "split": it["split"], "lang": it["lang"],
                "sentence_id": it["sentence_id"], "gold": it["gold"], "timestamp_utc": now_utc(), **res}

    records = run_parallel(todo, call, log, workers=workers)
    rows, fallbacks = [], 0
    for r in records:
        resp = r["response"] if r["status"] == 200 else {}
        pred, probs, ctype, fb = parse(resp, labels)
        fallbacks += fb
        u = resp.get("usage") or {}
        rows.append({"dataset": r["dataset"], "split": r["split"], "lang": r["lang"], "sentence_id": r["sentence_id"],
                     "gold": r["gold"], "pred": pred, "probs": probs,
                     "confidence": max(probs) if probs else None, "confidence_type": ctype,
                     "model_id": model, "model_version": resp.get("system_fingerprint"),
                     "timestamp_utc": r["timestamp_utc"], "latency_ms": r["latency_ms"],
                     "input_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens"),
                     "cost_usd": cost_upper(model, u) if u else None,
                     "raw_response_path": str(log.path.relative_to(RUNS.parent)), "client_location": "ID"})
    after = balance(key)
    df = frame(rows, run_id=run_id, arm=arm, condition=f"model={model};thinking={'on' if thinking else 'off'}")
    write(df, RUNS / "deepseek" / f"{run_id}.parquet")
    print(f"{len(df)} rows, INVALID {(df['pred'] == INVALID).mean():.2%}, logprob fallback {fallbacks}, "
          f"balance {before:.4f} -> {after:.4f} (spent ~${before - after:.4f}; billing may lag)")
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m jevexp.runners.deepseek")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", choices=["valid", "test"], default="valid")
    ap.add_argument("--model", choices=["deepseek-flash", "deepseek-v4-pro"], default="deepseek-flash")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--scope", choices=["pilot", "full", "subset"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    run(a.dataset, a.split, a.model, a.thinking, a.limit, a.workers, a.scope)


if __name__ == "__main__":
    main()
