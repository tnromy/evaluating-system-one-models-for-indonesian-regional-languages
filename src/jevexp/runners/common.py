"""Shared plumbing for API runners: .env, test-split guard, raw JSONL log with resume, retries, parallelism."""
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd
import requests

from ..data import DATASETS, load
from ..paths import PILOT, ROOT, RUNS, SUBSETS

PREREG_TAG = "prereg-v1"


def load_env():
    for line in (ROOT / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def guard_split(split):
    """Test data is only reachable after the pre-registration tag exists (docs/metodologi.md §1, K18)."""
    if split == "test":
        if not (ROOT / ".git").exists() and (ROOT / "PREREG_TAG").exists():
            # Colab bundle (no git): scripts/f5_bundle.sh writes PREREG_TAG only after checking the tag exists
            tags = (ROOT / "PREREG_TAG").read_text().split()[:1]
        else:
            tags = subprocess.run(["git", "-C", str(ROOT), "tag", "--list", PREREG_TAG],
                                  capture_output=True, text=True).stdout.split()
        if PREREG_TAG not in tags:
            raise PermissionError(f"test split is locked until git tag {PREREG_TAG} exists")


def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def items(dataset, split, scope=None, langs=None):
    """Items (dataset, split, lang, sentence_id, gold, text) of the parallel languages.

    scope: valid -> "pilot" (splits/pilot.json, the only valid scope);
           test  -> "full" (every item) or "subset" (splits/subsets.json), required (K50).
    """
    guard_split(split)
    scope = scope or ("pilot" if split == "valid" else None)
    allowed = {"valid": {"pilot"}, "test": {"full", "subset"}}[split]
    if scope not in allowed:
        raise ValueError(f"split={split} needs scope in {sorted(allowed)}, got {scope!r}")
    df = load(dataset, split=split, langs=langs or DATASETS[dataset]["parallel_langs"])
    if scope != "full":
        sel = json.loads((PILOT if scope == "pilot" else SUBSETS).read_text())
        df = df[df["sentence_id"].isin(set(sel[dataset]["sentence_ids"]))]
    return df.rename(columns={"label": "gold"}).to_dict("records")


def run_tag(split, scope):
    """Prefix of run ids: 'valid' for the pilot (unchanged), 'test-full' / 'test-subset' on test."""
    return split if split == "valid" else f"{split}-{scope}"


def post_json(url, key, body, timeout=60, retries=3):
    """POST with retries on network errors / 429 / 5xx. Returns dict(status, latency_ms, attempts, response)."""
    for attempt in range(1, retries + 1):
        t0 = time.perf_counter()
        try:
            r = requests.post(url, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                              json=body, timeout=timeout)
            status = r.status_code
            try:
                data = r.json()
            except ValueError:
                data = {"raw_text": r.text[:1000]}
        except requests.RequestException as e:
            status, data = None, {"error": f"{type(e).__name__}: {e}"}
        ms = (time.perf_counter() - t0) * 1000
        if status == 200 or (status is not None and status < 500 and status != 429):
            break
        time.sleep(2 ** attempt)
    return {"status": status, "latency_ms": round(ms, 1), "attempts": attempt, "response": data}


class RawLog:
    """Append-only JSONL, one line per completed call, written immediately (never lost on crash)."""

    def __init__(self, arm, run_id):
        self.path = RUNS / arm / "raw" / f"{run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def records(self):
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def done_keys(self):
        return {r["key"] for r in self.records()}

    def append(self, record):
        with self._lock, self.path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def item_key(it):
    return f"{it['dataset']}|{it['split']}|{it['lang']}|{it['sentence_id']}"


INFRA_PATTERNS = ("usage limit", "rate limit", "rate_limit", "overloaded", "quota", "timeout", "timed out",
                  "authentication", "unauthorized", "not logged in", "connection", "temporarily unavailable",
                  "does not support this model")


def is_infra_text(text):
    """True if an error message describes an infrastructure failure (no model answer was produced)."""
    text = (text or "").lower()
    return any(p in text for p in INFRA_PATTERNS)


def run_parallel(todo, fn, log, workers=8, every=50, max_infra=5):
    """Call fn(item) -> record for every item whose key is not yet in `log`; resume-safe.

    Records flagged `_infra` (quota / rate limit / network / auth: no model answer exists) are NOT written to the
    log, so a later resume retries them (K52); they go to `<run_id>.infra.jsonl` for audit. After `max_infra`
    consecutive infra failures the remaining queue is cancelled and the run stops cleanly.
    """
    done = log.done_keys()
    pending = [it for it in todo if item_key(it) not in done]
    print(f"{log.path.name}: {len(todo)} items, {len(todo) - len(pending)} already done, {len(pending)} to run",
          flush=True)
    infra_log = log.path.with_suffix(".infra.jsonl")
    t0, n, streak, n_infra, stopped = time.perf_counter(), 0, 0, 0, False
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fn, it): it for it in pending}
        for fut in as_completed(futures):
            if fut.cancelled():  # queue cancelled after an infra stop
                continue
            rec = fut.result()
            rec["key"] = item_key(futures[fut])
            if rec.get("_infra"):
                n_infra += 1
                streak += 1
                with log._lock, infra_log.open("a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if streak >= max_infra and not stopped:
                    stopped = True
                    for other in futures:
                        other.cancel()
                    print(f"  STOP: {streak} consecutive infra failures (quota/rate limit/network). "
                          f"Last: {str(rec.get('response'))[:200]}. Resume later.", flush=True)
                continue
            streak = 0
            log.append(rec)
            n += 1
            if n % every == 0 or n == len(pending):
                print(f"  {n}/{len(pending)}  {time.perf_counter() - t0:6.0f}s", flush=True)
    if n_infra:
        print(f"  infra failures (not recorded, will be retried on resume): {n_infra}", flush=True)
    if stopped:
        raise SystemExit(3)
    return log.records()


def frame(records, **const):
    return pd.DataFrame([{**const, **r} for r in records])
