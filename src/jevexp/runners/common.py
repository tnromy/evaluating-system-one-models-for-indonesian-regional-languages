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


def run_parallel(todo, fn, log, workers=8, every=50):
    """Call fn(item) -> record for every item whose key is not yet in `log`; resume-safe."""
    done = log.done_keys()
    pending = [it for it in todo if item_key(it) not in done]
    print(f"{log.path.name}: {len(todo)} items, {len(todo) - len(pending)} already done, {len(pending)} to run",
          flush=True)
    t0, n = time.perf_counter(), 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fn, it): it for it in pending}
        for fut in as_completed(futures):
            rec = fut.result()
            rec["key"] = item_key(futures[fut])
            log.append(rec)
            n += 1
            if n % every == 0 or n == len(pending):
                print(f"  {n}/{len(pending)}  {time.perf_counter() - t0:6.0f}s", flush=True)
    return log.records()


def frame(records, **const):
    return pd.DataFrame([{**const, **r} for r in records])
