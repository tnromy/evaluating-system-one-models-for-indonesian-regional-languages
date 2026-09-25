"""D3 (appendix): Claude via headless Claude Code CLI, one isolated call per item (K10, metodologi §4.1).

Each call runs from an empty directory with no tools, no settings, no MCP and no session persistence, so no
project context can leak in. The answer shape is enforced with --json-schema (answer in `structured_output`, K45). Confidence is verbalized only (no logprobs). Shadow cost = the answering model's
`costUSD` in `modelUsage` (the CLI's internal auxiliary call, e.g. Haiku, is excluded).

CLI:  python -m jevexp.runners.claude --dataset nusax_senti --model claude-sonnet-5 --limit 3
"""
import argparse
import json
import subprocess
import time

from ..data import DATASETS
from ..eval import INVALID
from ..parsing import llm_json
from ..paths import RUNS
from ..questions import output_schema, to_llm_messages
from ..schema import write
from .common import RawLog, frame, items, now_utc, run_parallel, run_tag

EMPTY_CWD = RUNS / "claude" / "cwd_empty"


def call_cli(model, system, user, schema, timeout=180):
    cmd = ["claude", "-p", "--model", model, "--system-prompt", system, "--tools", "", "--setting-sources", "",
           "--strict-mcp-config", "--no-session-persistence", "--output-format", "json",
           "--json-schema", json.dumps(schema)]
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, input=user, capture_output=True, text=True, timeout=timeout, cwd=EMPTY_CWD)
        out = json.loads(p.stdout) if p.stdout.strip() else {"error": p.stderr[-500:]}
        status = 0 if p.returncode == 0 and not out.get("is_error") else p.returncode or 1
    except (subprocess.TimeoutExpired, ValueError) as e:
        out, status = {"error": f"{type(e).__name__}: {e}"}, 1
    return {"status": status, "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "response": out}


def run(dataset, split, model, limit=None, workers=4, scope=None):
    EMPTY_CWD.mkdir(parents=True, exist_ok=True)
    run_id = f"{run_tag(split, scope)}-D3-{model}-{dataset}" + (f"-n{limit}" if limit else "")
    log = RawLog("claude", run_id)
    todo = items(dataset, split, scope)
    if limit:
        keep = sorted({it["sentence_id"] for it in todo})[:limit]
        todo = [it for it in todo if it["sentence_id"] in keep]
    labels = list(DATASETS[dataset]["labels"])

    def call(it):
        system, user = (m["content"] for m in to_llm_messages(dataset, it["text"]))
        res = call_cli(model, system, user, output_schema(dataset))
        return {"dataset": it["dataset"], "split": it["split"], "lang": it["lang"],
                "sentence_id": it["sentence_id"], "gold": it["gold"], "timestamp_utc": now_utc(), **res}

    records = run_parallel(todo, call, log, workers=workers, every=10)
    rows = []
    for r in records:
        resp = r["response"]
        answer = resp.get("structured_output")  # enforced by --json-schema (K45); `result` is empty then
        answer = json.dumps(answer) if isinstance(answer, dict) else resp.get("result")
        pred, probs = llm_json(answer, labels) if r["status"] == 0 else (INVALID, None)
        mu = (resp.get("modelUsage") or {}).get(model, {})
        rows.append({"dataset": r["dataset"], "split": r["split"], "lang": r["lang"], "sentence_id": r["sentence_id"],
                     "gold": r["gold"], "pred": pred, "probs": probs, "confidence": max(probs) if probs else None,
                     "confidence_type": "verbalized" if probs else "none", "model_id": model,
                     # the answering model only; the CLI's internal auxiliary call (e.g. Haiku) is excluded
                     "model_version": model if model in (resp.get("modelUsage") or {}) else
                     ",".join(sorted((resp.get("modelUsage") or {}).keys())) or None,
                     "timestamp_utc": r["timestamp_utc"], "latency_ms": r["latency_ms"],
                     # total prompt tokens: uncached + cache read + cache write (inputTokens alone excludes cache)
                     "input_tokens": sum(mu.get(k) or 0 for k in ("inputTokens", "cacheReadInputTokens",
                                                                   "cacheCreationInputTokens")) if mu else None,
                     "output_tokens": mu.get("outputTokens"),
                     "cost_usd": mu.get("costUSD"),
                     "raw_response_path": str(log.path.relative_to(RUNS.parent)), "client_location": "ID"})
    df = frame(rows, run_id=run_id, arm="D3", condition=f"model={model}")
    write(df, RUNS / "claude" / f"{run_id}.parquet")
    toks = df["input_tokens"].dropna()
    print(f"{len(df)} rows, INVALID {(df['pred'] == INVALID).mean():.2%}, input tokens min/max {toks.min()}/{toks.max()}, "
          f"shadow cost ${df['cost_usd'].fillna(0).sum():.4f}")
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m jevexp.runners.claude")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", choices=["valid", "test"], default="valid")
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--scope", choices=["pilot", "full", "subset"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    run(a.dataset, a.split, a.model, a.limit, a.workers, a.scope)


if __name__ == "__main__":
    main()
