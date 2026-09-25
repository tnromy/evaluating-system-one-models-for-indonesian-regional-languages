"""D4 (LLM reference ladder): OpenAI GPT models via headless Codex CLI (`codex exec`), one isolated call per item.

Isolation (K44): empty cwd, read-only sandbox, --ephemeral, --ignore-user-config, --ignore-rules, every enabled
feature disabled, and the harness instructions replaced by `model_instructions_file`. A residual harness context
(~7k input tokens) cannot be removed; it is identical for every item and is reported as a limitation.
The answer shape is enforced with --output-schema. Confidence is verbalized only. Reasoning effort = "low" (the
minimum; GPT models in Codex have no non-thinking mode).

CLI:  python -m jevexp.runners.codex --dataset nusax_senti --model gpt-6-luna --limit 3
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
from .common import RawLog, frame, items, now_utc, run_parallel

BASE = RUNS / "codex"
EMPTY_CWD = BASE / "cwd_empty"
INSTRUCTIONS = BASE / "instructions.md"
INSTRUCTIONS_TEXT = "You are a text classifier. Follow the user's instructions exactly and answer with JSON only.\n"
EFFORT = "low"


def enabled_features():
    out = subprocess.run(["codex", "features", "list"], capture_output=True, text=True).stdout
    return [ln.split()[0] for ln in out.splitlines() if ln.split() and ln.split()[-1] == "true"]


def parse_events(stdout):
    """Codex --json event stream -> final message, usage, tool-call count, error messages."""
    msg, usage, tools, errors = None, {}, 0, []
    for line in stdout.splitlines():
        if not line.strip().startswith("{"):
            continue
        e = json.loads(line)
        if e.get("type") == "item.completed":
            it = e.get("item") or {}
            if it.get("type") == "agent_message":
                msg = it.get("text")
            elif it.get("type") == "error":
                errors.append(it.get("message", "")[:200])
            elif it.get("type") not in ("reasoning",):
                tools += 1  # command_execution, file_change, mcp_tool_call, web_search, ...
        elif e.get("type") == "turn.completed":
            usage = e.get("usage") or {}
        elif e.get("type") in ("turn.failed", "error"):
            errors.append(json.dumps(e)[:200])
    return msg, usage, tools, errors


def call_cli(model, dataset, prompt, disabled, timeout=300):
    cmd = ["codex", "exec", "-m", model, "-c", f'model_reasoning_effort="{EFFORT}"',
           "-c", f'model_instructions_file="{INSTRUCTIONS}"', *sum((["--disable", f] for f in disabled), []),
           "--output-schema", str(BASE / f"schema_{dataset}.json"), "--json", "--ephemeral",
           "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check", "-s", "read-only",
           "-C", str(EMPTY_CWD), "-"]
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout)
        msg, usage, tools, errors = parse_events(p.stdout)
        status = 0 if p.returncode == 0 and msg is not None else (p.returncode or 1)
        out = {"result": msg, "usage": usage, "tool_calls": tools, "errors": errors, "stderr": p.stderr[-300:]}
    except subprocess.TimeoutExpired as e:
        status, out = 1, {"error": f"timeout: {e}"}
    return {"status": status, "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "response": out}


def run(dataset, split, model, limit=None, workers=4):
    EMPTY_CWD.mkdir(parents=True, exist_ok=True)
    INSTRUCTIONS.write_text(INSTRUCTIONS_TEXT)
    (BASE / f"schema_{dataset}.json").write_text(json.dumps(output_schema(dataset)))
    disabled = enabled_features()
    run_id = f"{split}-D4-{model}-{dataset}" + (f"-n{limit}" if limit else "")
    log = RawLog("codex", run_id)
    todo = items(dataset, split)
    if limit:
        keep = sorted({it["sentence_id"] for it in todo})[:limit]
        todo = [it for it in todo if it["sentence_id"] in keep]
    labels = list(DATASETS[dataset]["labels"])

    def call(it):
        system, user = (m["content"] for m in to_llm_messages(dataset, it["text"]))
        res = call_cli(model, dataset, f"{system}\n\n{user}", disabled)
        return {"dataset": it["dataset"], "split": it["split"], "lang": it["lang"],
                "sentence_id": it["sentence_id"], "gold": it["gold"], "timestamp_utc": now_utc(), **res}

    records = run_parallel(todo, call, log, workers=workers, every=10)
    rows = []
    for r in records:
        resp = r["response"]
        pred, probs = llm_json(resp.get("result"), labels) if r["status"] == 0 else (INVALID, None)
        u = resp.get("usage") or {}
        rows.append({"dataset": r["dataset"], "split": r["split"], "lang": r["lang"], "sentence_id": r["sentence_id"],
                     "gold": r["gold"], "pred": pred, "probs": probs, "confidence": max(probs) if probs else None,
                     "confidence_type": "verbalized" if probs else "none", "model_id": model,
                     "model_version": f"{model};effort={EFFORT}", "timestamp_utc": r["timestamp_utc"],
                     "latency_ms": r["latency_ms"], "input_tokens": u.get("input_tokens"),
                     "output_tokens": (u.get("output_tokens") or 0) + (u.get("reasoning_output_tokens") or 0),
                     "raw_response_path": str(log.path.relative_to(RUNS.parent)), "client_location": "ID"})
    df = frame(rows, run_id=run_id, arm="D4", condition=f"model={model};effort={EFFORT}")
    write(df, RUNS / "codex" / f"{run_id}.parquet")
    tools = sum(r["response"].get("tool_calls", 0) for r in records)
    toks = df["input_tokens"].dropna()
    print(f"{len(df)} rows, INVALID {(df['pred'] == INVALID).mean():.2%}, tool calls {tools}, "
          f"input tokens min/max {toks.min()}/{toks.max()}, latency p50 {df['latency_ms'].median():.0f} ms")
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m jevexp.runners.codex")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", choices=["valid", "test"], default="valid")
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    run(a.dataset, a.split, a.model, a.limit, a.workers)


if __name__ == "__main__":
    main()
