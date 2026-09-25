"""Smoke test of the two paid APIs (F2 prerequisite). Uses hand-written sentences only, never dataset items.

  Jev      via OpenRouter (OPENROUTER_API_KEY is for Jev only): /api/v1/systemone and /api/alpha/decisions
  DeepSeek via api.deepseek.com (DEEPSEEK_API_KEY): deepseek-flash (non-thinking / thinking), deepseek-v4-pro

Raw results -> runs/smoke/<timestamp>.json. Keys are read from .env and never printed.
Run:  .venv/bin/python scripts/smoke_api.py
"""
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]

OPTIONS = {
    "negative": "The text expresses a negative opinion or feeling.",
    "neutral": "The text is factual or expresses no clear opinion.",
    "positive": "The text expresses a positive opinion or feeling.",
}
INSTRUCTIONS = "What is the sentiment of the text?"
SENTENCES = {
    "ind": "Pelayanan hotel ini sangat ramah dan kamarnya bersih, saya pasti kembali lagi.",
    "jav": "Panganane ora enak lan larang banget, aku kuciwa.",
}


def load_env():
    for line in (ROOT / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def post(url, key, body, timeout=60):
    t0 = time.perf_counter()
    try:
        r = requests.post(url, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                          json=body, timeout=timeout)
        status = r.status_code
        try:
            data = r.json()
        except ValueError:
            data = {"raw_text": r.text[:500]}
    except requests.RequestException as e:
        status, data = None, {"error": f"{type(e).__name__}: {e}"}
    ms = (time.perf_counter() - t0) * 1000
    print(f"  {status} {ms:8.0f} ms  {url.split('//')[1]}  {body.get('model')}", flush=True)
    return {"status": status, "latency_ms": round(ms, 1), "response": data}


def jev_body(model, text):
    return {"model": model, "state": {"text": text}, "questions": {
        "sentiment": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": OPTIONS},
        "is_positive": {"type": "noul", "instructions": "Does the text express a positive sentiment?"},
    }}


def deepseek_body(model, text, thinking):
    system = ("You classify text. Answer ONLY with a JSON object of the form "
              '{"choice": "<option>", "probabilities": {"negative": p, "neutral": p, "positive": p}} '
              "where <option> is one of: negative, neutral, positive, and the probabilities sum to 1.")
    user = (f"{INSTRUCTIONS}\nOptions:\n" + "\n".join(f"- {k}: {v}" for k, v in OPTIONS.items())
            + f"\n\nText: {text}")
    body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_object"}, "logprobs": True, "top_logprobs": 20,
            "thinking": {"type": "enabled" if thinking else "disabled"}}
    if not thinking:
        body["temperature"] = 0
    return body


def label_distribution(choice):
    """Distribution over OPTIONS from top_logprobs at the first token that starts a label (log-sum-exp)."""
    content = ((choice.get("logprobs") or {}).get("content")) or []
    for tok in content:
        head = tok["token"].strip().strip('"')
        if not head or not any(lab.startswith(head) for lab in OPTIONS):
            continue
        cands = {}  # variants like "positive" and " positive" collapse to one key: keep the most likely
        for c in tok.get("top_logprobs", []):
            key = c["token"].strip().strip('"')
            cands[key] = max(cands.get(key, -math.inf), c["logprob"])
        hits = {}
        for lab in OPTIONS:
            lps = [lp for t, lp in cands.items() if t and lab.startswith(t)]
            if lps:
                hits[lab] = max(lps)
        if not hits:
            return None, tok["token"]
        m = max(hits.values())
        z = sum(math.exp(v - m) for v in hits.values())
        return {lab: round(math.exp(v - m) / z, 4) for lab, v in hits.items()}, tok["token"]
    return None, None


def main():
    load_env()
    or_key, ds_key = os.environ["OPENROUTER_API_KEY"], os.environ["DEEPSEEK_API_KEY"]
    path = ROOT / "runs" / "smoke" / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "jev": {}, "deepseek": {}}

    def save():  # raw responses are persisted after every call, before any parsing
        path.write_text(json.dumps(out, indent=1, ensure_ascii=False))

    print("Jev via OpenRouter", flush=True)
    sysone = "https://openrouter.ai/api/v1/systemone"
    out["jev"]["systemone_ind"] = post(sysone, or_key, jev_body("jev-1.13", SENTENCES["ind"])); save()
    out["jev"]["decisions_ind"] = post("https://openrouter.ai/api/alpha/decisions", or_key,
                                       jev_body("typesafe/jev-1.13", SENTENCES["ind"])); save()
    out["jev"]["systemone_ind_repeats"] = []
    for _ in range(3):
        out["jev"]["systemone_ind_repeats"].append(post(sysone, or_key, jev_body("jev-1.13", SENTENCES["ind"])))
        save()
    out["jev"]["systemone_jav"] = post(sysone, or_key, jev_body("jev-1.13", SENTENCES["jav"])); save()

    print("DeepSeek direct", flush=True)
    chat = "https://api.deepseek.com/chat/completions"
    for name, model, thinking in [("flash_nonthinking", "deepseek-flash", False),
                                  ("flash_thinking", "deepseek-flash", True),
                                  ("pro_nonthinking", "deepseek-v4-pro", False)]:
        out["deepseek"][name] = post(chat, ds_key, deepseek_body(model, SENTENCES["ind"], thinking),
                                     timeout=180 if thinking else 60)
        save()

    for name, res in out["deepseek"].items():
        try:
            ch = (res["response"].get("choices") or [{}])[0]
            dist, tok = label_distribution(ch)
            res["summary"] = {
                "model": res["response"].get("model"), "fingerprint": res["response"].get("system_fingerprint"),
                "content": (ch.get("message") or {}).get("content"),
                "has_reasoning": bool((ch.get("message") or {}).get("reasoning_content")),
                "has_logprobs": bool((ch.get("logprobs") or {}).get("content")),
                "logprob_label_distribution": dist, "label_token": tok, "usage": res["response"].get("usage"),
            }
        except Exception as e:  # parsing must never lose the raw data
            res["summary"] = {"parse_error": f"{type(e).__name__}: {e}"}
        save()
    print(path)


if __name__ == "__main__":
    main()
