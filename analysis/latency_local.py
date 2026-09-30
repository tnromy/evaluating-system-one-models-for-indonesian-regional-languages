"""Per-decision latency of the local fallback models on the laptop GPU (H3b cascade cost/latency; K60 exports, K71).

Batch 1, fp16, 20 warm-up + 200 NusaX test sentences stratified by language; time = tokenise + forward + softmax.
B2/B3: transformers sequence classifier. C1: `laya.Agent.system_one` (the official inference path).
Output -> results/latency_local.csv
"""
import platform
import tarfile
import time

import numpy as np
import pandas as pd
import torch

from jevexp.data import load
from jevexp.paths import RESULTS, ROOT
from jevexp.questions import to_jev
from jevexp.runners.common import guard_split

EXPORTS = ROOT / "models" / "f5"
N_WARM, N_ITEMS, SEED = 20, 200, 20260924


def texts():
    df = load("nusax_senti", "test")
    per = N_ITEMS // df["lang"].nunique() + 1
    pick = df.groupby("lang", group_keys=False).apply(lambda g: g.sample(per, random_state=SEED))
    return pick.sample(frac=1, random_state=SEED)["text"].tolist()[:N_ITEMS]


def extract(name):
    out = EXPORTS / name
    if not out.exists():
        with tarfile.open(EXPORTS / f"{name}.tar") as t:
            t.extractall(EXPORTS, filter="data")
    return out


def hf_predictor(path, dev):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, torch_dtype=torch.float16).to(dev).eval()

    @torch.no_grad()
    def predict(text):
        enc = tok(text, truncation=True, max_length=256, return_tensors="pt").to(dev)
        p = torch.softmax(model(**enc).logits.float(), -1).cpu()
        return p
    return predict


def laya_predictor(path, dev):
    from laya import Agent
    agent = Agent(str(path), device=str(dev))
    questions = to_jev("nusax_senti", "choice_en", "")["questions"]
    return lambda text: agent.system_one(text, questions)


def measure(predict, items, dev):
    for t in items[:N_WARM]:
        predict(t)
    ms = []
    for t in items:
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        predict(t)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1000)
    return np.asarray(ms)


def main():
    guard_split("test")
    items = texts()
    rows = []
    for arm, name, make in [("B2", "b2-nusax_senti-pooled-s0-nfull", hf_predictor),
                            ("B3", "b3-nusax_senti-pooled-s0-nfull", hf_predictor),
                            ("C1", "c1-nusax_senti-pooled-s0-nfull", laya_predictor)]:
        path = extract(name)
        for dev in [torch.device("cuda"), torch.device("cpu")] if torch.cuda.is_available() else [torch.device("cpu")]:
            try:
                predict = make(path, dev)
                ms = measure(predict, items, dev)
            except torch.OutOfMemoryError as e:  # 4 GB laptop GPU: recorded, then CPU
                rows.append({"arm": arm, "device": dev.type, "error": f"OOM: {str(e)[:80]}"})
                continue
            finally:
                predict = None
                torch.cuda.empty_cache()
            rows.append({"arm": arm, "export": name, "device": dev.type,
                         "device_name": torch.cuda.get_device_name() if dev.type == "cuda" else platform.processor(),
                         "n": len(ms), "mean_ms": ms.mean(), "p50_ms": np.median(ms), "p95_ms": np.quantile(ms, 0.95)})
            print(rows[-1])
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "latency_local.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
