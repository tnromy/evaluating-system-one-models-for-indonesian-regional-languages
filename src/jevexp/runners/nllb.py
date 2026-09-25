"""A3 translate-test: NLLB-200-distilled-600M, regional language -> Indonesian, local GPU (fp16).

Only languages covered by FLORES-200 are translated (J55). Quality is measured with chrF against the
human-written Indonesian parallel sentence (NusaX `ind`, NusaTranslation MT `ind_text`).

CLI:  python -m jevexp.runners.nllb --dataset nusax_senti --split valid
"""
import argparse
import time

import pandas as pd
import torch
from sacrebleu.metrics import CHRF
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ..data import ind_source
from ..paths import RUNS
from .common import items, run_tag

MODEL = "facebook/nllb-200-distilled-600M"
NLLB_CODES = {"ace": "ace_Latn", "ban": "ban_Latn", "bjn": "bjn_Latn", "bug": "bug_Latn",
              "jav": "jav_Latn", "min": "min_Latn", "sun": "sun_Latn"}
TARGET = "ind_Latn"


def translate(texts, src_code, tok, model, batch_size=16, max_length=256):
    tok.src_lang = src_code
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True, truncation=True,
                  max_length=max_length).to(model.device)
        with torch.inference_mode():
            gen = model.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(TARGET),
                                 max_length=max_length, num_beams=1, do_sample=False)
        out += tok.batch_decode(gen, skip_special_tokens=True)
    return out


def run(dataset, split, batch_size=16, scope=None):
    todo = [it for it in items(dataset, split, scope) if it["lang"] in NLLB_CODES]
    langs = sorted({it["lang"] for it in todo})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, dtype=torch.float16 if device == "cuda" else torch.float32)
    model.to(device).eval()
    ref = ind_source(dataset)
    rows, stats = [], []
    for lang in langs:
        sub = [it for it in todo if it["lang"] == lang]
        t0 = time.perf_counter()
        mt = translate([it["text"] for it in sub], NLLB_CODES[lang], tok, model, batch_size)
        secs = time.perf_counter() - t0
        refs = [ref[it["sentence_id"]] for it in sub]
        chrf = CHRF().corpus_score(mt, [refs]).score
        stats.append({"lang": lang, "n": len(sub), "chrf": round(chrf, 2), "sec_per_item": round(secs / len(sub), 4)})
        print(stats[-1], flush=True)
        rows += [{"lang": it["lang"], "sentence_id": it["sentence_id"], "src": it["text"], "mt": m, "ref_ind": r}
                 for it, m, r in zip(sub, mt, refs)]
    out = RUNS / "nllb"
    out.mkdir(parents=True, exist_ok=True)
    tag = run_tag(split, scope)
    pd.DataFrame(rows).to_parquet(out / f"{dataset}_{tag}.parquet", index=False)
    pd.DataFrame(stats).assign(model=MODEL, device=device).to_csv(out / f"{dataset}_{tag}_stats.csv", index=False)
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m jevexp.runners.nllb")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", choices=["valid", "test"], default="valid")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--scope", choices=["pilot", "full", "subset"])
    a = ap.parse_args(argv)
    run(a.dataset, a.split, a.batch_size, a.scope)


if __name__ == "__main__":
    main()
