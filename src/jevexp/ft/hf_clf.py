"""B2 (IndoBERT-base-p1) / B3 (XLM-R-base): sequence classification fine-tuning, NusaX protocol (Table 10):
Adam with a constant learning rate, at most 100 epochs, early stopping patience 3 on validation macro-F1.

Resumable: model, optimizer, scaler, RNG, best weights and history go into the rolling checkpoint.
"""
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..data import DATASETS
from .common import MAX_EPOCHS, PATIENCE, device, hf_kwargs, rng_state, set_rng_state
from .plan import MODELS

MAX_LEN = 256    # p99 of train lengths is <= 128 tokens for both tokenizers; 0.02 % exceed 256
EVAL_BS = 64


def load_model(arm, n_labels, ctx):
    repo, rev = MODELS[arm]
    tok = AutoTokenizer.from_pretrained(repo, revision=rev, **hf_kwargs(ctx))
    model = AutoModelForSequenceClassification.from_pretrained(repo, revision=rev, num_labels=n_labels,
                                                               **hf_kwargs(ctx))
    return tok, model


def encode(tok, texts):
    return tok(list(texts), truncation=True, max_length=MAX_LEN)["input_ids"]


def _batch(tok, seqs, dev):
    enc = tok.pad({"input_ids": seqs}, return_tensors="pt")
    return {k: v.to(dev) for k, v in enc.items()}


@torch.no_grad()
def predict(model, tok, seqs, dev):
    """Softmax probabilities in input order; length-sorted batches."""
    model.eval()
    order = np.argsort([len(s) for s in seqs], kind="stable")
    out = np.zeros((len(seqs), model.config.num_labels), dtype=np.float64)
    for i in range(0, len(seqs), EVAL_BS):
        idx = order[i:i + EVAL_BS]
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            logits = model(**_batch(tok, [seqs[j] for j in idx], dev)).logits
        out[idx] = torch.softmax(logits.float(), -1).cpu().numpy()
    return out


def macro_f1(gold_ids, probs, n_labels):
    return float(f1_score(gold_ids, probs.argmax(1), labels=list(range(n_labels)), average="macro"))


def train(run, data, lr, bs, ctx):
    """Returns dict(valid_probs, test_probs, valid_f1, best_epoch, epochs, history, train_time_s, model, tok)."""
    labels = list(DATASETS[run["dataset"]]["labels"])
    lab = {x: i for i, x in enumerate(labels)}
    train_df, valid_df, test_df = data
    dev = device()
    torch.manual_seed(run["seed"])  # also fixes the classification-head initialisation
    tok, model = load_model(run["arm"], len(labels), ctx)
    model.to(dev)
    xtr, xva = encode(tok, train_df["text"]), encode(tok, valid_df["text"])
    ytr = torch.tensor([lab[x] for x in train_df["label"]])
    yva = np.array([lab[x] for x in valid_df["label"]])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(dev.type, enabled=dev.type == "cuda")
    st = {"epoch": 0, "best_f1": -1.0, "best_epoch": 0, "bad": 0, "history": [], "train_time_s": 0.0,
          "best_state": None}
    ck = ctx["ckpt"].load()
    if ck:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        set_rng_state(ck["rng"])
        st = ck["st"]
        ctx["log"](f"  resumed at epoch {st['epoch']} (best valid F1 {st['best_f1']:.4f} @ {st['best_epoch']})")
    max_epochs = ctx.get("max_epochs", MAX_EPOCHS)
    while st["epoch"] < max_epochs and st["bad"] < PATIENCE:
        t0 = time.perf_counter()
        model.train()
        g = torch.Generator().manual_seed(run["seed"] * 100_003 + st["epoch"])
        perm = torch.randperm(len(xtr), generator=g).tolist()
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            batch = _batch(tok, [xtr[j] for j in idx], dev)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                loss = torch.nn.functional.cross_entropy(model(**batch).logits.float(), ytr[idx].to(dev))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        f1 = macro_f1(yva, predict(model, tok, xva, dev), len(labels))
        st["epoch"] += 1
        st["train_time_s"] += time.perf_counter() - t0
        st["history"].append({"epoch": st["epoch"], "valid_f1": round(f1, 5), "loss_last": round(loss.item(), 5)})
        if f1 > st["best_f1"]:
            st.update(best_f1=f1, best_epoch=st["epoch"], bad=0,
                      best_state={k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()})
        else:
            st["bad"] += 1
        ctx["heartbeat"](f"epoch {st['epoch']} valid F1 {f1:.4f} (best {st['best_f1']:.4f} @ {st['best_epoch']})")
        if st["bad"] < PATIENCE and st["epoch"] < max_epochs:
            ctx["ckpt"].maybe_save(lambda: {"model": model.state_dict(), "opt": opt.state_dict(),
                                            "scaler": scaler.state_dict(), "rng": rng_state(), "st": st})
    model.load_state_dict(st["best_state"])
    valid_probs = predict(model, tok, xva, dev)
    test_probs = None if test_df is None else predict(model, tok, encode(tok, test_df["text"]), dev)
    return {"valid_probs": valid_probs, "test_probs": test_probs, "valid_f1": st["best_f1"],
            "best_epoch": st["best_epoch"], "epochs": st["epoch"], "history": st["history"],
            "train_time_s": st["train_time_s"], "train_n": len(xtr), "model": model, "tok": tok}


def export(result, out_dir):
    """fp16 weights + tokenizer for laptop inference (< 1 GB VRAM)."""
    result["model"].half().save_pretrained(out_dir)
    result["tok"].save_pretrained(out_dir)
