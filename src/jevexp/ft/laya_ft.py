"""C1: Laya-multilingual fine-tuning, the official RLCD recipe
(NandhaKishorM/laya@9d955671 notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb, `train_ddp.py`)
adapted to one T4 (K60):

  * same model build/load, gradient checkpointing, fp16 autocast, GRPO-style logit noise (G=4, sigma 0.4->0.1),
    proper_reward(w_sph=0.75, w_rps=1.0) + soft CE, AdamW (encoder 2.5e-5, head 1e-4, wd 0.01), clip 1.0,
    cosine to 1e-6, 4 epochs, micro-batch 8;
  * one GPU: grad-accum 8 keeps the effective batch at 64 (2 GPUs x 8 x 4 in the recipe);
  * targets are one-hot gold labels (K13: never Jev outputs); the question is the shared choice spec;
  * the per-qtype temperature is fitted on the validation split (the recipe holds out a train slice; our
    validation split is never trained on, so the whole train split is used for training);
  * cosine T_max counts the real number of optimizer steps (ceil); the recipe's floor is 0 for tiny train sets;
  * the shuffle seed includes the run seed, so the 3 seeds differ in data order and noise.

C2 (zero-shot) uses the same encoder/inference path on the shipped checkpoint and its shipped temperatures.
Inference reproduces `laya.Agent._decode_answers` (same sequences, temperature rule and clamp) in full precision
instead of the Agent's 4-decimal rounded probabilities; `tests` / the smoke test check both agree.
"""
import json
import math
import os
import random
import shutil
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from ..data import DATASETS
from ..questions import to_jev
from .common import hf_kwargs, rng_state, set_rng_state
from .plan import MODELS

EPOCHS, MICRO_BATCH, GRAD_ACCUM, GROUP_SIZE = 4, 8, 8, 4
LR_ENCODER, LR_HEAD, SIGMA_START, SIGMA_END = 2.5e-5, 1.0e-4, 0.4, 0.1
EVAL_BS = 32


def base_dir(ctx):
    from huggingface_hub import snapshot_download
    from laya.agent import _fix_tokenizer_config
    repo, rev = MODELS["C1"]
    d = snapshot_download(repo, revision=rev, **hf_kwargs(ctx))
    _fix_tokenizer_config(d)
    return d


def load_base(model_dir, cfg_overrides=None):
    from laya.common import build_model
    from transformers import AutoTokenizer
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    cfg.update(cfg_overrides or {})
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    return cfg, tok, model


def question(dataset):
    from laya.agent import Agent
    qdef = to_jev(dataset, "choice_en", "")["questions"]["label"]
    return Agent._to_internal(qdef)


def build_items(tok, cfg, dataset, df, with_target=True):
    from laya.common import QTYPES, build_sequence
    q = question(dataset)
    labels = list(DATASETS[dataset]["labels"])
    assert list(q["crit"]) == labels
    items = []
    for text, gold in zip(df["text"], df["label"]):
        seq, markers = build_sequence(tok, {"text": text}, q, cfg["max_len"], cfg["head_max_len"])
        assert len(markers) == len(labels), "an option marker was truncated"
        it = {"ids": seq, "markers": markers, "qtype": QTYPES["choice"]}
        if with_target:
            it["target"] = [1.0 if lab == gold else 0.0 for lab in labels]
            it["label"] = labels.index(gold)
        items.append(it)
    return items


def collate(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, :len(it["ids"])] = torch.tensor(it["ids"])
        att[i, :len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        if "target" in it:
            target[i, :k] = torch.tensor(it["target"])
    return {"input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask,
            "target": target, "qtype": torch.tensor([it["qtype"] for it in items])}


@torch.no_grad()
def logits_of(model, tok, items, dev):
    """Raw option logits [n, k] in input order (length-sorted batches, fp16 autocast on CUDA)."""
    model.eval()
    order = np.argsort([len(it["ids"]) for it in items], kind="stable")
    k = len(items[0]["markers"])
    out = np.zeros((len(items), k), dtype=np.float64)
    for i in range(0, len(items), EVAL_BS):
        idx = order[i:i + EVAL_BS]
        b = collate([items[j] for j in idx], tok.pad_token_id)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            lg, _ = model(*(b[x].to(dev) for x in ("input_ids", "attention_mask", "marker_pos", "marker_mask",
                                                    "qtype")))
        out[idx] = lg.float().cpu().numpy()[:, :k]
    return out


def agent_temperature(cfg, k):
    """The scale `laya.Agent` applies to a choice question with k options."""
    from laya.common import QTYPES, clamp_temperature, temp_bucket
    by_opt = {b: clamp_temperature(v) for b, v in cfg.get("temperature_by_options", {}).items()}
    qt = QTYPES["choice"]
    return by_opt.get(temp_bucket(qt, k), clamp_temperature(cfg.get("temperature", [1.0] * 3)[qt]))


def probs_from_logits(z, t):
    z = z / t
    p = np.exp(z - z.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def fit_one_temp(sel):
    """Verbatim from the recipe: LBFGS on log T, soft-target NLL, clamp [0.1, 10]."""
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, :len(z)] = torch.tensor(z, dtype=torch.float32)
        T[i, :len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss
    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


def _dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def zeroshot(run, data, ctx):
    _, _, test_df = data
    dev = _dev()
    cfg, tok, model = load_base(base_dir(ctx))
    model.to(dev)
    items = build_items(tok, cfg, run["dataset"], test_df, with_target=False)
    z = logits_of(model, tok, items, dev)
    t = agent_temperature(cfg, z.shape[1])
    return {"valid_probs": None, "test_probs": probs_from_logits(z, t), "train_n": 0, "train_time_s": 0.0,
            "temperature": t, "epochs": 0}


def train(run, data, ctx):
    train_df, valid_df, test_df = data
    dev = _dev()
    amp = dev.type == "cuda"
    src = base_dir(ctx)
    cfg, tok, model = load_base(src, {"gradient_checkpointing": True, "max_tokens_per_batch": 4096,
                                      "max_len": 1024, "head_max_len": 256})
    if amp:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True
    model.to(dev)
    tr = build_items(tok, cfg, run["dataset"], train_df)
    va = build_items(tok, cfg, run["dataset"], valid_df)
    epochs = ctx.get("max_epochs") or EPOCHS
    enc = [p for n, p in model.named_parameters() if "encoder." in n]
    head = [p for n, p in model.named_parameters() if "encoder." not in n]
    opt = torch.optim.AdamW([{"params": enc, "lr": LR_ENCODER}, {"params": head, "lr": LR_HEAD}], weight_decay=0.01)
    steps_per_epoch = math.ceil(math.ceil(len(tr) / MICRO_BATCH) / GRAD_ACCUM)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps_per_epoch * epochs), eta_min=1e-6)
    scaler = torch.amp.GradScaler(dev.type, enabled=amp)
    torch.manual_seed(run["seed"])
    st = {"epoch": 0, "history": [], "train_time_s": 0.0}
    ck = ctx["ckpt"].load()
    if ck:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        set_rng_state(ck["rng"])
        st = ck["st"]
        ctx["log"](f"  resumed at epoch {st['epoch']}/{epochs}")
    while st["epoch"] < epochs:
        epoch, t0 = st["epoch"], time.perf_counter()
        model.train()
        order = list(range(len(tr)))
        random.Random(run["seed"] * 1000 + 42 + epoch).shuffle(order)
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * (epoch / max(1, epochs - 1))
        opt.zero_grad(set_to_none=True)
        tot, nb, accum = 0.0, 0, 0
        for b_idx in range(0, len(order), MICRO_BATCH):
            batch = collate([tr[i] for i in order[b_idx:b_idx + MICRO_BATCH]], tok.pad_token_id)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                logits, act = model(*(batch[x].to(dev) for x in ("input_ids", "attention_mask", "marker_pos",
                                                                  "marker_mask", "qtype")))
            loss = _rlcd_loss(logits.float(), act, batch, dev, sigma)
            scaler.scale(loss / GRAD_ACCUM).backward()
            accum += 1
            if accum % GRAD_ACCUM == 0 or b_idx + MICRO_BATCH >= len(order):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
            tot += loss.item()
            nb += 1
        st["epoch"] += 1
        st["train_time_s"] += time.perf_counter() - t0
        st["history"].append({"epoch": st["epoch"], "avg_loss": round(tot / max(1, nb), 5)})
        ctx["heartbeat"](f"epoch {st['epoch']}/{epochs} loss {tot / max(1, nb):.4f}")
        if st["epoch"] < epochs:
            ctx["ckpt"].maybe_save(lambda: {"model": model.state_dict(), "opt": opt.state_dict(),
                                            "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                                            "rng": rng_state(), "st": st})
    del opt, scaler, sched
    if amp:
        torch.cuda.empty_cache()
    zv = logits_of(model, tok, va, dev)
    temp = [1.2, 1.2, 1.2]  # recipe defaults for the question types without calibration data
    temp[0] = fit_one_temp([(z, it["target"]) for z, it in zip(zv, va)])
    cfg.update(fine_tuned=True, model_name=f"laya-jevexp-{run['run_id']}", temperature=temp, amp_dtype="fp16")
    cfg.pop("temperature_by_options", None)
    for k in ("gradient_checkpointing", "max_tokens_per_batch"):
        cfg.pop(k, None)
    t = agent_temperature(cfg, zv.shape[1])
    zt = logits_of(model, tok, build_items(tok, cfg, run["dataset"], test_df, with_target=False), dev)
    return {"valid_probs": probs_from_logits(zv, t), "test_probs": probs_from_logits(zt, t),
            "temperature": temp[0], "temperature_applied": t, "epochs": st["epoch"], "history": st["history"],
            "train_time_s": st["train_time_s"], "train_n": len(tr), "model": model, "tok": tok, "cfg": cfg,
            "src": src}


def _rlcd_loss(logits, act, batch, dev, sigma):
    """The recipe's loss: policy gradient on noisy logit samples scored by proper_reward + soft CE."""
    from laya.common import proper_reward
    mask = batch["marker_mask"].to(dev)
    k = mask.sum(-1, keepdim=True).float()
    target = batch["target"].to(dev)
    qtype = batch["qtype"].to(dev)
    eps = torch.randn((GROUP_SIZE,) + logits.shape, device=dev) * sigma * mask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
    with torch.no_grad():
        r = proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph=0.75, w_rps=1.0)
        adv = r - r.mean(0, keepdim=True)
        adv = adv / (adv.std() + 1e-6)
    logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
    loss_rl = -(adv * logp).mean()
    loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
    return loss_rl + 1.0 * loss_ce + 0.0 * act.sum()


def export(result, out_dir):
    """A `laya.Agent`-loadable directory (fp16 weights, fitted temperature)."""
    os.makedirs(out_dir, exist_ok=True)
    sd = {k: v.half().contiguous().cpu() for k, v in result["model"].state_dict().items()}
    save_file(sd, os.path.join(out_dir, "model.safetensors"))
    result["model"].encoder.config.save_pretrained(os.path.join(out_dir, "encoder"))
    shutil.copytree(os.path.join(result["src"], "tokenizer"), os.path.join(out_dir, "tokenizer"), dirs_exist_ok=True)
    with open(os.path.join(out_dir, "rl_agent_config.json"), "w") as f:
        json.dump(result["cfg"], f, indent=2)
