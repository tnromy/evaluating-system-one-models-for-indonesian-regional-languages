"""Idempotent F5 worker (K60). The store (MinIO on Colab) is the only source of truth:

  runs/<id>/done.json        written last; a run with it is never redone
  runs/<id>/pred_test.parquet, pred_valid.parquet, meta.json
  runs/<id>/ckpt/latest.pt   rolling mid-run checkpoint, deleted once done
  runs/<id>/lease.json       {session, t}: with shared=True, another live worker holds this run (two Colab
                             sessions sharing one stage); a single session ignores leases, so a restart after a
                             crash resumes its own interrupted run at once
  runs/<id>/error.json       failed attempts; retried once, then skipped and reported
  export/<id>.tar            representative checkpoints for the laptop
  plan/active.json           plan + code hashes; a worker with different code refuses to start
  status/progress.{txt,json} last status block printed by the worker (progress + ETA, K63)

Kill it anywhere, start it again: finished runs are skipped, a checkpointed run resumes from its last epoch.
"""
import hashlib
import io
import json
import tarfile
import time
import traceback
import uuid
from collections import defaultdict
from pathlib import Path

import pandas as pd

from ..data import DATASETS
from ..paths import ROOT
from ..schema import validate
from . import hf_clf, laya_ft
from .common import Checkpoint, env_versions, gpu_name, now_utc, pred_frame, run_data
from .plan import MODELS, load_plan
from .progress import Progress

LEASE_S = 1800
MAX_ATTEMPTS = 2


def code_sha():
    h = hashlib.sha256()
    for p in sorted((ROOT / "src" / "jevexp").rglob("*.py")):
        h.update(str(p.relative_to(ROOT)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _parquet_bytes(df):
    buf = io.BytesIO()
    validate(df).to_parquet(buf, index=False)
    return buf.getvalue()


class Worker:
    def __init__(self, store, work="work", stage=1, hf_token=None, cache_dir=None, ckpt_every_s=900,
                 max_epochs=None, plan_path=None, shared=False, status_every_s=600, out_dir=None, log=print):
        self.store, self.work, self.stage = store, Path(work), stage
        self.plan, self.plan_sha = load_plan(plan_path) if plan_path else load_plan()
        self.code_sha = code_sha()
        self.session = uuid.uuid4().hex[:12]
        self.ctx_base = {"hf_token": hf_token, "cache_dir": cache_dir, "max_epochs": max_epochs, "log": log}
        self.ckpt_every_s, self.log, self.shared = ckpt_every_s, log, shared
        self.status_every_s, self.out_dir = status_every_s, Path(out_dir) if out_dir else None
        self.progress = Progress(store, self.plan)
        self.session_info = {"id": self.session, "started": time.time(), "runs": 0}
        self.current = None
        self._last_status = 0.0

    # ---------------------------------------------------------------- state
    def runs(self, stages=None):
        stages = stages or ([self.stage] if isinstance(self.stage, int) else self.stage)
        return [r for r in self.plan["runs"] if r["stage"] in stages]

    def is_done(self, rid):
        return self.store.exists(f"runs/{rid}/done.json")

    def status(self, stages=None):
        """Per (stage, kind, arm): done / total, plus runs that failed MAX_ATTEMPTS times."""
        done = {k.split("/")[1] for k in self.store.list("runs/") if k.endswith("/done.json")}
        errors = {k.split("/")[1] for k in self.store.list("runs/") if k.endswith("/error.json")}
        rows = {}
        for r in self.runs(stages):
            key = (r["stage"], r["kind"], r["arm"])
            d = rows.setdefault(key, {"stage": r["stage"], "kind": r["kind"], "arm": r["arm"], "done": 0,
                                      "total": 0, "failed": 0})
            d["total"] += 1
            d["done"] += r["run_id"] in done
            d["failed"] += r["run_id"] in errors and r["run_id"] not in done
        return pd.DataFrame(rows.values())

    def check_plan(self, allow_new_code=False):
        """Refuse to mix code versions inside one F5 collection unless explicitly allowed (then log a K entry)."""
        active = self.store.get_json("plan/active.json")
        mine = {"plan_sha": self.plan_sha, "code_sha": self.code_sha}
        if active and {k: active[k] for k in mine} != mine:
            if not allow_new_code:
                raise RuntimeError(f"store was started with plan {active['plan_sha'][:12]} / code "
                                   f"{active['code_sha'][:12]}; this bundle is plan {self.plan_sha[:12]} / code "
                                   f"{self.code_sha[:12]}. Upload the matching bundle, or set ALLOW_NEW_CODE=True "
                                   "after logging a decision.")
            old = {k: active[k] for k in (*mine, "since")}
            self.store.put_json("plan/active.json", {**mine, "since": now_utc(),
                                                    "history": active.get("history", []) + [old]})
        elif not active:
            self.store.put_json("plan/active.json", {**mine, "since": now_utc(), "history": []})
        if not self.store.exists(f"plan/{self.plan_sha}.json"):
            self.store.put_bytes(f"plan/{self.plan_sha}.json", json.dumps(self.plan).encode())
        return mine

    def _lease_ok(self, rid):
        if not self.shared:
            return True
        lease = self.store.get_json(f"runs/{rid}/lease.json")
        return not lease or lease["session"] == self.session or time.time() - lease["t"] > LEASE_S

    def _take_lease(self, rid):
        self.store.put_json(f"runs/{rid}/lease.json", {"session": self.session, "t": time.time()})

    def best_config(self, arm):
        """Grid winner for B2/B3: highest validation macro-F1 on NusaX ind, ties -> first in plan order."""
        cells = [r for r in self.plan["runs"] if r["kind"] == "grid" and r["arm"] == arm]
        res = [self.store.get_json(f"runs/{r['run_id']}/done.json") for r in cells]
        if any(x is None for x in res):
            return None
        best = max(range(len(res)), key=lambda i: (res[i]["valid_f1"], -i))
        return {"lr": cells[best]["lr"], "bs": cells[best]["bs"], "grid_run": cells[best]["run_id"],
                "valid_f1": res[best]["valid_f1"]}

    # ---------------------------------------------------------------- loop
    def print_status(self):
        """Print the progress block (console) and publish it to the store + out_dir. Never fails the run."""
        try:
            text, data = self.progress.report(stage=self.stage, current=self.current, session=self.session_info)
        except Exception as e:  # a status hiccup (network) must not stop training
            self.log(f"(status tidak tersedia: {type(e).__name__}: {e})")
            return
        self.log(text)
        self._last_status = time.time()
        try:
            self.store.put_bytes("status/progress.txt", text.encode("utf-8"))
            self.store.put_json("status/progress.json", data)
            if self.out_dir:
                self.out_dir.mkdir(parents=True, exist_ok=True)
                (self.out_dir / "progress.txt").write_text(text + "\n")
        except Exception as e:
            self.log(f"(status tidak tersimpan: {type(e).__name__}: {e})")

    def run_all(self, start_stage=1, auto_next=True):
        """Keep-alive entry point for the last notebook cell: stage after stage until nothing is left (K63).
        Results are collected into out_dir after every stage, so f5_results.zip is always current."""
        for st in range(start_stage, 4 if auto_next else start_stage + 1):
            self.stage = st
            if self.run() == 0:  # nothing new in this stage: no need to collect again (keeps the GPU busy)
                continue
            if self.out_dir:
                try:
                    self.collect(self.out_dir)
                except Exception as e:
                    self.log(f"(collect gagal, coba Cell kumpulkan manual: {type(e).__name__}: {e})")
        self.current = None
        self.log("SEMUA STAGE SELESAI untuk sesi ini (run yang gagal permanen tercantum di status)."
                 if auto_next else f"Stage {self.stage} selesai.")
        self.print_status()

    def run(self, limit=None):
        todo = [r for r in self.runs() if not self.is_done(r["run_id"])]
        self.log(f"session {self.session}: {len(todo)} runs left in stage {self.stage}")
        self.print_status()
        t_start, n = time.time(), 0
        waiting = defaultdict(int)
        for i, r in enumerate(todo):
            rid = r["run_id"]
            if self.is_done(rid) or not self._lease_ok(rid):
                continue
            err = self.store.get_json(f"runs/{rid}/error.json")
            if err and err["attempts"] >= MAX_ATTEMPTS:
                continue
            cfg = None
            if r["kind"] == "train" and r["arm"] in ("B2", "B3"):
                cfg = self.best_config(r["arm"])
                if cfg is None:
                    waiting[r["arm"]] += 1
                    continue
            self._take_lease(rid)
            self.log(f"[{i + 1}/{len(todo)}] {rid}")
            t0 = time.time()
            self.current = {"run_id": rid, "started": t0, "msg": None}
            try:
                self._one(r, cfg)
            except Exception as e:  # recorded, the loop goes on
                if "out of memory" in str(e).lower():
                    import torch
                    torch.cuda.empty_cache()
                attempts = (err or {}).get("attempts", 0) + 1
                self.store.put_json(f"runs/{rid}/error.json", {"attempts": attempts, "session": self.session,
                                                               "time": now_utc(), "error": traceback.format_exc()})
                self.store.delete(f"runs/{rid}/lease.json")
                self.log(f"  FAILED (attempt {attempts}/{MAX_ATTEMPTS}): {type(e).__name__}: {e}")
                self.current = None
                self.print_status()
                continue
            n += 1
            self.session_info["runs"] += 1
            self.progress.add_wall(rid, time.time() - t0)
            self.current = None
            el = time.time() - t_start
            self.log(f"  done in {time.time() - t0:.0f}s | this session {n} runs, {el / 3600:.2f} h")
            self.print_status()
            if limit and n >= limit:
                break
        for arm, k in waiting.items():
            self.log(f"{k} run {arm} dilewati: grid {arm} belum lengkap (lihat error.json grid)")
        self.log("no runnable run left in this stage" if not limit or n < limit else f"stopped after {n} runs")
        return n

    def _ctx(self, rid):
        ctx = dict(self.ctx_base)
        ctx["ckpt"] = Checkpoint(self.store, rid, self.work, self.ckpt_every_s,
                                 on_save=lambda: self.log("  checkpoint saved"))
        last = {"t": 0.0}

        def heartbeat(msg):
            self.log(f"  {msg}")
            if self.current:
                self.current["msg"] = msg
            if time.time() - last["t"] > 60:
                self._take_lease(rid)
                last["t"] = time.time()
            if time.time() - self._last_status >= self.status_every_s:
                self.print_status()
        ctx["heartbeat"] = heartbeat
        return ctx

    def _one(self, r, cfg):
        rid = r["run_id"]
        t_start = time.time()
        ctx = self._ctx(rid)
        data = run_data(r, self.plan)
        repo, rev = MODELS[r["arm"]]
        if r["arm"] in ("B2", "B3"):
            lr, bs = (r["lr"], r["bs"]) if r["kind"] == "grid" else (cfg["lr"], cfg["bs"])
            res = hf_clf.train(r, data, lr, bs, ctx)
            conf = {"lr": lr, "bs": bs, "optimizer": "Adam", "max_len": hf_clf.MAX_LEN, **(cfg or {})}
        elif r["kind"] == "zeroshot":
            res = laya_ft.zeroshot(r, data, ctx)
            conf = {"temperature": res["temperature"]}
        else:
            res = laya_ft.train(r, data, ctx)
            conf = {k: getattr(laya_ft, k) for k in ("EPOCHS", "MICRO_BATCH", "GRAD_ACCUM", "GROUP_SIZE",
                                                     "LR_ENCODER", "LR_HEAD", "SIGMA_START", "SIGMA_END")}
            conf.update(temperature_fitted=res["temperature"], temperature_applied=res["temperature_applied"])
        model_id = f"{repo}@{rev[:12]}"
        version = f"ft:{rid}" if r["kind"] != "zeroshot" else model_id
        meta = {"run": r, "config": conf, "model_id": model_id, "valid_f1": res.get("valid_f1"),
                "best_epoch": res.get("best_epoch"), "epochs": res.get("epochs"), "history": res.get("history"),
                "train_n": res["train_n"], "train_time_s": res["train_time_s"], "gpu": gpu_name(),
                "versions": env_versions(), "plan_sha": self.plan_sha, "code_sha": self.code_sha,
                "session": self.session, "finished_utc": now_utc()}
        if r["kind"] != "grid":
            train_df, valid_df, test_df = data
            test = pred_frame(r, "test", test_df, res["test_probs"], model_id, version, res["train_n"],
                              res["train_time_s"])
            self.store.put_bytes(f"runs/{rid}/pred_test.parquet", _parquet_bytes(test))
            if res.get("valid_probs") is not None:
                valid = pred_frame(r, "valid", valid_df, res["valid_probs"], model_id, version, res["train_n"],
                                   res["train_time_s"])
                self.store.put_bytes(f"runs/{rid}/pred_valid.parquet", _parquet_bytes(valid))
                if meta["valid_f1"] is None:
                    from sklearn.metrics import f1_score
                    meta["valid_f1"] = float(f1_score(valid["gold"], valid["pred"], average="macro",
                                                      labels=list(DATASETS[r["dataset"]]["labels"])))
        if r.get("export"):
            self._export(r, res)
        meta["wall_s"] = round(time.time() - t_start, 1)
        self.store.put_json(f"runs/{rid}/meta.json", meta)
        self.store.put_json(f"runs/{rid}/done.json", {"valid_f1": meta["valid_f1"], "finished_utc": now_utc(),
                                                      "session": self.session})
        ctx["ckpt"].clear()
        for k in ("lease.json", "error.json"):
            if self.store.exists(f"runs/{rid}/{k}"):
                self.store.delete(f"runs/{rid}/{k}")
        self.log(f"  valid macro-F1 {meta['valid_f1']:.4f}" if meta["valid_f1"] is not None else "  (no valid)")

    def _export(self, r, res):
        out = self.work / "export" / r["run_id"]
        (laya_ft if r["arm"] == "C1" else hf_clf).export(res, str(out))
        tar = out.with_suffix(".tar")
        with tarfile.open(tar, "w") as t:
            t.add(out, arcname=r["run_id"])
        self.store.put_file(f"export/{r['run_id']}.tar", tar)
        tar.unlink()
        self.log(f"  exported checkpoint -> export/{r['run_id']}.tar")

    # ---------------------------------------------------------------- collect
    def collect(self, out_dir, include_export=False):
        """Copy every finished run's predictions + meta into out_dir (layout mirrors jev/runs) and zip it."""
        out = Path(out_dir)
        n = 0
        keys = self.store.list("runs/")
        done = {k.split("/")[1] for k in keys if k.endswith("/done.json")}
        for key in keys:
            rid, name = key.split("/")[1], key.split("/")[-1]
            if name not in ("pred_test.parquet", "pred_valid.parquet", "meta.json", "done.json") \
                    or "/ckpt/" in key:
                continue
            if rid not in done:
                continue
            arm = rid.split("-")[0].upper() if not rid.startswith("grid") else "grid"
            dest = {"pred_test.parquet": out / "runs" / arm / f"{rid}.parquet",
                    "pred_valid.parquet": out / "runs" / arm / "valid" / f"{rid}.parquet",
                    "meta.json": out / "meta" / f"{rid}.json",
                    "done.json": out / "done" / f"{rid}.json"}[name]
            n += name == "done.json"
            if dest.exists():  # outputs of a finished run never change: incremental collect
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(self.store.get_bytes(key))
        if include_export:
            for key in self.store.list("export/"):
                self.store.get_file(key, out / key)
        st = self.status(stages=[1, 2, 3])
        st.to_csv(out / "status.csv", index=False)
        (out / "active_plan.json").write_text(json.dumps(self.store.get_json("plan/active.json"), indent=1))
        zip_path = _zip(out)
        self.log(f"collected {n} finished runs -> {zip_path}")
        return zip_path


def _zip(out):
    import zipfile
    path = out / "f5_results.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob("*")):
            if f.is_file() and f != path and f.suffix != ".tar":
                z.write(f, f.relative_to(out))
    return path
