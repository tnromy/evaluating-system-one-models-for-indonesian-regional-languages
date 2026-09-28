"""F5 progress + ETA from the store (K63). Used by the Colab worker (printed to the console, copied to
`status/progress.txt`) and by the laptop:  python -m jevexp.ft.progress --config dist/f5/config.json

ETA per remaining run = mean wall time of finished runs in the most specific group that has any
(kind, arm, dataset, regime, n) -> (kind, arm, regime, n) -> (kind, arm, dataset, regime) -> (kind, arm, regime);
otherwise a prior guess (PRIOR_S),
marked with '*'. No test metric is ever shown (analysis_plan §7); only validation F1 lives in meta.json.
"""
import argparse
import time
from collections import defaultdict
from datetime import datetime, timezone

MAX_ATTEMPTS = 2

# Prior wall-time guesses on a T4 (seconds), used only until a group has a finished run.
PRIOR_S = {
    ("grid", "B2"): 180, ("grid", "B3"): 240,
    ("train", "B2", "nusax_senti", "per_lang"): 150, ("train", "B3", "nusax_senti", "per_lang"): 200,
    ("train", "B2", "nusax_senti", "pooled"): 1200, ("train", "B3", "nusax_senti", "pooled"): 1600,
    ("train", "B2", "nt_senti", "per_lang"): 600, ("train", "B3", "nt_senti", "per_lang"): 800,
    ("train", "B2", "nt_emot", "per_lang"): 550, ("train", "B3", "nt_emot", "per_lang"): 750,
    ("train", "C1", "nusax_senti", "per_lang"): 150, ("train", "C1", "nusax_senti", "pooled"): 650,
    ("train", "C1", "nt_senti", "pooled"): 2700, ("train", "C1", "nt_emot", "pooled"): 2400,
    ("zeroshot", "C2"): 240,
    ("curve", "B2"): 90, ("curve", "C1"): 120,
}


def regime(r):
    if r["kind"] in ("grid", "zeroshot"):
        return r["kind"]
    if r["n"]:
        return "curve"
    return "pooled" if r["train_lang"] == "pooled" else "per_lang"


def wall_of(meta):
    """Wall seconds of a finished run; older metas (before K63) only have train_time_s."""
    if meta.get("wall_s"):
        return float(meta["wall_s"])
    return float(meta.get("train_time_s") or 0) * 1.15 + 60


class Progress:
    def __init__(self, store, plan):
        self.store, self.plan = store, plan
        self._walls = {}  # run_id -> wall seconds (finished metas never change)

    def snapshot(self):
        keys = self.store.list("runs/")
        done = {k.split("/")[1] for k in keys if k.endswith("/done.json")}
        errs = {k.split("/")[1] for k in keys if k.endswith("/error.json")}
        dead = set()
        for rid in errs - done:
            e = self.store.get_json(f"runs/{rid}/error.json") or {}
            if e.get("attempts", 0) >= MAX_ATTEMPTS:
                dead.add(rid)
        for rid in done - set(self._walls):
            meta = self.store.get_json(f"runs/{rid}/meta.json")
            if meta:
                self._walls[rid] = wall_of(meta)
        return done, errs - done - dead, dead

    def add_wall(self, rid, wall_s):
        self._walls[rid] = wall_s

    def _estimator(self, runs):
        by = defaultdict(list)
        for r in runs:
            w = self._walls.get(r["run_id"])
            if w is None:
                continue
            reg = regime(r)
            for k in _keys(r, reg):
                by[k].append(w)

        def est(r):
            reg = regime(r)
            for k in _keys(r, reg):
                if by.get(k):
                    return sum(by[k]) / len(by[k]), False
            for k in ((r["kind"], r["arm"], r["dataset"], reg), (reg, r["arm"]), (r["kind"], r["arm"])):
                if k in PRIOR_S:
                    return PRIOR_S[k], True
            return 300, True
        return est

    def report(self, stage=None, current=None, session=None):
        """(text, dict). `current` = {run_id, started, msg}; `session` = {id, started, runs}."""
        runs = self.plan["runs"]
        done, pending_err, dead = self.snapshot()
        est = self._estimator(runs)
        now = time.time()
        stages, lines, total_left, any_prior = {}, [], 0.0, False
        for st in (1, 2, 3):
            rs = [r for r in runs if r["stage"] == st]
            left = [r for r in rs if r["run_id"] not in done and r["run_id"] not in dead]
            secs, prior = 0.0, False
            for r in left:
                s, p = est(r)
                if current and r["run_id"] == current["run_id"]:
                    s = max(0.0, s - (now - current["started"]))
                secs += s
                prior |= p
            total_left += secs
            any_prior |= prior
            groups = defaultdict(lambda: [0, 0, 0])
            for r in rs:
                g = groups[(r["kind"], r["arm"], r["dataset"], regime(r))]
                g[0] += r["run_id"] in done
                g[1] += 1
                g[2] += r["run_id"] in dead
            stages[st] = {"done": sum(r["run_id"] in done for r in rs), "total": len(rs),
                          "failed": sum(r["run_id"] in dead for r in rs), "retry": sum(r["run_id"] in pending_err
                                                                                       for r in rs),
                          "eta_h": round(secs / 3600, 2), "prior": prior,
                          "groups": {" ".join(k): v for k, v in groups.items()}}
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        head = f"==== STATUS F5 · {ts}"
        if session:
            head += (f" · sesi {session['id']} ({(now - session['started']) / 3600:.1f} j, "
                     f"{session['runs']} run selesai)")
        lines.append(head + " ====")
        for st, s in stages.items():
            mark = "*" if s["prior"] else ""
            bar = _bar(s["done"], s["total"])
            extra = (f" · {s['failed']} gagal permanen" if s["failed"] else "") + \
                    (f" · {s['retry']} akan dicoba ulang" if s["retry"] else "")
            here = "  <- sekarang" if st == stage else ""
            lines.append(f"Stage {st} {bar} {s['done']:>4}/{s['total']:<4} selesai{extra} · sisa ±{_h(s['eta_h'])}"
                         f"{mark}{here}")
            if st == stage:
                for g, (d, t, f) in sorted(s["groups"].items(), key=lambda kv: kv[0]):
                    lines.append(f"    {g:<36} {d:>4}/{t:<4}" + (f"  ({f} gagal)" if f else ""))
        if current:
            lines.append(f"Sedang: {current['run_id']} — {current.get('msg') or 'mulai'} "
                         f"(berjalan {(now - current['started']) / 60:.0f} mnt)")
        lines.append(f"Total sisa semua stage: ±{_h(total_left / 3600)}"
                     + ("  (* = sebagian masih perkiraan awal; makin akurat setelah run di kelompok itu selesai)"
                        if any_prior else ""))
        data = {"time_utc": ts, "stage": stage, "current": current, "session": session,
                "eta_total_h": round(total_left / 3600, 2), "stages": stages}
        return "\n".join(lines), data


def _keys(r, reg):
    """Empirical groups, most specific first. The regime is always kept: a pooled run says nothing about the
    duration of a 10-example curve run."""
    return ((r["kind"], r["arm"], r["dataset"], reg, r["n"]), (r["kind"], r["arm"], reg, r["n"]),
            (r["kind"], r["arm"], r["dataset"], reg), (r["kind"], r["arm"], reg))


def _bar(d, t, width=20):
    k = int(round(width * d / t)) if t else 0
    return "[" + "#" * k + "." * (width - k) + "]"


def _h(hours):
    if hours < 1:
        return f"{hours * 60:.0f} mnt"
    return f"{hours:.1f} j"


def main():
    from .plan import load_plan
    from .store import s3_store
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    store, where = s3_store(a.config)
    print("MinIO:", where)
    live = store.get_json("status/progress.json")
    if live:
        print(f"laporan terakhir dari Colab: {live['time_utc']}")
        print(store.get_bytes("status/progress.txt").decode())
        print()
    plan, _ = load_plan()
    stage = live.get("stage") if live else None
    text, _ = Progress(store, plan).report(stage=stage, current=(live or {}).get("current"),
                                           session=(live or {}).get("session"))
    print("dihitung sekarang dari MinIO:")
    print(text)


if __name__ == "__main__":
    main()
