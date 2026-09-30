"""Design facts for the manuscript (journal 010, F7) that no result CSV carries.

Dataset sizes per split/language, subset sizes, collection volumes and API costs (F4),
fine-tuning volume and GPU hours (F5), B2/B3 grid winners, observed model versions.
The Node.js manuscript writer reads this file instead of hand-typed numbers (K74).
Output -> results/manuscript_facts.json
"""
import glob
import json

import pandas as pd

from jevexp.data import DATASETS, SUBSET_SIZES, load
from jevexp.paths import RESULTS


def dataset_sizes():
    out = {}
    for d, spec in DATASETS.items():
        per = {}
        for split in ("train", "valid", "test"):
            df = load(d, split, langs=spec["parallel_langs"])
            per[split] = {"total": int(len(df)), "per_lang": int(df.groupby("lang").size().median())}
        out[d] = {"langs": list(spec["parallel_langs"]), "n_langs": len(spec["parallel_langs"]),
                  "labels": list(spec["labels"]), "splits": per, "subset_per_lang": SUBSET_SIZES[d],
                  "subset_total": SUBSET_SIZES[d] * len(spec["parallel_langs"])}
    return out


def f4():
    st = pd.read_csv(RESULTS / "f4_collection" / "status.csv")
    by = st.groupby("arm").agg(runs=("run", "count"), items=("rows", "sum"), cost_usd=("cost_usd", "sum"),
                               invalid_max_pct=("invalid_pct", "max"))
    versions = sorted(set(v for vs in st["versions"].dropna() for v in str(vs).split(";")))
    return {"runs": int(len(st)), "items": int(st["rows"].sum()), "by_arm": by.round(4).reset_index().to_dict("records"),
            "versions": versions}


def f5():
    meta = [json.load(open(f)) for f in glob.glob(str(RESULTS / "f5_collect" / "meta" / "*.json"))]
    status = pd.read_csv(RESULTS / "f5_collect" / "status.csv")
    grid = pd.DataFrame([{**m["run"], "valid_f1": m["valid_f1"]} for m in meta if m["run"]["kind"] == "grid"])
    winners = {arm: g.sort_values("valid_f1", ascending=False).iloc[0][["lr", "bs", "valid_f1"]].to_dict()
               for arm, g in grid.groupby("arm")}
    return {"runs": int(status["total"].sum()), "done": int(status["done"].sum()), "failed": int(status["failed"].sum()),
            "gpu_hours": round(sum(m.get("wall_s") or 0 for m in meta) / 3600, 1),
            "sessions": len({m["session"] for m in meta}), "gpu": sorted({m["gpu"] for m in meta}),
            "grid_size": {arm: int(len(g)) for arm, g in grid.groupby("arm")},
            "grid_lrs": sorted(grid["lr"].unique().tolist()), "grid_bs": sorted(grid["bs"].unique().tolist()),
            "grid_winners": winners}


def main():
    facts = {"datasets": dataset_sizes(), "f4": f4(), "f5": f5()}
    (RESULTS / "manuscript_facts.json").write_text(json.dumps(facts, indent=1, default=float))
    print(json.dumps(facts, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
