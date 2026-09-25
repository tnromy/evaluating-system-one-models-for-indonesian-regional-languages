"""F4 collection monitor. Deliberately computes NO performance metric (no accuracy / F1 / ECE) on test:
analysis_plan.md §7 forbids interim looks before every MVP arm (incl. F5) is collected.

CLI:  python -m jevexp.report_f4            # table of all test runs
      python -m jevexp.report_f4 --check X  # exit 1 if run X has INVALID > 1 % or > 1 Jev snapshot
"""
import argparse
import sys

import pandas as pd

from .eval import INVALID
from .paths import RESULTS, RUNS
from .schema import read

JEV_SNAPSHOT = "typesafe/jev-1.13-20260917"


def status(path):
    df = read(path)
    s = {"run": path.stem, "arm": df["arm"].iloc[0], "rows": len(df),
         "invalid_pct": round(100 * (df["pred"] == INVALID).mean(), 3),
         "versions": ";".join(sorted(df["model_version"].dropna().astype(str).unique()))[:80],
         "latency_p50_ms": round(df["latency_ms"].median(), 0),
         "latency_p95_ms": round(df["latency_ms"].quantile(0.95), 0),
         "cost_usd": round(df["cost_usd"].fillna(0).sum(), 4) if "cost_usd" in df else None,
         "no_logprob": int((df["confidence_type"] == "none").sum())}
    return s


def check(path):
    s = status(path)
    problems = []
    if s["invalid_pct"] > 1:
        problems.append(f"INVALID {s['invalid_pct']} % > 1 %")
    if s["arm"].startswith("A") and s["versions"] != JEV_SNAPSHOT:
        problems.append(f"Jev snapshot(s) {s['versions']} != {JEV_SNAPSHOT}")
    print(f"{s['run']}: rows {s['rows']}, INVALID {s['invalid_pct']} %, versions {s['versions']}, "
          f"cost ${s['cost_usd']}" + (f"  PROBLEM: {problems}" if problems else "  OK"))
    return not problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check")
    a = ap.parse_args()
    if a.check:
        sys.exit(0 if check(next(RUNS.glob(f"*/{a.check}.parquet"))) else 1)
    rows = [status(p) for p in sorted(RUNS.glob("*/test-*.parquet"))]
    df = pd.DataFrame(rows)
    out = RESULTS / "f4_collection"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "status.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
