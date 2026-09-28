"""Laptop side of F5: bring finished runs from MinIO (or a downloaded f5_results.zip) into jev/runs.

CLI:  python -m jevexp.ft.pull --config /path/to/config.json [--export]
      python -m jevexp.ft.pull --zip f5_results.zip
Prints only completeness (no test metric: analysis_plan §7).
"""
import argparse
import shutil
import zipfile

import pandas as pd

from ..paths import RESULTS, RUNS
from ..schema import read
from .store import s3_store
from .worker import Worker

OUT = RESULTS / "f5_collect"


def install(out):
    n = 0
    for f in sorted((out / "runs").rglob("*.parquet")):
        read(f)  # schema check
        dest = RUNS / f.relative_to(out / "runs")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, dest)
        n += 1
    for f in (out / "export").glob("*.tar") if (out / "export").exists() else []:
        dest = RUNS.parent / "models" / "f5" / f.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, dest)
    return n


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--config")
    g.add_argument("--zip")
    ap.add_argument("--export", action="store_true", help="also fetch export/*.tar (large)")
    a = ap.parse_args()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    if a.config:
        store, where = s3_store(a.config)
        print("MinIO:", where)
        Worker(store, work=OUT / "work", log=print).collect(OUT, include_export=a.export)
    else:
        zipfile.ZipFile(a.zip).extractall(OUT)
    print(f"{install(OUT)} prediction files copied into {RUNS}")
    print(pd.read_csv(OUT / "status.csv").to_string(index=False))


if __name__ == "__main__":
    main()
