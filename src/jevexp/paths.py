from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
MANIFEST = DATA / "MANIFEST.json"
SPLITS = ROOT / "splits"
SUBSETS = SPLITS / "subsets.json"
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
