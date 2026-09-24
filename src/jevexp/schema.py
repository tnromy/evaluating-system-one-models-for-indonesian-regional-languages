"""Per-item prediction schema shared by every arm (docs/metodologi.md §6).

One row = one item x one condition. Stored as Parquet under runs/<arm>/<run_id>.parquet.
"""
import numpy as np
import pandas as pd

from .data import DATASETS
from .eval import INVALID

REQUIRED = [
    "run_id", "arm", "condition",
    "dataset", "split", "lang", "sentence_id", "gold",
    "pred", "confidence_type", "model_id", "timestamp_utc",
]
OPTIONAL = [
    "probs", "confidence", "model_version",
    "latency_ms", "input_tokens", "output_tokens", "cost_usd",
    "request_hash", "raw_response_path", "client_location",
    "train_n", "seed", "train_time_s", "gpu",
]
CONFIDENCE_TYPES = {"model_prob", "logprob", "verbalized", "none"}
PROB_TOL = 1e-6


def validate(df):
    """Raise ValueError on any schema violation; return df unchanged otherwise."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    unknown = set(df.columns) - set(REQUIRED) - set(OPTIONAL)
    if unknown:
        raise ValueError(f"unknown columns: {sorted(unknown)}")
    if df[REQUIRED].isna().any().any():
        raise ValueError("null values in required columns")
    bad_ct = set(df["confidence_type"]) - CONFIDENCE_TYPES
    if bad_ct:
        raise ValueError(f"bad confidence_type: {sorted(bad_ct)}")
    for dataset, g in df.groupby("dataset"):
        labels = set(DATASETS[dataset]["labels"])
        if not set(g["gold"]) <= labels:
            raise ValueError(f"{dataset}: gold outside {sorted(labels)}")
        if not set(g["pred"]) <= labels | {INVALID}:
            raise ValueError(f"{dataset}: pred outside {sorted(labels)} + INVALID")
        if "probs" in g.columns:
            rows = [p for p in g["probs"] if p is not None]
            if rows:
                arr = np.asarray([list(p) for p in rows], dtype=float)
                if arr.shape[1] != len(labels):
                    raise ValueError(f"{dataset}: probs must have {len(labels)} columns (label order of DATASETS)")
                if np.any(arr < -PROB_TOL) or np.any(np.abs(arr.sum(axis=1) - 1) > PROB_TOL):
                    raise ValueError(f"{dataset}: probs rows must be non-negative and sum to 1")
    key = ["run_id", "condition", "dataset", "split", "lang", "sentence_id"]
    if df.duplicated(key).any():
        raise ValueError("duplicate item rows for the same run/condition")
    return df


def write(df, path):
    validate(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read(path):
    return validate(pd.read_parquet(path))
