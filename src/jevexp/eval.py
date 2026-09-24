"""Metrics shared by every arm (docs/metodologi.md §7-8).

Conventions:
  y_true, y_pred : arrays of label strings; y_pred may contain INVALID (always counted wrong).
  probs          : (n, k) array aligned with `labels`; rows sum to 1.
  confidence     : (n,) array; for selective prediction, higher = more confident.
"""
import numpy as np
from sklearn.metrics import f1_score

INVALID = "INVALID"


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def macro_f1(y_true, y_pred, labels):
    """Macro-F1 over the task's labels only; INVALID predictions count as errors for every class."""
    return float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0))


def _top(probs):
    probs = np.asarray(probs, dtype=float)
    return probs.max(axis=1), probs.argmax(axis=1)


def _bin_edges(conf, n_bins, scheme):
    if scheme == "width":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if scheme == "mass":
        edges = np.quantile(conf, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        return edges
    raise ValueError(scheme)


def reliability_bins(probs, y_idx, n_bins=15, scheme="width"):
    """Per bin: (count, mean confidence, accuracy). y_idx = gold class index into probs columns."""
    conf, pred = _top(probs)
    correct = pred == np.asarray(y_idx)
    edges = _bin_edges(conf, n_bins, scheme)
    # right-closed bins, first bin also includes 0
    which = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = which == b
        if m.any():
            rows.append((int(m.sum()), float(conf[m].mean()), float(correct[m].mean())))
    return rows


def ece(probs, y_idx, n_bins=15, scheme="width"):
    """Expected calibration error of the top-class confidence (Naeini 2015; adaptive = Nixon 2019)."""
    n = len(y_idx)
    return float(sum(c / n * abs(acc - cf) for c, cf, acc in reliability_bins(probs, y_idx, n_bins, scheme)))


def brier(probs, y_idx):
    """Multiclass Brier score: mean over items of sum_k (p_k - 1[y=k])^2."""
    probs = np.asarray(probs, dtype=float)
    onehot = np.eye(probs.shape[1])[np.asarray(y_idx)]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def coverage_accuracy_curve(confidence, correct):
    """Sort by confidence (desc); return coverage and accuracy of the kept prefix at every cut."""
    order = np.argsort(-np.asarray(confidence, dtype=float), kind="stable")
    c = np.asarray(correct, dtype=float)[order]
    k = np.arange(1, len(c) + 1)
    return k / len(c), np.cumsum(c) / k


def aurc(confidence, correct):
    """Area under the risk-coverage curve (Geifman et al. 2019): mean selective risk over coverages."""
    _, acc = coverage_accuracy_curve(confidence, correct)
    return float(np.mean(1.0 - acc))


def cluster_bootstrap(df, metric_fn, cluster="sentence_id", n_boot=2000, seed=0, alpha=0.05):
    """Percentile CI resampling whole clusters (all languages of a sentence move together).

    metric_fn(sub_df) -> float. Returns (point, lo, hi).
    """
    rng = np.random.default_rng(seed)
    groups = {k: g.index.to_numpy() for k, g in df.groupby(cluster, sort=True)}
    keys = np.array(list(groups))
    stats = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(keys, size=len(keys), replace=True)
        stats[b] = metric_fn(df.loc[np.concatenate([groups[k] for k in pick])])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(metric_fn(df)), float(lo), float(hi)
