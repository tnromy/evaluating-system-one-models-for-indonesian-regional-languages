"""Turn raw API responses into (pred, probs, confidence). Probs follow DATASETS label order."""
import json
import math

from .data import DATASETS
from .eval import INVALID
from .questions import FORMULATIONS


def _normalise(values, labels):
    z = sum(values.get(lab, 0.0) for lab in labels)
    if z <= 0:
        return None
    return [values.get(lab, 0.0) / z for lab in labels]


def _pred(probs, labels):
    return labels[max(range(len(labels)), key=lambda i: probs[i])]


def jev_answer(resp, dataset, formulation):
    """Jev System One response -> (pred, probs, confidence). INVALID if the answer is missing."""
    labels = list(DATASETS[dataset]["labels"])
    answers = (resp or {}).get("answers") or {}
    kind = FORMULATIONS[formulation][0]
    if kind == "choice":
        a = answers.get("label") or {}
        probs = _normalise({k: float(v) for k, v in (a.get("probabilities") or {}).items()}, labels)
        if a.get("choice") not in labels or probs is None:
            return INVALID, None, None
        # the reported choice is the prediction; confidence = its probability
        return a["choice"], probs, probs[labels.index(a["choice"])]
    nouls = {lab: (answers.get(f"is_{lab}") or {}).get("noul") for lab in labels}
    if any(v is None for v in nouls.values()):
        return INVALID, None, None
    probs = _normalise({k: float(v) for k, v in nouls.items()}, labels)
    if probs is None:  # every noul exactly 0 -> uniform
        probs = [1 / len(labels)] * len(labels)
    pred = _pred(probs, labels)
    return pred, probs, max(probs)


def llm_json(content, labels):
    """LLM JSON answer -> (pred, verbalized probs or None). INVALID on malformed JSON / unknown label."""
    try:
        obj = json.loads(content)
    except (TypeError, ValueError):
        return INVALID, None
    choice = obj.get("choice") if isinstance(obj, dict) else None
    if choice not in labels:
        return INVALID, None
    raw = obj.get("probabilities")
    probs = None
    if isinstance(raw, dict):
        try:
            probs = _normalise({k: max(float(v), 0.0) for k, v in raw.items() if k in labels}, labels)
        except (TypeError, ValueError):
            probs = None
    return choice, probs


def logprob_label_dist(choice_obj, labels):
    """Distribution over labels from top_logprobs at the first generated token that begins a label.

    Token variants that collapse to the same string ("positive", " positive", "\"positive") are merged by max.
    Label probability = max over candidate tokens that are a prefix of the label. Log-sum-exp normalisation.
    Returns None if no label token is found.
    """
    content = ((choice_obj or {}).get("logprobs") or {}).get("content") or []
    for tok in content:
        head = tok["token"].strip().strip('"')
        if not head or not any(lab.startswith(head) for lab in labels):
            continue
        cands = {}
        for c in tok.get("top_logprobs") or []:
            key = c["token"].strip().strip('"')
            if key:
                cands[key] = max(cands.get(key, -math.inf), c["logprob"])
        hits = {}
        for lab in labels:
            lps = [lp for t, lp in cands.items() if lab.startswith(t)]
            if lps:
                hits[lab] = max(lps)
        if not hits:
            return None
        m = max(hits.values())
        z = sum(math.exp(v - m) for v in hits.values())
        return [math.exp(hits[lab] - m) / z if lab in hits else 0.0 for lab in labels]
    return None
