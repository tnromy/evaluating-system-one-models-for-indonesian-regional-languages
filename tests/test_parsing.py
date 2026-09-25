import json

import pytest

from jevexp.data import DATASETS
from jevexp.eval import INVALID
from jevexp.parsing import jev_answer, llm_json, logprob_label_dist
from jevexp.questions import FORMULATIONS, spec, to_jev, to_llm_messages

SENTI = ["negative", "neutral", "positive"]


@pytest.mark.parametrize("dataset", list(DATASETS))
@pytest.mark.parametrize("qlang", ["en", "id"])
def test_spec_keys_are_canonical_labels(dataset, qlang):
    assert tuple(spec(dataset, qlang)["criteria"]) == DATASETS[dataset]["labels"]


def test_render_jev_and_llm_are_deterministic_and_share_text():
    a, b = to_jev("nusax_senti", "choice_en", "x"), to_jev("nusax_senti", "choice_en", "x")
    assert a == b and list(a["questions"]["label"]["criteria"]) == SENTI
    noul = to_jev("nusax_senti", "noul_en", "x")["questions"]
    assert set(noul) == {"is_negative", "is_neutral", "is_positive"}
    msgs = to_llm_messages("nusax_senti", "x")
    assert spec("nusax_senti")["instructions"] in msgs[1]["content"]
    assert "Jev" not in json.dumps(msgs)


def test_jev_choice_reorders_probs_to_canonical():
    resp = {"answers": {"label": {"type": "choice", "choice": "positive", "confidence": 0.7,
                                  "probabilities": {"positive": 0.78, "negative": 0.02, "neutral": 0.2}}}}
    pred, probs, conf = jev_answer(resp, "nusax_senti", "choice_en")
    assert pred == "positive" and probs == pytest.approx([0.02, 0.2, 0.78]) and conf == pytest.approx(0.78)


def test_jev_noul_normalised_and_invalid_when_missing():
    resp = {"answers": {"is_negative": {"noul": 0.1}, "is_neutral": {"noul": 0.3}, "is_positive": {"noul": 0.6}}}
    pred, probs, conf = jev_answer(resp, "nusax_senti", "noul_en")
    assert pred == "positive" and sum(probs) == pytest.approx(1) and conf == pytest.approx(0.6)
    del resp["answers"]["is_neutral"]
    assert jev_answer(resp, "nusax_senti", "noul_en")[0] == INVALID
    assert jev_answer({"error": "x"}, "nusax_senti", "choice_en")[0] == INVALID


@pytest.mark.parametrize("content,expected", [
    ('{"choice": "neutral", "probabilities": {"negative": 0.1, "neutral": 0.6, "positive": 0.3}}', "neutral"),
    ('{"choice": "Neutral"}', INVALID),
    ('not json', INVALID),
    ('```json\n{"choice": "positive"}\n```', "positive"),
    ('```\n{"choice": "negative"}\n```', "negative"),
    ('Here: ```json {"choice": "positive"}```', INVALID),
    ('["neutral"]', INVALID),
])
def test_llm_json(content, expected):
    assert llm_json(content, SENTI)[0] == expected


def _tok(token, tops):
    return {"token": token, "logprob": 0.0, "top_logprobs": [{"token": t, "logprob": lp} for t, lp in tops]}


def test_logprob_merges_whitespace_variants_by_max():
    # regression for the smoke-test bug: " positive" (-9999) must not overwrite "positive" (0.0)
    choice = {"logprobs": {"content": [
        _tok('{"', [('{"', 0.0)]), _tok("choice", [("choice", 0.0)]), _tok(' "', [(' "', 0.0)]),
        _tok("positive", [("positive", 0.0), ("negative", -9999.0), (" positive", -9999.0), ("neutral", -9999.0)]),
    ]}}
    assert logprob_label_dist(choice, SENTI) == pytest.approx([0.0, 0.0, 1.0])


def test_logprob_real_distribution_and_prefix_tokens():
    import math
    choice = {"logprobs": {"content": [
        _tok("pos", [("pos", math.log(0.6)), ("neg", math.log(0.3)), ("neut", math.log(0.1)), ("A", -1.0)]),
    ]}}
    assert logprob_label_dist(choice, SENTI) == pytest.approx([0.3, 0.1, 0.6])
    assert logprob_label_dist({"logprobs": {"content": [_tok("{", [("{", 0.0)])]}}, SENTI) is None


def test_codex_output_schema_and_event_parser():
    from jevexp.questions import output_schema
    from jevexp.runners.codex import parse_events
    s = output_schema("nt_emot")
    assert s["properties"]["choice"]["enum"] == ["anger", "fear", "happy", "love", "sadness"]
    stdout = "\n".join([
        json.dumps({"type": "thread.started", "thread_id": "x"}),
        json.dumps({"type": "item.completed", "item": {"type": "error", "message": "warn"}}),
        json.dumps({"type": "item.completed", "item": {"type": "command_execution", "command": "ls"}}),
        json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": '{"choice": "fear"}'}}),
        json.dumps({"type": "turn.completed", "usage": {"input_tokens": 7000, "output_tokens": 30}}),
    ])
    msg, usage, tools, errors = parse_events(stdout)
    assert msg == '{"choice": "fear"}' and usage["input_tokens"] == 7000 and tools == 1 and errors == ["warn"]
