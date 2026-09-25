"""Single question spec shared by Jev and the LLM arms (docs/metodologi.md §4.2, K35).

A spec = instructions + criteria {canonical label: description}, per dataset and question language.
Option keys are always the canonical labels, identical for every arm (option names alone shift answers, j63).
"""
import json

from .data import DATASETS

_SPECS = {
    ("nusax_senti", "en"): {
        "instructions": "What is the sentiment of the text?",
        "criteria": {
            "negative": "The text expresses a negative opinion or feeling.",
            "neutral": "The text is factual or expresses no clear positive or negative opinion.",
            "positive": "The text expresses a positive opinion or feeling.",
        },
    },
    ("nusax_senti", "id"): {
        "instructions": "Apa sentimen dari teks ini?",
        "criteria": {
            "negative": "Teks mengungkapkan pendapat atau perasaan negatif.",
            "neutral": "Teks bersifat faktual atau tidak mengungkapkan pendapat positif maupun negatif yang jelas.",
            "positive": "Teks mengungkapkan pendapat atau perasaan positif.",
        },
    },
    ("nt_senti", "en"): {
        "instructions": "What is the sentiment of the text?",
        "criteria": {
            "negative": "The text expresses a negative opinion or feeling.",
            "positive": "The text expresses a positive opinion or feeling.",
        },
    },
    ("nt_senti", "id"): {
        "instructions": "Apa sentimen dari teks ini?",
        "criteria": {
            "negative": "Teks mengungkapkan pendapat atau perasaan negatif.",
            "positive": "Teks mengungkapkan pendapat atau perasaan positif.",
        },
    },
    ("nt_emot", "en"): {
        "instructions": "Which emotion does the text mainly express?",
        "criteria": {
            "anger": "Anger, annoyance or irritation.",
            "fear": "Fear, worry or anxiety.",
            "happy": "Happiness, joy or excitement.",
            "love": "Love, affection or caring.",
            "sadness": "Sadness, disappointment or grief.",
        },
    },
    ("nt_emot", "id"): {
        "instructions": "Emosi apa yang terutama diungkapkan oleh teks ini?",
        "criteria": {
            "anger": "Marah, kesal atau jengkel.",
            "fear": "Takut, khawatir atau cemas.",
            "happy": "Bahagia, gembira atau senang.",
            "love": "Cinta, sayang atau peduli.",
            "sadness": "Sedih, kecewa atau berduka.",
        },
    },
}

FORMULATIONS = {"choice_en": ("choice", "en"), "choice_id": ("choice", "id"), "noul_en": ("noul", "en")}


def spec(dataset, qlang="en"):
    s = _SPECS[(dataset, qlang)]
    assert tuple(s["criteria"]) == DATASETS[dataset]["labels"], "criteria keys must be the canonical labels"
    return s


def to_jev(dataset, formulation, text):
    """Request body fields `state` + `questions` for the Jev System One API."""
    kind, qlang = FORMULATIONS[formulation]
    s = spec(dataset, qlang)
    if kind == "choice":
        questions = {"label": {"type": "choice", "instructions": s["instructions"], "criteria": dict(s["criteria"])}}
    else:  # one Noul per label; answers are normalised into a distribution by parsing.jev_answer
        questions = {f"is_{lab}": {"type": "noul", "instructions": f"{s['instructions']} Is it '{lab}'? {desc}"}
                     for lab, desc in s["criteria"].items()}
    return {"state": {"text": text}, "questions": questions}


def to_llm_messages(dataset, text, qlang="en"):
    """Neutral chat prompt (no persona, no mention of Jev) from the same spec; JSON answer."""
    s = spec(dataset, qlang)
    labels = list(s["criteria"])
    schema = {"choice": "<one of: " + ", ".join(labels) + ">",
              "probabilities": {lab: "<number between 0 and 1>" for lab in labels}}
    system = ("You are a text classifier. Answer ONLY with a JSON object of the form "
              f"{json.dumps(schema)}. The choice must be exactly one of the options and the probabilities "
              "must sum to 1.")
    options = "\n".join(f"- {lab}: {desc}" for lab, desc in s["criteria"].items())
    user = f"{s['instructions']}\nOptions:\n{options}\n\nText: {text}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
