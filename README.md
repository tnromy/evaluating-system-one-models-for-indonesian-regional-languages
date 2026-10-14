# Evaluating System One Models for Indonesian Regional Languages

Code, pre-registration, predictions and results of a pre-registered study of **sentiment and emotion
classification in Indonesian regional languages**. It compares calibrated typed-decision ("System One") models
with **classical classifiers**, **fine-tuned encoders** and **large language models (LLMs)**.

> Preprint: R. Fadli, Rusdah, W. R. P. Utami and Z. M. Amana, *Does System One Transfer? Calibrated Typed-Decision
> Models, Label Budgets, and Translation for Indonesian Regional Languages*, arXiv:2610.14112, 2026.
> https://arxiv.org/abs/2610.14112 (DOI: https://doi.org/10.48550/arXiv.2610.14112).

The study asks four questions:
1. How does zero-shot accuracy depend on how close a regional language is to Indonesian?
2. Do the model probabilities stay calibrated on regional languages?
3. How many labels does a retrained classifier need to match a zero-shot model?
4. Can a confidence-gated cascade improve the cost–accuracy trade-off?

## Data

| Dataset | Source | Languages (parallel) | Labels | Test items |
|---|---|---|---|---|
| NusaX-senti | [IndoNLP/nusax](https://github.com/IndoNLP/nusax) | 12: ace, ban, bbc, bjn, bug, eng, ind, jav, mad, min, nij, sun | negative, neutral, positive | 4,800 |
| NusaTranslation sentiment (NT-senti) | [IndoNLP/nusa-writes](https://github.com/IndoNLP/nusa-writes) | 7: bew, btk, jav, mad, mak, min, sun | negative, positive | 8,400 |
| NusaTranslation emotion (NT-emot) | [IndoNLP/nusa-writes](https://github.com/IndoNLP/nusa-writes) | 7 (as above) | anger, fear, happy, love, sadness | 7,000 |

Both sources are released under CC BY-SA 4.0. The data are not redistributed here. `python -m jevexp.data fetch`
downloads the pinned commits and checks every file against `data/MANIFEST.json` (SHA-256).

## Systems compared

### System One (typed-decision) models
- **Jev** (TypeSafe AI, model `jev-1.13`, snapshot `typesafe/jev-1.13-20260917`) is queried zero-shot through
  the OpenRouter System One endpoint. It is asked in two formulations: one yes/no ("noul") question per label
  (`noul_en`, arm A1), and the native single-choice question (`choice_en`, arm A1-choice, used for calibration).
  Two further arms vary the input. A4 gives Jev the Indonesian source sentence. A3 first translates the
  regional sentence to Indonesian with NLLB-200 (distilled 600M).
- **Laya-multilingual** (Convai Innovations, open weights, mmBERT encoder) is run in two ways:
  - **zero-shot** (arm C2), with the shipped checkpoint;
  - **fine-tuned** (arm C1), with the official RLCD recipe (policy gradient against a proper scoring rule plus
    cross-entropy) adapted to a single GPU. It is trained per language and pooled, with three seeds.

### Classical baselines (B1, following the NusaX protocol, scikit-learn)
- **Naive Bayes**, **linear SVM** and **logistic regression**, each on **bag-of-words** and **TF-IDF**
  features. Hyperparameters are selected on the validation split, and the model is refitted on
  training + validation data. Logistic regression on bag-of-words (B1-LR) is the primary trained comparator.

### Fine-tuned encoders
- **B2: IndoBERT-base** (`indobenchmark/indobert-base-p1`) and **B3: XLM-R-base**, trained per language and
  pooled, three seeds. Learning rate and batch size are chosen by a 5 × 4 grid on Indonesian NusaX validation data.

### Large language models
- **DeepSeek V4.1 Flash** (D1, no reasoning) runs on the full test sets. Its prediction is the label with the
  highest token log-probability, so it also gives a probability distribution.
- **Claude** (D3) — Haiku 4.5, Sonnet 5, Opus 5.5 — via Claude Code with a JSON schema.
- **OpenAI GPT** (D4) — GPT-6 Luna, GPT-6 Sol, GPT-5.6 Terra (low reasoning effort) — via Codex CLI with a JSON
  output schema.

Claude and GPT form a secondary "LLM ladder". It runs on fixed stratified test subsets (`splits/subsets.json`),
with verbalised probabilities.

All zero-shot systems receive the same task specification (`src/jevexp/questions.py`). It has an English
instruction and one description per label, and the canonical label names serve as option keys.

## Pre-registration

The hypotheses, tests, correction families and analysis code were committed and tagged **`prereg-v1`** before
the test splits were opened (`prereg/`, with `prereg/SHA256SUMS`). Every deviation is documented in the manuscript.
Confirmatory analyses:
- **H1a**: logistic mixed model of correctness on sentence-level overlap with Indonesian (R `lme4`).
- **H1b**: A4 vs A1, exact McNemar with Holm correction.
- **H2a/H2b**: ΔECE and ΔAURC, regional minus Indonesian, with a cluster bootstrap.
- **H3a**: label budget n\* from learning curves, tested with Spearman's ρ.

## Repository layout

| Path | Contents |
|---|---|
| `src/jevexp/` | data loading, metrics (`eval.py`), prediction schema, query runners (`runners/`), fine-tuning (`ft/`) |
| `analysis/` | confirmatory (`run_confirmatory.py`, `glmm.R`), secondary, cascade, exploratory analyses; figures |
| `prereg/` | pre-registration (protocol, hypotheses, analysis plan, power analysis) |
| `splits/` | fixed test subsets, validation pilot, frozen learning-curve plan |
| `features/` | sentence-level overlap with Indonesian (chrF), language features |
| `runs/` | **all parsed predictions** (one Parquet file per run; schema in `src/jevexp/schema.py`) |
| `results/` | all result tables and figures of the paper; `results/f5_collect/meta` holds per-run training logs |
| `notebooks/`, `scripts/` | resumable Colab fine-tuning and collection scripts |
| `tests/` | unit tests (`pytest -q`) |

Not included are raw API responses, model checkpoints and the raw datasets. The datasets can be re-downloaded
with the fetch command above.

## Reproducing the analysis

The predictions are included, so all analyses run without any API key:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
.venv/bin/python -m jevexp.data fetch
.venv/bin/pytest -q
.venv/bin/python analysis/run_confirmatory.py --split test   # H1a, H1b, H2, H3a (needs R 4.3 + lme4)
.venv/bin/python analysis/run_secondary.py                   # B1/D1 families, translate-test, LLM ladder
.venv/bin/python analysis/run_cascade.py                     # confidence-gated cascades
.venv/bin/python analysis/run_exploratory.py
.venv/bin/python analysis/make_figures.py
```

Re-collecting predictions needs `OPENROUTER_API_KEY` (Jev), `DEEPSEEK_API_KEY`, the Claude Code and Codex CLIs,
and `HF_TOKEN` (fine-tuning). The scripts are `scripts/f4_collect.sh`, `scripts/f4_cli.sh` and
`notebooks/f5_colab.ipynb`. Test-split access is guarded until the `prereg-v1` tag exists.

## Licence

- Code: MIT (`LICENSE`).
- Predictions and result tables, which are derived from CC BY-SA 4.0 data, are released under CC BY-SA 4.0
  (`LICENSE-DATA`).
- Jev outputs are released for benchmarking only. The provider's terms prohibit using them to train or distil
  models.
