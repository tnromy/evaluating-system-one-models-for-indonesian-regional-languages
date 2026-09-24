# jevexp — experiments for journal 010

Calibrated typed-decision models ("System One": Jev, Laya) vs trained classifiers and LLMs on Indonesian and regional-language sentiment/emotion. The locked methodology lives in the journal folder: `../docs/metodologi.md`.

## Setup (reproducible)

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt   # every package pinned (pip freeze)
.venv/bin/pip install -e .                  # this package (src/jevexp)
.venv/bin/python -m jevexp.data fetch       # pinned CSVs -> data/raw, verified against data/MANIFEST.json
.venv/bin/pytest -q
```

**Rule:** every newly installed package must be followed by
`.venv/bin/pip freeze --exclude-editable` into `requirements.txt` (keep the 2 header lines).

## Layout

| Path | What | In git |
|---|---|---|
| `src/jevexp/data.py` | pinned fetch, uniform loader, parallelism report, fixed subsets | yes |
| `src/jevexp/eval.py` | accuracy, macro-F1, ECE (width/mass), Brier, coverage–accuracy, AURC, cluster bootstrap | yes |
| `src/jevexp/schema.py` | per-item prediction schema + validation (Parquet) | yes |
| `data/MANIFEST.json` | url, sha256, rows of every CSV | yes |
| `splits/subsets.json` | S-NX / S-NT / S-NE (seed 20260924) | yes |
| `data/raw/`, `runs/`, `results/`, `models/`, `.venv/`, `.env` | regenerable or large or secret | no |

## Data (pinned)

| Dataset | Source @ commit | Languages | Labels | Test |
|---|---|---|---|---|
| `nusax_senti` | IndoNLP/nusax @ 75b0be5 | 12, all parallel | 3 | 400 / lang |
| `nt_senti` | IndoNLP/nusa-writes @ b337f51 | 11; **7 parallel**: bew btk jav mad mak min sun | 2 | 1,200 / lang (parallel) |
| `nt_emot` | IndoNLP/nusa-writes @ b337f51 | 11; same 7 parallel | 5 | 1,000 / lang (parallel) |

`abs, bhp, mui, rej` in NusaTranslation are smaller independent samples (test 500/450). They only partly overlap, so paired analyses use the 7 parallel languages.
