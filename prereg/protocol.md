# Protocol (DRAFT for F3 pre-registration — not yet locked)

Status: draft from the F2 pilot (2026-09-25). Becomes binding when tagged `prereg-v1` together with
`hypotheses.md`, `analysis_plan.md` and the SHA-256 of these files.

## Question spec
`src/jevexp/questions.py` at the tagged commit (single spec for Jev and LLMs; canonical labels as option keys).

## Arms (MVP)
| Arm | Endpoint / model | Settings |
|---|---|---|
| A1 Jev | OpenRouter `POST /api/v1/systemone`, model `jev-1.13` (observed snapshot `typesafe/jev-1.13-20260917`) | formulation **`noul_en`** (selected by the pilot rule) |
| A1-choice (K40, approved 2026-09-25) | same | `choice_en`, full test; primary distribution for H2 calibration |
| RQ4-min | same | `choice_en`, `choice_id` on S-NX / S-NT / S-NE |
| A3 | NLLB-200-distilled-600M fp16 greedy max_length 256 → `ind_Latn`, then Jev `noul_en` | NusaX ace ban bjn bug jav min sun; NT jav min sun |
| A4 | human Indonesian parallel text → Jev `noul_en` | NusaX via `ind`; NT via MT `ind_text` (normalised, majority) |
| D1 | DeepSeek direct, `deepseek-flash`, thinking disabled, temperature 1, `logprobs`, `top_logprobs=20`, `response_format json_object` | prediction = argmax of label logprob distribution |
| B1 | as `docs/laporan_f1.md` (K32) | NB-BoW, SVM-TF-IDF, LR-BoW |

Secondary "LLM reference ladder" (S-NX/S-NT/S-NE, verbalized confidence only):
D3a–c `claude -p` (`claude-haiku-4-5-20251001`, `claude-sonnet-5`, `claude-opus-5-5`);
D4a–c `codex exec`, effort low (`gpt-6-luna`, `gpt-6-sol`, `gpt-5.6-terra`).

Appendix arms: D2 (`deepseek-flash` thinking, `deepseek-v4-pro`), A2, C2, RQ4-ext.

## Invalid outputs
Counted as wrong, reported per arm; no selective retries (network-level retries only, max 3).

## Items
Test split: NusaX 12 × 400; NT-senti 7 × 1,200; NT-emot 7 × 1,000 (parallel languages). Subsets: `splits/subsets.json`.
