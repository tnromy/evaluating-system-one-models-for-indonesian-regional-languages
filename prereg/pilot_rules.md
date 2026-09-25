# F2 pilot rules (committed BEFORE the first pilot call)

Date: 2026-09-25. Methodology: `docs/metodologi.md` §2 F2 (journal folder). Test data is never read by pilot runners.

## Data
- NusaX sentiment: full **validation** split, 100 sentences × 12 languages = 1,200 items.
- NusaTranslation senti / emot: 100 validation `sentence_id`s (label-stratified, seed 20260924), × 7 parallel
  languages (bew btk jav mad mak min sun) = 700 items each. Ids in `splits/pilot.json`.

## Candidate Jev formulations (RQ4-min)
| name | primitive | question language |
|---|---|---|
| `choice_en` | one Choice over the canonical labels | English |
| `choice_id` | one Choice over the canonical labels | Indonesian |
| `noul_en` | one Noul per label, normalised to a distribution, argmax = prediction | English |

Option keys are always the canonical labels (`negative/neutral/positive`, `anger/fear/happy/love/sadness`).

## Selection rule for the primary formulation (arm A1)
1. Metric: mean over the 12 languages of macro-F1 on NusaX validation.
2. The highest-scoring formulation wins, **unless** it beats `choice_en` by ≤ 1.0 point, in which case
   `choice_en` is chosen (simplest, matches the vendor documentation).
3. The two non-selected formulations stay as RQ4-min conditions in F4.

## LLM arms
- One neutral prompt rendered from the same question spec. **No prompt tuning on validation.** The pilot only
  checks JSON validity, logprob extraction, cost and latency.
- DeepSeek D1: `deepseek-flash`, thinking disabled, temperature 1, `logprobs` + `top_logprobs=20`,
  prediction = argmax of the label logprob distribution; fallback to the JSON `choice` if no label token is found
  (counted and reported).
- Claude D3 (appendix): 30 NusaX validation items, headless per item, to check isolation and parsing only.

## Gate
Every pilot arm: ≥ 99 % valid outputs, complete raw logs, schema-valid Parquet, cost projection for F4.
