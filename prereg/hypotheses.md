# Pre-registered hypotheses (prereg-v1)

Study: do calibrated typed-decision ("System One") models transfer to low-resource languages, and at what label
budget do small trained models win? Instances: Jev (TypeSafe, via OpenRouter) and Laya-multilingual.
Data: test splits of NusaX-senti (12 languages × 400), NusaTranslation-senti (7 parallel languages × 1,200) and
NusaTranslation-emot (7 × 1,000). "Regional" = every language except `ind` and `eng`.
Arms and settings: `protocol.md`. Tests, models and corrections: `analysis_plan.md`.
All results on validation (F2 pilot) are preliminary and were used only to choose the Jev formulation (rule
committed in `pilot_rules.md` before any call) and to size this study (`power.md`).

## Confirmatory

**H1a — resource gradient (sentence level).** For Jev A1 (`noul_en`), the probability of a correct answer
increases with `overlap_ind`, the sentence chrF between the regional sentence and its human Indonesian parallel
(`features/sentence_overlap.parquet`).
Test: GLMM per dataset, regional languages only. Confirmed if β(overlap_z) > 0 and its 95 % profile CI excludes 0.

**H1b — cost of not translating.** For each regional language, Jev on the human Indonesian parallel (A4) is more
accurate than Jev on the regional text (A1).
Test: paired exact McNemar per language (family F3, Holm). Confirmed if A4 > A1 significantly in a majority of
regional languages of each dataset.

**H2a — calibration does not transfer.** Using Jev's native Choice distribution (A1-choice, `choice_en`), the
expected calibration error on regional languages pooled exceeds ECE on Indonesian (`ind` for NusaX; Jev
`choice_en` on the human Indonesian parallel, arm A4-choice, for NusaTranslation). Prediction from the literature on
multilingual calibration (J35).
Test: ΔECE = ECE(regional) − ECE(ind), 15 equal-width bins, 95 % cluster-bootstrap CI resampling `sentence_id`
(B = 2,000). Confirmed if the CI lies above 0. Co-reported: mean overconfidence (confidence − accuracy) per group.

**H2b — selective prediction degrades.** AURC of A1-choice is higher on regional languages than on `ind`.
Test: ΔAURC with the same bootstrap. Confirmed if the CI lies above 0.

**H3a — label budget.** For each language, n\* is the smallest n ∈ {10, 25, 50, 100, 250, 500} (NusaX) at which
the lower 95 % bound (over 3 seeds × bootstrap) of a trained model's macro-F1 reaches Jev A1's macro-F1; n\* = ">500"
if never.
Prediction: n\* is smaller where Jev is weaker. Test: Spearman ρ(n\*, Jev macro-F1) > 0 across NusaX regional
languages (one-sided, α = 0.05). Trained models: B1-LR (primary), B2 IndoBERT, C1 Laya-ft (secondary).

## Secondary (reported with CIs, Holm within family, not used to claim confirmation)
- **F1 family:** A1 vs B1-LR (full training data) per language, McNemar.
- **F2 family:** A1 vs D1 (DeepSeek, logprob) per language, McNemar. Calibration A1-choice vs D1 (same confidence
  type: model probabilities).
- **H1c:** A3 − A1 (NLLB translate-test) per language, related to NLLB chrF (descriptive; 7 languages).
- **H3b:** offline confidence-gated cascade Jev → local trained model; cost & macro-F1 curves.
- **H3-ladder:** cost–accuracy Pareto of A1, D1, Claude (Haiku, Sonnet, Opus), GPT (Luna, Sol, Terra), B1–B3, C1
  on S-NX / S-NT / S-NE.
- **RQ4-min:** `noul_en` vs `choice_en` vs `choice_id` on the subsets.

## Exploratory
Everything else, labelled "exploratory" in the paper (e.g. genetic distance, per-emotion errors, appendix arms).

## Disclosure
Before tagging, the analysis code was dry-run on the VALIDATION pilot to verify it executes
(`results/confirmatory_valid/`). Those outputs were seen by the authors (e.g. on validation ΔECE was not positive
while ΔAURC was). The hypotheses above were NOT changed after the dry-run; they follow the literature-based
predictions written in `docs/rencana_penelitian.md` v0.2–v0.5 (2026-09-23/24).
