# Pre-registered analysis plan (prereg-v1)

Code at the tagged commit is binding: `src/jevexp/eval.py` (metrics), `src/jevexp/parsing.py` (answer parsing),
`analysis/` (confirmatory scripts). This plan supersedes `docs/metodologi.md` §8 where they differ.

## 1. Data handling
- Unit = one item (sentence × language) × one arm × one condition. Items are identical across arms
  (`splits/subsets.json` for subset arms).
- INVALID predictions (unparseable, label outside the options, failure after 3 network retries) count as wrong.
  The INVALID rate is reported per arm. No selective re-runs.
- B1 comparators: NB-BoW, SVM-TF-IDF, LR-BoW (K32); the primary trained comparator is LR-BoW.
- Model versions observed in responses are reported. If a Jev snapshot other than `typesafe/jev-1.13-20260917`
  answers any item, results are split by snapshot and the change is reported.

## 2. Metrics
Macro-F1 (primary), accuracy, ECE (15 equal-width bins; equal-mass as sensitivity), multiclass Brier, AURC,
coverage–accuracy curves, latency p50/p95, cost per 1,000 decisions. Calibration is compared only between arms
of the same `confidence_type` (model probabilities vs logprob-derived vs verbalized are never pooled).

## 3. Paired tests
Exact two-sided McNemar test (binomial on discordant pairs) per language and dataset.
Holm correction within each family:
- F1 = A1 vs B1-LR, 26 tests;
- F2 = A1 vs D1, 26 tests;
- F3 = A4 vs A1 for regional languages, 24 tests.
Effect size = accuracy difference with 95 % cluster-bootstrap CI.

## 4. GLMM (H1a)
R 4.3.3, lme4 1.1-35.1 (Ubuntu packages `r-base-core`, `r-cran-lme4`):
`glmer(correct ~ overlap_z + (1|sentence_id) + (1|lang), family = binomial, control = glmerControl(optimizer = "bobyqa"))`,
one model per dataset, regional languages, A1 predictions. `overlap_z` = `overlap_ind` standardised within dataset.
95 % profile CI.

Pre-declared fallback if the fit is singular or does not converge:
1. drop `(1|lang)`;
2. if still failing, GEE (binomial, exchangeable, clusters = `sentence_id`) with robust SE.

The model actually used is reported.

## 5. Bootstrap
Cluster bootstrap over `sentence_id` (all languages of a sentence resampled together), B = 2,000,
seed 20260924, percentile 95 % CI (`jevexp.eval.cluster_bootstrap`).

## 6. Learning curves (H3a)
Per language, stratified random training subsets of n ∈ {10, 25, 50, 100, 250, 500} (NusaX; NusaTranslation up to
1,000), seeds {0, 1, 2}. B1 hyperparameters are re-selected on the validation split for every subset with the
official protocol. n\* is defined in `hypotheses.md`.

## 7. Order and stopping
1. Collect all MVP arms on test before any confirmatory analysis is run on test.
2. No interim looks.
3. Deviations from this plan are logged in `docs/log_keputusan.md` with date and reason, and reported in the
   paper.
