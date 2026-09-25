# Power analysis (prereg-v1, K25)

Sized from the VALIDATION pilot only (F2). Scripts: `analysis/power.py` (McNemar, exact) and
`analysis/power_glmm.R` (GLMM simulation). Raw numbers: `power_mcnemar.csv`, `power_glmm.csv`.

## Paired McNemar per language (families F2 and F3)
Minimum detectable accuracy difference at power 0.8, exact two-sided test; "Holm" = worst-case α = 0.05/26.

| Comparison | Dataset | Pilot discordance | Pilot diff | Test n / language | MDD α = 0.05 | MDD Holm |
|---|---|---|---|---|---|---|
| A1 vs D1 | NusaX | 0.195 | −0.003 | 400 | 0.065 | 0.090 |
| A1 vs D1 | NT-senti | 0.107 | −0.004 | 1,200 | 0.030 | 0.040 |
| A1 vs D1 | NT-emot | 0.194 | −0.037 | 1,000 | 0.040 | 0.060 |
| A4 vs A1 | NusaX | 0.190 | +0.100 | 400 | 0.065 | 0.090 |
| A4 vs A1 | NT-senti | 0.146 | +0.091 | 1,200 | 0.035 | 0.045 |
| A4 vs A1 | NT-emot | 0.264 | +0.133 | 1,000 | 0.050 | 0.065 |

Reading:
- The per-language NusaX tests detect only differences ≥ 9 points after Holm. Per-language NusaX null results
  are therefore reported as "not detected at this size", never as equivalence.
- H1b effects seen in the pilot (+9 to +13 points) exceed the MDD in NusaTranslation.
- The F1 family (A1 vs B1-LR) is sized with the same discordance proxy; B1 has no validation predictions in the
  pilot.

## GLMM for H1a
Parametric simulation from the pilot fit `glmer(correct ~ overlap_z + (1|sentence_id) + (1|lang))`, test-size data
(test sentences × regional languages), 200 simulations per cell, one-sided Wald z > 1.645.

| Dataset | Pilot β | Test sentences × languages | Power (β as pilot) | Power (β × 0.5) |
|---|---|---|---|---|
| NusaX | 0.551 | 400 × 10 | 1.00 | 1.00 |
| NT-senti | 0.748 | 1,200 × 7 | 1.00 | 1.00 |
| NT-emot | 0.355 | 1,000 × 7 | 1.00 | 1.00 |

Caveats:
- The simulation keeps the pilot's language effects (BLUPs) fixed and uses the Wald test, so it is optimistic.
- H1a is nonetheless well powered: the sentence-level covariate varies within every language, which makes this
  test far more powerful than a language-level (10 points) analysis would be.

Environment: R 4.3.3, lme4 1.1.35.1 (Ubuntu `r-base-core`, `r-cran-lme4`), seed 20260924, run 2026-09-25 (16 min).
