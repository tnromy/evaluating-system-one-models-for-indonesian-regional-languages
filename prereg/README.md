# Pre-registration `prereg-v1`

| File | Content |
|---|---|
| `pilot_rules.md` | F2 pilot rules, committed before any pilot call (`88a72d9`) |
| `hypotheses.md` | Confirmatory / secondary / exploratory hypotheses + disclosure of the validation dry-run |
| `analysis_plan.md` | Tests, models, corrections, fallbacks, stopping rule (binding) |
| `protocol.md` | Arms, endpoints, model IDs, settings, observed versions |
| `power.md` (+ `power_*.csv`) | Power analysis from validation pilot data |
| `SHA256SUMS` | Hashes of every binding file (prereg, analysis code, spec/parsing/eval/data code, subsets, requirements) |

Verify: `sha256sum -c prereg/SHA256SUMS` from the repo root, and `git show prereg-v1`.

Test-split access before this tag:
- B1 classical baselines only (K31).
- Test texts, ids and labels were also read for subset sampling (F0) and for the `overlap_ind` data feature.
  No model predictions were produced on them.
