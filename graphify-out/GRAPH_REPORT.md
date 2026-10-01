# Graph Report - jev  (2026-10-01)

## Corpus Check
- 62 files · ~30,649 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 599 nodes · 1212 edges · 35 communities (30 shown, 5 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 44 edges (avg confidence: 0.77)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `1404c4a2`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]

## God Nodes (most connected - your core abstractions)
1. `LocalStore` - 23 edges
2. `load()` - 22 edges
3. `read()` - 21 edges
4. `Worker` - 20 edges
5. `RawLog` - 16 edges
6. `run_parallel()` - 16 edges
7. `run()` - 14 edges
8. `items()` - 14 edges
9. `run_tag()` - 14 edges
10. `run()` - 14 edges

## Surprising Connections (you probably didn't know these)
- `test_test_split_requires_explicit_scope()` --calls--> `run_tag()`  [INFERRED]
  tests/test_scope.py → src/jevexp/runners/common.py
- `tiny()` --calls--> `frame()`  [INFERRED]
  tests/test_ft.py → src/jevexp/runners/common.py
- `measure()` --calls--> `predict()`  [INFERRED]
  analysis/latency_local.py → src/jevexp/ft/hf_clf.py
- `main()` --calls--> `guard_split()`  [INFERRED]
  analysis/latency_local.py → src/jevexp/runners/common.py
- `main()` --calls--> `guard_split()`  [INFERRED]
  analysis/run_cascade.py → src/jevexp/runners/common.py

## Import Cycles
- None detected.

## Communities (35 total, 5 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.27
Nodes (9): _parquet_bytes(), Per-item prediction schema shared by every arm (docs/metodologi.md §6).  One row, Raise ValueError on any schema violation; return df unchanged otherwise., validate(), test_pred_frame_valid(), _row(), test_rejects_missing_required_and_duplicates(), test_rejects_violations() (+1 more)

### Community 1 - "Community 1"
Cohesion: 0.16
Nodes (21): align(), build(), cascade_preds(), choose_rate(), curve(), diff_ci(), fallbacks(), local_latency() (+13 more)

### Community 2 - "Community 2"
Cohesion: 0.18
Nodes (18): arms(), f2_calibration(), h1c(), ladder(), latest(), main(), metrics(), metrics_table() (+10 more)

### Community 3 - "Community 3"
Cohesion: 0.07
Nodes (48): _alloc(), build(), curve_subsets(), dumps(), load_plan(), main(), F5 run plan (docs/metodologi.md §4, K60/K61): every fine-tuning run, frozen in `, At least one per label, the rest proportional by largest remainder. (+40 more)

### Community 4 - "Community 4"
Cohesion: 0.06
Nodes (25): Checkpoint, condition(), env_versions(), gpu_name(), now_utc(), pred_frame(), Shared F5 plumbing: run data, prediction frames, resumable checkpoints., Rolling `runs/<id>/ckpt/latest.pt` in the store, written at most every `every_s` (+17 more)

### Community 5 - "Community 5"
Cohesion: 0.11
Nodes (35): device(), hf_kwargs(), rng_state(), set_rng_state(), _batch(), encode(), export(), load_model() (+27 more)

### Community 6 - "Community 6"
Cohesion: 0.16
Nodes (17): _bin_edges(), brier(), cluster_bootstrap(), ece(), Metrics shared by every arm (docs/metodologi.md §7-8).  Conventions:   y_true, y, Per bin: (count, mean confidence, accuracy). y_idx = gold class index into probs, Expected calibration error of the top-class confidence (Naeini 2015; adaptive =, Multiclass Brier score: mean over items of sum_k (p_k - 1[y=k])^2. (+9 more)

### Community 7 - "Community 7"
Cohesion: 0.14
Nodes (23): a4_frame(), curve_preds(), f1_draws(), gee_fit(), h1a(), h1b(), h2(), h3a() (+15 more)

### Community 8 - "Community 8"
Cohesion: 0.17
Nodes (11): Classical baselines (B1, following the NusaX protocol, scikit-learn), Data, Evaluating System One Models for Indonesian Regional Languages, Fine-tuned encoders, Large language models, Licence, Pre-registration, Repository layout (+3 more)

### Community 10 - "Community 10"
Cohesion: 0.22
Nodes (8): 1. Data handling, 2. Metrics, 3. Paired tests, 4. GLMM (H1a), 5. Bootstrap, 6. Learning curves (H3a), 7. Order and stopping, Pre-registered analysis plan (prereg-v1)

### Community 11 - "Community 11"
Cohesion: 0.33
Nodes (5): Confirmatory, Disclosure, Exploratory, Pre-registered hypotheses (prereg-v1), Secondary (reported with CIs, Holm within family, not used to claim confirmation)

### Community 13 - "Community 13"
Cohesion: 0.13
Nodes (25): CommaFormatter, fig_cascade(), fig_forest(), fig_frontier(), fig_learning_curves(), fig_reliability(), localise(), mark() (+17 more)

### Community 14 - "Community 14"
Cohesion: 0.29
Nodes (6): Candidate Jev formulations (RQ4-min), Data, F2 pilot rules (committed BEFORE the first pilot call), Gate, LLM arms, Selection rule for the primary formulation (arm A1)

### Community 15 - "Community 15"
Cohesion: 0.29
Nodes (6): Arms (MVP), Invalid outputs, Items, Observed versions at registration (2026-09-25), Protocol (prereg-v1), Question spec

### Community 16 - "Community 16"
Cohesion: 0.50
Nodes (3): GLMM for H1a, Paired McNemar per language (families F2 and F3), Power analysis (prereg-v1, K25)

### Community 18 - "Community 18"
Cohesion: 0.07
Nodes (23): Exception, LocalStore, boto3 client (see the notebook, Cell 2) + bucket + prefix. Secrets never pass th, S3Store, Store, _ctx(), _mini_plan(), F5 plumbing: store, learning-curve subsets, resumable training, idempotent worke (+15 more)

### Community 19 - "Community 19"
Cohesion: 0.83
Nodes (3): f4_collect.sh script, check(), step()

### Community 23 - "Community 23"
Cohesion: 0.11
Nodes (21): _bar(), _h(), _keys(), main(), Progress, F5 progress + ETA from the store (K63). Used by the Colab worker (printed to the, Empirical groups, most specific first. The regime is always kept (a pooled run s, Wall seconds of a finished run; older metas (before K63) only have train_time_s. (+13 more)

### Community 24 - "Community 24"
Cohesion: 0.06
Nodes (77): laya_predictor(), jev_answer(), llm_json(), logprob_label_dist(), _normalise(), _pred(), Turn raw API responses into (pred, probs, confidence). Probs follow DATASETS lab, Jev System One response -> (pred, probs, confidence). INVALID if the answer is m (+69 more)

### Community 25 - "Community 25"
Cohesion: 0.43
Nodes (5): extract(), main(), measure(), Per-decision latency of the local fallback models on the laptop GPU (H3b cascade, texts()

### Community 26 - "Community 26"
Cohesion: 0.22
Nodes (6): _curve(), F6 additions to analysis/run_confirmatory.py (K67): test-file prefix, GEE fallba, Curve whose accuracy at n is acc_by_n[n]; deterministic, identical items across, With one bad seed the pooled 2.5th percentile falls far below the mean-based one, test_nstar_picks_smallest_n_whose_lower_bound_reaches_target(), test_nstar_pools_seed_draws()

### Community 27 - "Community 27"
Cohesion: 0.12
Nodes (24): Published NusaX sentiment results (macro-F1 %, test split), Winata et al. EACL 2, compare(), main(), _markdown(), F1 gate report: our B1 replication vs NusaX Table 3 (docs/metodologi.md §5, K20,, Rows (lang, clf, variant, f1, feature) for the three variants., variants(), write() (+16 more)

### Community 28 - "Community 28"
Cohesion: 0.27
Nodes (11): accuracy(), macro_f1(), Macro-F1 over the task's labels only; INVALID predictions count as errors for ev, main(), ops(), per_lang(), F2 pilot summaries (validation only): per-language macro-F1 per run, formulation, scores: {formulation: mean macro-F1}. Rule from prereg/pilot_rules.md. (+3 more)

### Community 29 - "Community 29"
Cohesion: 0.39
Nodes (7): discordance(), main(), mcnemar_crit(), mdd(), power(), Power analysis for prereg-v1 (K25), sized from VALIDATION pilot data only.  McNe, Smallest k such that the exact two-sided test on m discordant pairs rejects when

### Community 30 - "Community 30"
Cohesion: 0.46
Nodes (7): emotions(), lang_features(), main(), EXPLORATORY analyses (hypotheses.md "Exploratory"; labelled exploratory in the p, reliability(), seeds(), read()

### Community 31 - "Community 31"
Cohesion: 0.50
Nodes (5): aurc(), coverage_accuracy_curve(), Sort by confidence (desc); return coverage and accuracy of the kept prefix at ev, Area under the risk-coverage curve (Geifman et al. 2019): mean selective risk ov, test_coverage_curve_and_aurc()

### Community 32 - "Community 32"
Cohesion: 0.43
Nodes (5): _frames(), H3b cascade simulation (analysis/run_cascade.py, K71). Synthetic data only., test_extremes_are_primary_and_fallback_and_cost_is_linear(), test_multi_seed_fallback_is_averaged(), test_routing_takes_lowest_confidence_first_and_is_stable()

### Community 33 - "Community 33"
Cohesion: 0.53
Nodes (5): dataset_sizes(), f4(), f5(), main(), Design facts for the manuscript (journal 010, F7) that no result CSV carries.  D

### Community 34 - "Community 34"
Cohesion: 0.70
Nodes (4): check(), main(), F4 collection monitor. Deliberately computes NO performance metric (no accuracy, status()

## Knowledge Gaps
- **35 isolated node(s):** `f4_cli.sh script`, `f5_bundle.sh script`, `Data`, `System One (typed-decision) models`, `Classical baselines (B1, following the NusaX protocol, scikit-learn)` (+30 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Worker` connect `Community 4` to `Community 23`?**
  _High betweenness centrality (0.054) - this node is a cross-community bridge._
- **Why does `LocalStore` connect `Community 18` to `Community 23`?**
  _High betweenness centrality (0.033) - this node is a cross-community bridge._
- **Why does `S3Store` connect `Community 18` to `Community 23`?**
  _High betweenness centrality (0.029) - this node is a cross-community bridge._
- **Are the 12 inferred relationships involving `LocalStore` (e.g. with `Stop` and `test_collect_incremental_and_skipped_when_nothing_ran()`) actually correct?**
  _`LocalStore` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `Worker` (e.g. with `Checkpoint` and `Progress`) actually correct?**
  _`Worker` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `RawLog` (e.g. with `test_infra_records_are_not_logged_and_are_retried()` and `test_stops_after_consecutive_infra_failures()`) actually correct?**
  _`RawLog` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Per-decision latency of the local fallback models on the laptop GPU (H3b cascade`, `Manuscript figures from the F6 result CSVs only (no analysis is recomputed). Out`, `Number text in the figure language (decimal comma for id).` to the rest of the system?**
  _171 weakly-connected nodes found - possible documentation gaps or missing edges._