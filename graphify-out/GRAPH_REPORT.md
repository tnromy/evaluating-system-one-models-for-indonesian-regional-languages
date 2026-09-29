# Graph Report - jev  (2026-09-29)

## Corpus Check
- 54 files · ~23,991 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 478 nodes · 975 edges · 24 communities (19 shown, 5 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 36 edges (avg confidence: 0.76)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `f528f671`
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

## God Nodes (most connected - your core abstractions)
1. `LocalStore` - 23 edges
2. `load()` - 22 edges
3. `Worker` - 20 edges
4. `RawLog` - 16 edges
5. `run_parallel()` - 16 edges
6. `run()` - 14 edges
7. `items()` - 14 edges
8. `run()` - 14 edges
9. `run()` - 14 edges
10. `write()` - 14 edges

## Surprising Connections (you probably didn't know these)
- `test_test_split_requires_explicit_scope()` --calls--> `run_tag()`  [INFERRED]
  tests/test_scope.py → src/jevexp/runners/common.py
- `tiny()` --calls--> `frame()`  [INFERRED]
  tests/test_ft.py → src/jevexp/runners/common.py
- `main()` --calls--> `guard_split()`  [INFERRED]
  analysis/run_confirmatory.py → src/jevexp/runners/common.py
- `test_pred_frame_valid()` --calls--> `pred_frame()`  [INFERRED]
  tests/test_ft.py → src/jevexp/ft/common.py
- `_ctx()` --calls--> `Checkpoint`  [INFERRED]
  tests/test_ft.py → src/jevexp/ft/common.py

## Import Cycles
- None detected.

## Communities (24 total, 5 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.09
Nodes (30): discordance(), main(), mcnemar_crit(), mdd(), power(), Power analysis for prereg-v1 (K25), sized from VALIDATION pilot data only.  McNe, Smallest k such that the exact two-sided test on m discordant pairs rejects when, install() (+22 more)

### Community 1 - "Community 1"
Cohesion: 0.08
Nodes (39): _all_sources(), _consistent_gold(), fetch(), ind_source(), load(), main(), make_pilot(), make_subsets() (+31 more)

### Community 2 - "Community 2"
Cohesion: 0.08
Nodes (43): a4_frame(), h1a(), h1b(), h2(), holm(), load_arm(), main(), mcnemar() (+35 more)

### Community 3 - "Community 3"
Cohesion: 0.12
Nodes (23): Published NusaX sentiment results (macro-F1 %, test split), Winata et al. EACL 2, compare(), main(), _markdown(), F1 gate report: our B1 replication vs NusaX Table 3 (docs/metodologi.md §5, K20,, Rows (lang, clf, variant, f1, feature) for the three variants., variants(), fit_select() (+15 more)

### Community 4 - "Community 4"
Cohesion: 0.07
Nodes (24): Checkpoint, condition(), env_versions(), gpu_name(), now_utc(), pred_frame(), Shared F5 plumbing: run data, prediction frames, resumable checkpoints., Rolling `runs/<id>/ckpt/latest.pt` in the store, written at most every `every_s` (+16 more)

### Community 5 - "Community 5"
Cohesion: 0.11
Nodes (35): device(), hf_kwargs(), rng_state(), set_rng_state(), _batch(), encode(), export(), load_model() (+27 more)

### Community 6 - "Community 6"
Cohesion: 0.06
Nodes (75): jev_answer(), llm_json(), logprob_label_dist(), _normalise(), _pred(), Turn raw API responses into (pred, probs, confidence). Probs follow DATASETS lab, Jev System One response -> (pred, probs, confidence). INVALID if the answer is m, LLM JSON answer -> (pred, verbalized probs or None). INVALID on malformed JSON / (+67 more)

### Community 7 - "Community 7"
Cohesion: 0.14
Nodes (3): boto3 client (see the notebook, Cell 2) + bucket + prefix. Secrets never pass th, S3Store, Store

### Community 8 - "Community 8"
Cohesion: 0.40
Nodes (4): Data (pinned), jevexp — experiments for journal 010, Layout, Setup (reproducible)

### Community 10 - "Community 10"
Cohesion: 0.22
Nodes (8): 1. Data handling, 2. Metrics, 3. Paired tests, 4. GLMM (H1a), 5. Bootstrap, 6. Learning curves (H3a), 7. Order and stopping, Pre-registered analysis plan (prereg-v1)

### Community 11 - "Community 11"
Cohesion: 0.33
Nodes (5): Confirmatory, Disclosure, Exploratory, Pre-registered hypotheses (prereg-v1), Secondary (reported with CIs, Holm within family, not used to claim confirmation)

### Community 13 - "Community 13"
Cohesion: 0.36
Nodes (8): deepseek_body(), jev_body(), label_distribution(), load_env(), main(), post(), Smoke test of the two paid APIs (F2 prerequisite). Uses hand-written sentences o, Distribution over OPTIONS from top_logprobs at the first token that starts a lab

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
Cohesion: 0.12
Nodes (20): Exception, LocalStore, _ctx(), _mini_plan(), F5 plumbing: store, learning-curve subsets, resumable training, idempotent worke, The worker passes max_epochs=None (Colab default): training must fall back to MA, NT runs without history borrow the same arm's NusaX per-language duration x trai, Stop (+12 more)

### Community 19 - "Community 19"
Cohesion: 0.83
Nodes (3): f4_collect.sh script, check(), step()

### Community 23 - "Community 23"
Cohesion: 0.10
Nodes (24): _alloc(), build(), curve_subsets(), dumps(), load_plan(), main(), F5 run plan (docs/metodologi.md §4, K60/K61): every fine-tuning run, frozen in `, At least one per label, the rest proportional by largest remainder. (+16 more)

## Knowledge Gaps
- **29 isolated node(s):** `f4_cli.sh script`, `f5_bundle.sh script`, `Setup (reproducible)`, `Layout`, `Data (pinned)` (+24 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Worker` connect `Community 4` to `Community 0`, `Community 23`?**
  _High betweenness centrality (0.065) - this node is a cross-community bridge._
- **Why does `LocalStore` connect `Community 18` to `Community 0`, `Community 7`?**
  _High betweenness centrality (0.042) - this node is a cross-community bridge._
- **Why does `S3Store` connect `Community 7` to `Community 0`, `Community 18`?**
  _High betweenness centrality (0.035) - this node is a cross-community bridge._
- **Are the 12 inferred relationships involving `LocalStore` (e.g. with `Stop` and `test_collect_incremental_and_skipped_when_nothing_ran()`) actually correct?**
  _`LocalStore` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `Worker` (e.g. with `Checkpoint` and `Progress`) actually correct?**
  _`Worker` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `RawLog` (e.g. with `test_infra_records_are_not_logged_and_are_retried()` and `test_stops_after_consecutive_infra_failures()`) actually correct?**
  _`RawLog` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `run_parallel()` (e.g. with `test_infra_records_are_not_logged_and_are_retried()` and `test_stops_after_consecutive_infra_failures()`) actually correct?**
  _`run_parallel()` has 2 INFERRED edges - model-reasoned connections that need verification._