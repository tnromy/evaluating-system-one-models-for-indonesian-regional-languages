# Graph Report - jev  (2026-09-25)

## Corpus Check
- 37 files · ~13,566 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 268 nodes · 541 edges · 18 communities (16 shown, 2 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 6 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `b048e468`
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

## God Nodes (most connected - your core abstractions)
1. `load()` - 17 edges
2. `items()` - 14 edges
3. `RawLog` - 14 edges
4. `run_parallel()` - 14 edges
5. `run()` - 13 edges
6. `run()` - 13 edges
7. `run()` - 13 edges
8. `write()` - 13 edges
9. `run()` - 12 edges
10. `ind_source()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `load_arm()` --calls--> `read()`  [EXTRACTED]
  analysis/run_confirmatory.py → src/jevexp/schema.py
- `h2()` --calls--> `aurc()`  [EXTRACTED]
  analysis/run_confirmatory.py → src/jevexp/eval.py
- `h2()` --calls--> `cluster_bootstrap()`  [EXTRACTED]
  analysis/run_confirmatory.py → src/jevexp/eval.py
- `h2()` --calls--> `ece()`  [EXTRACTED]
  analysis/run_confirmatory.py → src/jevexp/eval.py
- `main()` --calls--> `guard_split()`  [INFERRED]
  analysis/run_confirmatory.py → src/jevexp/runners/common.py

## Import Cycles
- None detected.

## Communities (18 total, 2 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.19
Nodes (16): discordance(), main(), mcnemar_crit(), mdd(), power(), Power analysis for prereg-v1 (K25), sized from VALIDATION pilot data only.  McNe, Smallest k such that the exact two-sided test on m discordant pairs rejects when, Per-item prediction schema shared by every arm (docs/metodologi.md §6).  One row (+8 more)

### Community 1 - "Community 1"
Cohesion: 0.12
Nodes (28): _all_sources(), _consistent_gold(), fetch(), load(), main(), make_pilot(), make_subsets(), _mt_source() (+20 more)

### Community 2 - "Community 2"
Cohesion: 0.10
Nodes (33): accuracy(), aurc(), _bin_edges(), brier(), cluster_bootstrap(), coverage_accuracy_curve(), ece(), macro_f1() (+25 more)

### Community 3 - "Community 3"
Cohesion: 0.10
Nodes (21): Published NusaX sentiment results (macro-F1 %, test split), Winata et al. EACL 2, compare(), main(), _markdown(), F1 gate report: our B1 replication vs NusaX Table 3 (docs/metodologi.md §5, K20,, Rows (lang, clf, variant, f1, feature) for the three variants., variants(), fit_select() (+13 more)

### Community 4 - "Community 4"
Cohesion: 0.13
Nodes (27): call_cli(), main(), D3 (appendix): Claude via headless Claude Code CLI, one isolated call per item (, run(), frame(), item_key(), load_env(), now_utc() (+19 more)

### Community 5 - "Community 5"
Cohesion: 0.21
Nodes (13): ind_source(), sentence_id -> human-written Indonesian parallel text (oracle arm A4).      Nusa, _cosine_distance(), language_features(), main(), Covariates for H1 (docs/metodologi.md §8; prereg/hypotheses.md).  Sentence level, sentence_overlap(), items() (+5 more)

### Community 6 - "Community 6"
Cohesion: 0.10
Nodes (34): jev_answer(), llm_json(), logprob_label_dist(), _normalise(), _pred(), Turn raw API responses into (pred, probs, confidence). Probs follow DATASETS lab, Jev System One response -> (pred, probs, confidence). INVALID if the answer is m, LLM JSON answer -> (pred, verbalized probs or None). INVALID on malformed JSON / (+26 more)

### Community 7 - "Community 7"
Cohesion: 0.28
Nodes (12): a4_frame(), h1a(), h1b(), h2(), holm(), load_arm(), main(), mcnemar() (+4 more)

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

## Knowledge Gaps
- **27 isolated node(s):** `Setup (reproducible)`, `Layout`, `Data (pinned)`, `Pre-registration `prereg-v1``, `1. Data handling` (+22 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `load()` connect `Community 1` to `Community 3`, `Community 4`, `Community 5`?**
  _High betweenness centrality (0.034) - this node is a cross-community bridge._
- **Why does `write()` connect `Community 0` to `Community 3`, `Community 4`, `Community 6`?**
  _High betweenness centrality (0.023) - this node is a cross-community bridge._
- **Why does `RawLog` connect `Community 4` to `Community 6`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._
- **What connects `Power analysis for prereg-v1 (K25), sized from VALIDATION pilot data only.  McNe`, `Smallest k such that the exact two-sided test on m discordant pairs rejects when`, `Confirmatory analyses of prereg-v1 (hypotheses.md, analysis_plan.md).    --split` to the rest of the system?**
  _88 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.11612903225806452 - nodes in this community are weakly interconnected._
- **Should `Community 2` be split into smaller, more focused modules?**
  _Cohesion score 0.09682539682539683 - nodes in this community are weakly interconnected._
- **Should `Community 3` be split into smaller, more focused modules?**
  _Cohesion score 0.10344827586206896 - nodes in this community are weakly interconnected._