# Graph Report - jev  (2026-09-25)

## Corpus Check
- 27 files · ~10,017 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 218 nodes · 463 edges · 13 communities (12 shown, 1 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 5 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `35c5cd4d`
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
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]

## God Nodes (most connected - your core abstractions)
1. `load()` - 15 edges
2. `items()` - 14 edges
3. `RawLog` - 14 edges
4. `run_parallel()` - 14 edges
5. `run()` - 13 edges
6. `run()` - 13 edges
7. `run()` - 13 edges
8. `write()` - 13 edges
9. `run()` - 12 edges
10. `llm_json()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `test_spec_keys_are_canonical_labels()` --calls--> `spec()`  [EXTRACTED]
  tests/test_parsing.py → src/jevexp/questions.py
- `test_anchor_matches_official_notebook_output()` --calls--> `load()`  [EXTRACTED]
  tests/test_baseline.py → src/jevexp/data.py
- `test_subsets_fixed_balanced_and_parallel()` --calls--> `load()`  [EXTRACTED]
  tests/test_data.py → src/jevexp/data.py
- `test_ece_hand_computed()` --calls--> `ece()`  [EXTRACTED]
  tests/test_eval.py → src/jevexp/eval.py
- `test_ece_zero_for_perfectly_calibrated_one_hot()` --calls--> `ece()`  [EXTRACTED]
  tests/test_eval.py → src/jevexp/eval.py

## Import Cycles
- None detected.

## Communities (13 total, 1 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.21
Nodes (14): main(), ops(), F2 pilot summaries (validation only): per-language macro-F1 per run, formulation, scores: {formulation: mean macro-F1}. Rule from prereg/pilot_rules.md., select_formulation(), Per-item prediction schema shared by every arm (docs/metodologi.md §6).  One row, Raise ValueError on any schema violation; return df unchanged otherwise., read() (+6 more)

### Community 1 - "Community 1"
Cohesion: 0.11
Nodes (28): _all_sources(), _consistent_gold(), fetch(), load(), main(), make_pilot(), make_subsets(), _mt_source() (+20 more)

### Community 2 - "Community 2"
Cohesion: 0.11
Nodes (28): accuracy(), aurc(), _bin_edges(), brier(), cluster_bootstrap(), coverage_accuracy_curve(), ece(), macro_f1() (+20 more)

### Community 3 - "Community 3"
Cohesion: 0.12
Nodes (21): Published NusaX sentiment results (macro-F1 %, test split), Winata et al. EACL 2, compare(), main(), _markdown(), F1 gate report: our B1 replication vs NusaX Table 3 (docs/metodologi.md §5, K20,, Rows (lang, clf, variant, f1, feature) for the three variants., variants(), fit_select() (+13 more)

### Community 4 - "Community 4"
Cohesion: 0.11
Nodes (33): ind_source(), sentence_id -> human-written Indonesian parallel text (oracle arm A4).      Nusa, frame(), guard_split(), item_key(), items(), load_env(), now_utc() (+25 more)

### Community 5 - "Community 5"
Cohesion: 0.18
Nodes (16): jev_answer(), llm_json(), logprob_label_dist(), _normalise(), _pred(), Turn raw API responses into (pred, probs, confidence). Probs follow DATASETS lab, Jev System One response -> (pred, probs, confidence). INVALID if the answer is m, LLM JSON answer -> (pred, verbalized probs or None). INVALID on malformed JSON / (+8 more)

### Community 6 - "Community 6"
Cohesion: 0.15
Nodes (22): output_schema(), Single question spec shared by Jev and the LLM arms (docs/metodologi.md §4.2, K3, Request body fields `state` + `questions` for the Jev System One API., Neutral chat prompt (no persona, no mention of Jev) from the same spec; JSON ans, JSON Schema of the LLM answer, enforced by the CLIs (--json-schema for Claude, -, spec(), to_jev(), to_llm_messages() (+14 more)

### Community 8 - "Community 8"
Cohesion: 0.40
Nodes (4): Data (pinned), jevexp — experiments for journal 010, Layout, Setup (reproducible)

### Community 13 - "Community 13"
Cohesion: 0.36
Nodes (8): deepseek_body(), jev_body(), label_distribution(), load_env(), main(), post(), Smoke test of the two paid APIs (F2 prerequisite). Uses hand-written sentences o, Distribution over OPTIONS from top_logprobs at the first token that starts a lab

### Community 14 - "Community 14"
Cohesion: 0.29
Nodes (6): Candidate Jev formulations (RQ4-min), Data, F2 pilot rules (committed BEFORE the first pilot call), Gate, LLM arms, Selection rule for the primary formulation (arm A1)

### Community 15 - "Community 15"
Cohesion: 0.33
Nodes (5): Arms (MVP), Invalid outputs, Items, Protocol (DRAFT for F3 pre-registration — not yet locked), Question spec

## Knowledge Gaps
- **12 isolated node(s):** `Setup (reproducible)`, `Layout`, `Data (pinned)`, `Data`, `Candidate Jev formulations (RQ4-min)` (+7 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `load()` connect `Community 1` to `Community 3`, `Community 4`?**
  _High betweenness centrality (0.041) - this node is a cross-community bridge._
- **Why does `write()` connect `Community 0` to `Community 3`, `Community 4`, `Community 6`?**
  _High betweenness centrality (0.034) - this node is a cross-community bridge._
- **Why does `RawLog` connect `Community 4` to `Community 6`?**
  _High betweenness centrality (0.029) - this node is a cross-community bridge._
- **What connects `Smoke test of the two paid APIs (F2 prerequisite). Uses hand-written sentences o`, `Distribution over OPTIONS from top_logprobs at the first token that starts a lab`, `Experiment harness for journal 010 (see docs/metodologi.md in the journal folder` to the rest of the system?**
  _68 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.11290322580645161 - nodes in this community are weakly interconnected._
- **Should `Community 2` be split into smaller, more focused modules?**
  _Cohesion score 0.1103448275862069 - nodes in this community are weakly interconnected._
- **Should `Community 3` be split into smaller, more focused modules?**
  _Cohesion score 0.12 - nodes in this community are weakly interconnected._