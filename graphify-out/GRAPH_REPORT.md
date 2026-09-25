# Graph Report - jev  (2026-09-25)

## Corpus Check
- 26 files · ~9,097 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 207 nodes · 419 edges · 16 communities (15 shown, 1 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 4 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `b4293dd4`
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

## God Nodes (most connected - your core abstractions)
1. `load()` - 15 edges
2. `run()` - 13 edges
3. `run()` - 13 edges
4. `items()` - 12 edges
5. `RawLog` - 12 edges
6. `run_parallel()` - 12 edges
7. `run()` - 11 edges
8. `write()` - 11 edges
9. `ind_source()` - 8 edges
10. `ece()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `test_anchor_matches_official_notebook_output()` --calls--> `load()`  [EXTRACTED]
  tests/test_baseline.py → src/jevexp/data.py
- `test_subsets_fixed_balanced_and_parallel()` --calls--> `load()`  [EXTRACTED]
  tests/test_data.py → src/jevexp/data.py
- `test_ece_hand_computed()` --calls--> `ece()`  [EXTRACTED]
  tests/test_eval.py → src/jevexp/eval.py
- `test_ece_zero_for_perfectly_calibrated_one_hot()` --calls--> `ece()`  [EXTRACTED]
  tests/test_eval.py → src/jevexp/eval.py
- `test_jev_choice_reorders_probs_to_canonical()` --calls--> `jev_answer()`  [EXTRACTED]
  tests/test_parsing.py → src/jevexp/parsing.py

## Import Cycles
- None detected.

## Communities (16 total, 1 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.43
Nodes (7): Raise ValueError on any schema violation; return df unchanged otherwise., validate(), write(), _row(), test_rejects_missing_required_and_duplicates(), test_rejects_violations(), test_valid_frame_roundtrip()

### Community 1 - "Community 1"
Cohesion: 0.23
Nodes (9): load(), parallel_report(), Uniform frame: dataset, lang, split, sentence_id, text, label (canonical lowerca, Per split: ids shared by all languages, and ids whose gold label differs across, summary(), Gate F0: the pinned data matches docs/metodologi.md §3. Needs `python -m jevexp., test_nusatranslation_parallel_core(), test_nusax_shape_and_parallelism() (+1 more)

### Community 2 - "Community 2"
Cohesion: 0.09
Nodes (34): accuracy(), aurc(), _bin_edges(), brier(), cluster_bootstrap(), coverage_accuracy_curve(), ece(), macro_f1() (+26 more)

### Community 3 - "Community 3"
Cohesion: 0.18
Nodes (14): Per-item prediction schema shared by every arm (docs/metodologi.md §6).  One row, fit_select(), main(), _prediction_frame(), B1: classical baselines, replicating the official NusaX notebook (IndoNLP/nusax@, One notebook run. Frames have columns text, label. Returns dict with fitted sear, All languages x classifiers x features for one dataset; writes runs/B1 and retur, run() (+6 more)

### Community 4 - "Community 4"
Cohesion: 0.13
Nodes (27): call_cli(), main(), D3 (appendix): Claude via headless Claude Code CLI, one isolated call per item (, run(), frame(), item_key(), load_env(), now_utc() (+19 more)

### Community 5 - "Community 5"
Cohesion: 0.12
Nodes (24): jev_answer(), llm_json(), logprob_label_dist(), _normalise(), _pred(), Turn raw API responses into (pred, probs, confidence). Probs follow DATASETS lab, Jev System One response -> (pred, probs, confidence). INVALID if the answer is m, LLM JSON answer -> (pred, verbalized probs or None). INVALID on malformed JSON / (+16 more)

### Community 6 - "Community 6"
Cohesion: 0.27
Nodes (11): _all_sources(), fetch(), main(), _mt_source(), Pinned download, uniform loading and fixed subsets for the three datasets.  Data, Return (url, local relative path) for one CSV at the pinned commit., NusaTranslation MT file (id, ind_text, tgt_text): the Indonesian source of each, Download every CSV at the pinned commit; verify against / write MANIFEST.json. (+3 more)

### Community 7 - "Community 7"
Cohesion: 0.27
Nodes (8): Published NusaX sentiment results (macro-F1 %, test split), Winata et al. EACL 2, compare(), main(), _markdown(), F1 gate report: our B1 replication vs NusaX Table 3 (docs/metodologi.md §5, K20,, Rows (lang, clf, variant, f1, feature) for the three variants., variants(), test_report_select_valid_picks_best_validation()

### Community 8 - "Community 8"
Cohesion: 0.40
Nodes (4): Data (pinned), jevexp — experiments for journal 010, Layout, Setup (reproducible)

### Community 10 - "Community 10"
Cohesion: 0.25
Nodes (10): ind_source(), sentence_id -> human-written Indonesian parallel text (oracle arm A4).      Nusa, guard_split(), items(), Test data is only reachable after the pre-registration tag exists (docs/metodolo, Items (dataset, split, lang, sentence_id, gold, text) restricted to the ids list, main(), A3 translate-test: NLLB-200-distilled-600M, regional language -> Indonesian, loc (+2 more)

### Community 11 - "Community 11"
Cohesion: 0.29
Nodes (8): _consistent_gold(), make_pilot(), make_subsets(), sentence_id -> label for ids present in every language of `frame` with one consi, Proportional label-stratified draw of n ids (largest-remainder allocation). Retu, Fixed test subsets (S-NX, S-NT, S-NE): proportional-stratified by label over sen, F2 pilot items: validation only (docs/metodologi.md §2 F2; prereg/pilot_rules.md, _stratified_ids()

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

- **Why does `load()` connect `Community 1` to `Community 3`, `Community 4`, `Community 6`, `Community 10`, `Community 11`?**
  _High betweenness centrality (0.043) - this node is a cross-community bridge._
- **Why does `write()` connect `Community 0` to `Community 3`, `Community 4`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **What connects `Smoke test of the two paid APIs (F2 prerequisite). Uses hand-written sentences o`, `Distribution over OPTIONS from top_logprobs at the first token that starts a lab`, `Experiment harness for journal 010 (see docs/metodologi.md in the journal folder` to the rest of the system?**
  _65 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 2` be split into smaller, more focused modules?**
  _Cohesion score 0.0945945945945946 - nodes in this community are weakly interconnected._
- **Should `Community 4` be split into smaller, more focused modules?**
  _Cohesion score 0.13277310924369748 - nodes in this community are weakly interconnected._
- **Should `Community 5` be split into smaller, more focused modules?**
  _Cohesion score 0.1225071225071225 - nodes in this community are weakly interconnected._