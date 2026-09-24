# Graph Report - jev  (2026-09-24)

## Corpus Check
- 14 files · ~3,869 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 96 nodes · 158 edges · 13 communities (12 shown, 1 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 4 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `fd6990f7`
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

## God Nodes (most connected - your core abstractions)
1. `load()` - 11 edges
2. `run()` - 7 edges
3. `validate()` - 7 edges
4. `fetch()` - 6 edges
5. `reliability_bins()` - 6 edges
6. `ece()` - 6 edges
7. `fit_select()` - 6 edges
8. `parallel_report()` - 5 edges
9. `write()` - 5 edges
10. `test_fit_select_and_schema()` - 5 edges

## Surprising Connections (you probably didn't know these)
- `test_anchor_matches_official_notebook_output()` --calls--> `load()`  [EXTRACTED]
  tests/test_baseline.py → src/jevexp/data.py
- `test_subsets_fixed_balanced_and_parallel()` --calls--> `load()`  [EXTRACTED]
  tests/test_data.py → src/jevexp/data.py
- `test_reliability_bins_right_closed_edges()` --calls--> `reliability_bins()`  [EXTRACTED]
  tests/test_eval.py → src/jevexp/eval.py
- `test_ece_hand_computed()` --calls--> `ece()`  [EXTRACTED]
  tests/test_eval.py → src/jevexp/eval.py
- `test_ece_zero_for_perfectly_calibrated_one_hot()` --calls--> `ece()`  [EXTRACTED]
  tests/test_eval.py → src/jevexp/eval.py

## Import Cycles
- None detected.

## Communities (13 total, 1 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.35
Nodes (9): Per-item prediction schema shared by every arm (docs/metodologi.md §6).  One row, Raise ValueError on any schema violation; return df unchanged otherwise., read(), validate(), write(), _row(), test_rejects_missing_required_and_duplicates(), test_rejects_violations() (+1 more)

### Community 1 - "Community 1"
Cohesion: 0.15
Nodes (19): fetch(), load(), main(), make_subsets(), parallel_report(), Pinned download, uniform loading and fixed subsets for the three datasets.  Data, Per split: ids shared by all languages, and ids whose gold label differs across, Fixed test subsets (S-NX, S-NT, S-NE): proportional-stratified by label over sen (+11 more)

### Community 2 - "Community 2"
Cohesion: 0.47
Nodes (5): _bin_edges(), Metrics shared by every arm (docs/metodologi.md §7-8).  Conventions:   y_true, y, Per bin: (count, mean confidence, accuracy). y_idx = gold class index into probs, reliability_bins(), _top()

### Community 3 - "Community 3"
Cohesion: 0.19
Nodes (14): fit_select(), main(), _prediction_frame(), B1: classical baselines, replicating the official NusaX notebook (IndoNLP/nusax@, One notebook run. Frames have columns text, label. Returns dict with fitted sear, All languages x classifiers x features for one dataset; writes runs/B1 and retur, run(), tokenize() (+6 more)

### Community 4 - "Community 4"
Cohesion: 0.50
Nodes (5): accuracy(), macro_f1(), Macro-F1 over the task's labels only; INVALID predictions count as errors for ev, test_accuracy_and_macro_f1_match_sklearn(), test_invalid_counts_as_wrong()

### Community 5 - "Community 5"
Cohesion: 0.50
Nodes (5): aurc(), coverage_accuracy_curve(), Sort by confidence (desc); return coverage and accuracy of the kept prefix at ev, Area under the risk-coverage curve (Geifman et al. 2019): mean selective risk ov, test_coverage_curve_and_aurc()

### Community 6 - "Community 6"
Cohesion: 0.50
Nodes (4): cluster_bootstrap(), Percentile CI resampling whole clusters (all languages of a sentence move togeth, test_cluster_bootstrap_deterministic_and_brackets_point(), test_cluster_bootstrap_resamples_whole_clusters()

### Community 7 - "Community 7"
Cohesion: 0.31
Nodes (7): Published NusaX sentiment results (macro-F1 %, test split), Winata et al. EACL 2, compare(), main(), _markdown(), F1 gate report: our B1 replication vs NusaX Table 3 (docs/metodologi.md §5, K20,, Rows (lang, clf, variant, f1, feature) for the three variants., variants()

### Community 8 - "Community 8"
Cohesion: 0.40
Nodes (4): Data (pinned), jevexp — experiments for journal 010, Layout, Setup (reproducible)

### Community 10 - "Community 10"
Cohesion: 0.47
Nodes (5): ece(), Expected calibration error of the top-class confidence (Naeini 2015; adaptive =, test_ece_hand_computed(), test_ece_zero_for_perfectly_calibrated_one_hot(), test_reliability_bins_right_closed_edges()

### Community 11 - "Community 11"
Cohesion: 0.67
Nodes (3): brier(), Multiclass Brier score: mean over items of sum_k (p_k - 1[y=k])^2., test_brier()

## Knowledge Gaps
- **3 isolated node(s):** `Setup (reproducible)`, `Layout`, `Data (pinned)`
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `load()` connect `Community 1` to `Community 3`?**
  _High betweenness centrality (0.070) - this node is a cross-community bridge._
- **Why does `validate()` connect `Community 0` to `Community 3`?**
  _High betweenness centrality (0.047) - this node is a cross-community bridge._
- **Why does `ece()` connect `Community 10` to `Community 2`?**
  _High betweenness centrality (0.041) - this node is a cross-community bridge._
- **What connects `Experiment harness for journal 010 (see docs/metodologi.md in the journal folder`, `Pinned download, uniform loading and fixed subsets for the three datasets.  Data`, `Return (url, local relative path) for one CSV at the pinned commit.` to the rest of the system?**
  _28 weakly-connected nodes found - possible documentation gaps or missing edges._