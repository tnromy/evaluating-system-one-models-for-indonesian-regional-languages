# Graph Report - jev  (2026-09-24)

## Corpus Check
- 9 files · ~2,578 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 70 nodes · 109 edges · 13 communities (11 shown, 2 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `6593c53d`
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
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]

## God Nodes (most connected - your core abstractions)
1. `load()` - 8 edges
2. `fetch()` - 6 edges
3. `reliability_bins()` - 6 edges
4. `ece()` - 6 edges
5. `validate()` - 6 edges
6. `parallel_report()` - 5 edges
7. `_sources()` - 4 edges
8. `make_subsets()` - 4 edges
9. `summary()` - 4 edges
10. `main()` - 4 edges

## Surprising Connections (you probably didn't know these)
- `test_subsets_fixed_balanced_and_parallel()` --calls--> `load()`  [EXTRACTED]
  tests/test_data.py → src/jevexp/data.py
- `test_ece_hand_computed()` --calls--> `ece()`  [EXTRACTED]
  tests/test_eval.py → src/jevexp/eval.py
- `test_ece_zero_for_perfectly_calibrated_one_hot()` --calls--> `ece()`  [EXTRACTED]
  tests/test_eval.py → src/jevexp/eval.py
- `test_reliability_bins_right_closed_edges()` --calls--> `ece()`  [EXTRACTED]
  tests/test_eval.py → src/jevexp/eval.py
- `test_nusatranslation_parallel_core()` --calls--> `load()`  [EXTRACTED]
  tests/test_data.py → src/jevexp/data.py

## Import Cycles
- None detected.

## Communities (13 total, 2 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.35
Nodes (9): Per-item prediction schema shared by every arm (docs/metodologi.md §6).  One row, Raise ValueError on any schema violation; return df unchanged otherwise., read(), validate(), write(), _row(), test_rejects_missing_required_and_duplicates(), test_rejects_violations() (+1 more)

### Community 1 - "Community 1"
Cohesion: 0.36
Nodes (7): fetch(), Pinned download, uniform loading and fixed subsets for the three datasets.  Data, Return (url, local relative path) for one CSV at the pinned commit., Download every CSV at the pinned commit; verify against / write MANIFEST.json., _read_manifest(), _sha256(), _sources()

### Community 2 - "Community 2"
Cohesion: 0.38
Nodes (6): _bin_edges(), Metrics shared by every arm (docs/metodologi.md §7-8).  Conventions:   y_true, y, Per bin: (count, mean confidence, accuracy). y_idx = gold class index into probs, reliability_bins(), _top(), test_reliability_bins_right_closed_edges()

### Community 3 - "Community 3"
Cohesion: 0.40
Nodes (6): load(), parallel_report(), Per split: ids shared by all languages, and ids whose gold label differs across, Uniform frame: dataset, lang, split, sentence_id, text, label (canonical lowerca, test_nusatranslation_parallel_core(), test_nusax_shape_and_parallelism()

### Community 4 - "Community 4"
Cohesion: 0.50
Nodes (5): accuracy(), macro_f1(), Macro-F1 over the task's labels only; INVALID predictions count as errors for ev, test_accuracy_and_macro_f1_match_sklearn(), test_invalid_counts_as_wrong()

### Community 5 - "Community 5"
Cohesion: 0.50
Nodes (5): aurc(), coverage_accuracy_curve(), Sort by confidence (desc); return coverage and accuracy of the kept prefix at ev, Area under the risk-coverage curve (Geifman et al. 2019): mean selective risk ov, test_coverage_curve_and_aurc()

### Community 6 - "Community 6"
Cohesion: 0.50
Nodes (4): cluster_bootstrap(), Percentile CI resampling whole clusters (all languages of a sentence move togeth, test_cluster_bootstrap_deterministic_and_brackets_point(), test_cluster_bootstrap_resamples_whole_clusters()

### Community 8 - "Community 8"
Cohesion: 0.40
Nodes (4): Data (pinned), jevexp — experiments for journal 010, Layout, Setup (reproducible)

### Community 9 - "Community 9"
Cohesion: 0.50
Nodes (4): main(), make_subsets(), Fixed test subsets (S-NX, S-NT, S-NE): proportional-stratified by label over sen, summary()

### Community 10 - "Community 10"
Cohesion: 0.50
Nodes (4): ece(), Expected calibration error of the top-class confidence (Naeini 2015; adaptive =, test_ece_hand_computed(), test_ece_zero_for_perfectly_calibrated_one_hot()

### Community 11 - "Community 11"
Cohesion: 0.67
Nodes (3): brier(), Multiclass Brier score: mean over items of sum_k (p_k - 1[y=k])^2., test_brier()

## Knowledge Gaps
- **3 isolated node(s):** `Setup (reproducible)`, `Layout`, `Data (pinned)`
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `load()` connect `Community 3` to `Community 1`, `Community 9`, `Community 7`?**
  _High betweenness centrality (0.059) - this node is a cross-community bridge._
- **Why does `ece()` connect `Community 10` to `Community 2`?**
  _High betweenness centrality (0.054) - this node is a cross-community bridge._
- **What connects `Experiment harness for journal 010 (see docs/metodologi.md in the journal folder`, `Pinned download, uniform loading and fixed subsets for the three datasets.  Data`, `Return (url, local relative path) for one CSV at the pinned commit.` to the rest of the system?**
  _21 weakly-connected nodes found - possible documentation gaps or missing edges._