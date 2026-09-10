# Model Results

## Current Status

The results below are **local thesis-model experiments only**. No comparison
with a published paper or external model has been performed yet.

The current pipeline uses `projects_new`:

- 12 PROMISE CSVs and 5,430 rows;
- 5,303 rows mapped to Java source files;
- 5,303 AST graphs, including 11 tagged fallbacks;
- 5,303 CFG graphs: 4,573 validated Soot graphs and 730 tagged placeholders,
  containing 1,753,339 typed normal and exceptional control-flow edges;
- 12 NDGs with 5,303 file nodes and 36,986 typed dependency edges.

These are extraction counts, not prediction results. The stricter current
coverage excludes compiler-generated error bodies that older extraction runs
incorrectly treated as real methods.

Exact, checksum-verified Apache release bytecode now supplements incomplete
source builds for Camel 1.6 and Synapse 1.2. This raised real-CFG coverage from
4,046/5,303 (76.3%) to 4,573/5,303 (86.2%): Camel rose from 384 to 804 real
graphs and Synapse from 121 to 228. The remaining Xerces placeholders are
interfaces without executable method bodies, so fabricating CFGs for them would
not be a valid coverage improvement.

## Completed Local Experiments

Early fusion and late fusion were evaluated with strict nested
Leave-One-Project-Out (LOPO). Each of the 12 projects was held out once, giving
12 test folds and predictions for all 5,303 mapped files. The held-out project
was excluded from AST, CFG, and NDG training, metric normalization, epoch
selection, and decision-threshold selection.

The completed results below use earlier CFG artifacts. The canonical
exception-aware CFG was regenerated with higher Camel/Synapse coverage after
these runs, so these values remain historical baselines and must not be
presented as results from the improved extractor. A new twelve-fold run is
required for that comparison.

The table reports the unweighted mean across the 12 held-out projects:

| Metric | Early fusion | Late fusion | Better local result |
| --- | ---: | ---: | --- |
| Accuracy | 0.4573 | 0.5271 | Late |
| Balanced accuracy | 0.5390 | 0.5567 | Late |
| Precision | 0.4681 | 0.4805 | Late |
| Recall | 0.7023 | 0.8218 | Late |
| F1 | 0.4214 | 0.5156 | Late |
| MCC | 0.1007 | 0.0973 | Early, by a small margin |
| G-Mean | 0.2688 | 0.3361 | Late |
| ROC-AUC | 0.7131 | 0.6810 | Early |
| PR-AUC | 0.5722 | 0.5677 | Early, by a small margin |
| Brier score | 0.3277 | 0.2804 | Late (lower is better) |

Late fusion produced better threshold-based classification results in this
run, especially recall and F1. Early fusion produced better ranking results.
Therefore, late fusion is the current working choice, but this single-seed
experiment does not prove that it is universally better.

## Cluster Feature Status

The first k-means++ experiment directly concatenated five unsupervised cluster
features into the metric view. Its macro accuracy increased from 0.5271 to
0.5788, but recall fell from 0.8218 to 0.5485 and F1 fell from 0.5156 to 0.4077.
F1 was worse on 10 of 12 projects. This direct-concatenation design is therefore
retained only as a negative ablation in
`outputs/promise/nested_lopo_late_clusters/`; it is not the current model.

The revised gated k-means++ implementation keeps the 20 metrics unchanged and
sends cluster geometry plus cross-fitted defect risk through a separate branch.
It recovered macro F1 to 0.5160, versus 0.5156 without clustering, and produced
the best single-seed MCC (0.1387), G-Mean (0.3818), PR-AUC (0.5838), and Brier
score (0.2602) of the three late-fusion runs. Its F1 improvement over the
no-cluster baseline was effectively zero and was not statistically significant,
so it is promising rather than a confirmed improvement.

A BIC-selected GMM backend is available through `--cluster-method gmm`. It uses
the same gate, cross-fitted risk, and leakage boundary and must be written to a
separate result directory. Multiple seeds are still required after selecting
the stronger clustering backend.

The density-based HDBSCAN backend is available through `--cluster-method
hdbscan`. It uses a fixed six-feature density representation so its
data-dependent cluster count cannot change the selected NDG architecture. Its
result directory must also remain separate for a controlled comparison.

## Saved Result Files

- Early fusion: `outputs/promise/nested_lopo_early/`
- Late fusion: `outputs/promise/nested_lopo_late/`
- Direct cluster concatenation: `outputs/promise/nested_lopo_late_clusters/`
- Gated k-means++: `outputs/promise/nested_lopo_late_cluster_gate_kmeans/`
- Gated GMM: `outputs/promise/nested_lopo_late_cluster_gate_gmm/`
- Gated HDBSCAN: `outputs/promise/nested_lopo_late_cluster_gate_hdbscan/`

Each directory contains `nested_lopo_summary.json`, `fold_metrics.csv`, and
`all_test_node_predictions.csv`.

Wall-clock duration is not evidence of sufficient learning. Use learning
curves, selected epochs, variation across trials/seeds, and held-out metrics.
