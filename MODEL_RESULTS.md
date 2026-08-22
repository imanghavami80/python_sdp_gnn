# Model Results

## Current Status

The results below are **local thesis-model experiments only**. No comparison
with a published paper or external model has been performed yet.

The current pipeline uses `projects_new`:

- 12 PROMISE CSVs and 5,430 rows;
- 5,303 rows mapped to Java source files;
- 5,303 AST graphs, including 11 tagged fallbacks;
- 5,303 CFG graphs: 4,945 Soot graphs and 358 tagged placeholders;
- 12 NDGs with 5,303 file nodes and 36,986 typed dependency edges.

These are extraction counts, not prediction results.

## Completed Local Experiments

Early fusion and late fusion were evaluated with strict nested
Leave-One-Project-Out (LOPO). Each of the 12 projects was held out once, giving
12 test folds and predictions for all 5,303 mapped files. The held-out project
was excluded from AST, CFG, and NDG training, metric normalization, epoch
selection, and decision-threshold selection.

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

## Next Experiment

Add the proposed cluster-based features without using defect labels or held-out
project data during clustering. Then compare late fusion with and without those
features under the same local protocol. Multiple seeds should be run before
treating the final result as stable.

## Saved Result Files

- Early fusion: `outputs/promise/nested_lopo_early/`
- Late fusion: `outputs/promise/nested_lopo_late/`

Each directory contains `nested_lopo_summary.json`, `fold_metrics.csv`, and
`all_test_node_predictions.csv`.

Wall-clock duration is not evidence of sufficient learning. Use learning
curves, selected epochs, variation across trials/seeds, and held-out metrics.
