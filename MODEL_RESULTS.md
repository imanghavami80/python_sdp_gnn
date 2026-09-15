# 19 — Model Results and Historical Record

## Status after cleanup

An earlier scenario cleanup deleted the original generated artifacts. The
measurements below are historical and are not results of the current AST and NDG
inputs. Inputs have now been regenerated at the default paths, while existing
experiment directories were preserved. A short integration pilot does not replace
a full predictive comparison. See [representation improvements](docs/features/representation_improvements.md).

Every new run uses CFG and late fusion. Compare the six combinations of
cluster mode (none/simple/gated) and NDG structural features (off/on), using
the rebuild and run commands in [00 — README](README.md).

## Historical extraction

The previously measured benchmark had 12 projects, 5,430 input rows, 5,303
mapped files, and 12 NDGs. CFG yielded 4,573 valid CFGs and 730 placeholders;
Log4j had 178 valid CFGs out of 194 files. These counts must be verified again
from regenerated extraction summaries.

## Historical predictive comparison

Unweighted macro-project means, 12 held-out projects, seed 42:

| Metric | Older no-cluster run | CFG + gated k-means | CFG + NDG structural |
| --- | ---: | ---: | ---: |
| Accuracy | 0.5271 | 0.5428 | 0.5127 |
| Balanced accuracy | 0.5567 | 0.5326 | 0.5646 |
| Precision | 0.4805 | 0.4464 | 0.4057 |
| Recall | 0.8218 | 0.8480 | 0.6301 |
| F1 | 0.5156 | 0.5183 | 0.3994 |
| MCC | 0.0973 | 0.0785 | 0.1378 |
| G-Mean | 0.3361 | 0.2307 | 0.3877 |
| ROC-AUC | 0.6810 | 0.6936 | 0.6726 |
| PR-AUC | 0.5677 | 0.5777 | 0.5706 |
| Brier score (lower is better) | 0.2804 | 0.2465 | 0.3078 |

The older no-cluster run is not a verified current CFG baseline. The NDG
structural run disabled clustering. Its balanced accuracy, MCC, and G-Mean
increased relative to the gated-cluster run, while several other metrics
declined. It predicted no defective files in Camel and Xalan; threshold and
probability behavior need investigation. These results alone cannot establish
the causal contribution of structural features.

The current simple mode uses the same feature vector as gated mode and is not
equivalent to the historical five-feature direct-clustering implementation.
Removing unused modules also changes random initialization, so new matched
runs are required even for nominally similar scenarios.

See [20 — Short CFG history](docs/reports/cfg_view_progress_report_short.md)
and [21 — Detailed CFG history](docs/reports/cfg_view_progress_report.md) for
the recorded original-CFG, def-use, PDG, CFG before bytecode recovery, and CFG comparisons.
Those alternatives are historical only; CFG is the sole active extractor.
