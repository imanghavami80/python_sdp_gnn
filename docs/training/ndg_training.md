# Shared NDG Training Utilities

**Implementation:** `src/thesis_project/training/ndg.py`

## Purpose

This module contains reusable data and optimization operations for node-level
NDG experiments. Keeping them in the package avoids importing one command-line
script from another and provides a single implementation of leakage-sensitive
metric transformation.

## `ProjectGraph`

Represents one project NDG with:

```text
file names and source paths
metric, AST, and CFG node features
view-availability mask
per-node training loss weight
binary node labels
typed edge index
```

`to(device)` returns a graph with tensors moved to the selected PyTorch device
while preserving node metadata.

## Reverse Relations

`add_inverse_relations` creates a reverse edge for every extracted dependency
and offsets its relation ID by the number of forward relations. Extracted graph
files remain unchanged. The model can therefore distinguish both relation type
and message direction.

## Combining Projects

`combine_graphs` joins multiple project NDGs as disconnected components. Node
offsets are applied to edges, but no artificial cross-project edge is created.
Each project receives equal total loss weight, so a large project cannot
dominate optimization merely because it contains more files.

## Leakage-Safe Metric Transformation

`standardize_metrics` receives the training graph first. It:

1. Fits per-feature medians on training nodes.
2. Imputes missing training values.
3. Fits training mean and population standard deviation.
4. Applies those fixed parameters to validation and test graphs.

No validation or test value influences fitted transformations.

## Cluster-Derived Metric Features

After leakage-safe standardization, `training/clustering.py` creates a separate
k-means++, GMM, or HDBSCAN representation containing component/density features, soft
memberships, outlier evidence, and smoothed cluster defect risk. It does not modify the 20
original metrics. The clusterer is fitted only on the graph passed as training,
with equal total fitting weight per project, and transforms validation/test
graphs without refitting. Training-node
risk is leave-one-project-out encoded. See [Cluster-Based NDG
Features](../features/cluster_features.md).

## Optimization

- `loss_function` computes positive-class weighting using project-balanced
  training weights.
- `weighted_loss` combines class weighting with project-balanced node weights.
- `train_with_validation` selects an epoch using validation loss and patience.
- `select_f1_threshold` selects a decision threshold from the inner-validation
  project after restoring the best epoch.
- `retrain` fits a fresh model on all outer-training nodes for the selected
  number of epochs.
- Gradient norms are clipped to improve training stability.

## Evaluation

`evaluate` returns one embedding, probability, and label per file node.
`binary_metrics` computes accuracy, balanced accuracy, precision, recall, F1,
MCC, G-Mean, ROC-AUC, PR-AUC, and Brier score. Threshold-dependent metrics use the
inner-validation-selected threshold. Ranking metrics are omitted when the
evaluated labels contain only one class.

## Correct Usage

- Pass only outer-training graphs to fitting functions.
- Apply `standardize_metrics` separately for each outer fold.
- Fit cluster geometry and risk after standardization and only on the relevant training graph.
- Cross-fit risk for training projects; never encode a project from its own labels.
- Never combine a held-out project into the disconnected training graph.
- Preserve node ordering when writing predictions and embedding indexes.
