# Standalone CFG LOPO Evaluation

**Implementation:** `scripts/evaluate_cfg_lopo.py`

## Purpose

This script evaluates the edge-aware CFG encoder with one project held out at a
time. It produces one file embedding and one standalone CFG defect prediction
for every file with a real Soot CFG.

## Input Filtering

The script loads `outputs/promise/cfg/graph_index.csv` and removes every row
whose `extraction_mode` is not `soot`. Placeholder `ENTRY -> EXIT` graphs are
never used for standalone training or evaluation.

## Protocol

For each project:

1. Select its real CFGs as the outer test set.
2. Use real CFGs from all other projects for model fitting.
3. Create a stratified file-level validation split inside the outer training
   set.
4. Train the edge-aware CFG classifier with early stopping.
5. Restore the best model and encode the held-out project.
6. Save fold predictions, metrics, checkpoint, and embeddings.
7. Combine all held-out embeddings and predictions after the final fold.

The outer project is excluded, but inner validation is file-level. The strict
final NDG evaluator instead uses an inner project and retrains the CFG encoder
inside every outer NDG fold.

## Run

```bash
python scripts/evaluate_cfg_lopo.py
```

Important arguments configure categorical embedding dimensions, hidden/output
dimensions, attention heads, layer count, learning rate, epochs, patience,
dropout, seed, and device.

## Outputs

```text
outputs/promise/embeddings/cfg_lopo/cfg_embeddings.npy
outputs/promise/embeddings/cfg_lopo/cfg_embedding_index.csv
outputs/promise/embeddings/cfg_lopo/cfg_embedding_summary.json
outputs/promise/embeddings/cfg_lopo/fold_metrics.csv
outputs/promise/embeddings/cfg_lopo/aggregate_metrics.json
outputs/promise/embeddings/cfg_lopo/all_test_predictions.csv
outputs/promise/embeddings/cfg_lopo/folds/<project>/cfg_encoder.pt
outputs/promise/embeddings/cfg_lopo/folds/<project>/training_history.csv
outputs/promise/embeddings/cfg_lopo/folds/<project>/test_embeddings.npy
outputs/promise/embeddings/cfg_lopo/folds/<project>/test_predictions.csv
outputs/promise/embeddings/cfg_lopo/folds/<project>/split.json
```

## Expected Result

- One embedding per non-placeholder file CFG.
- Every embedding is generated while its project is held out.
- Per-project and aggregate accuracy, precision, recall, F1, ROC-AUC, and
  PR-AUC are available for standalone CFG analysis.

## Appropriate Use

Use this workflow to evaluate CFG extraction and encoder quality. Do not use its
precomputed embeddings in final nested NDG results.

