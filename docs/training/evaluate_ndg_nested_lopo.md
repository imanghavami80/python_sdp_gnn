# Strict Nested NDG LOPO Evaluation

**Implementation:** `scripts/evaluate_ndg_nested_lopo.py`

## Purpose

This is the final research evaluation entry point. It trains AST, CFG, and NDG
models inside a strict nested Leave-One-Project-Out protocol and produces one
defect probability and final contextual embedding per held-out file node.

Precomputed global or standalone LOPO embeddings are not consumed.

## Inputs

The script requires completed AST, CFG, and NDG extraction:

```text
outputs/promise/ast/graph_index.csv
outputs/promise/cfg/graph_index.csv
outputs/promise/ndg/graph_index.csv
their tensor directories and vocabulary JSON files
```

Keys are joined by `(dataset_name, class name)`. Missing reliable AST or CFG
views are represented through a mask, not by deleting the NDG node.

## Outer Fold

For each test project:

1. Remove the complete project from all training stages.
2. Select one class-adequate outer-training project for inner validation.
3. Use the remaining projects as the inner fit set.
4. Train AST and CFG selection models on inner-fit files.
5. Choose upstream epochs from the inner validation project.
6. Retrain fresh AST and CFG models on all outer-training projects.
7. Generate embeddings for outer-training files and the untouched test project.
8. Train and select the NDG model using the same inner project boundary.
9. Retrain a fresh NDG on all outer-training projects.
10. Predict and encode every file node in the outer test project.

This repeats until every project has been tested once, unless `--test-project`
restricts the run.

## Leakage Controls

The outer test project is excluded from:

- AST training and epoch selection.
- CFG training and epoch selection.
- NDG training and epoch selection.
- Median imputation and standard scaling.
- Class weighting and majority-baseline selection.

The classification threshold is fixed at `0.5`; it is not tuned on the test
project.

## Missing-View Policy

- Metric view: always available.
- AST fallback: masked by default; `--include-ast-fallbacks` is an explicit
  ablation option.
- CFG placeholder: always masked as unavailable.

The view gate normalizes weights over only the representations available for a
file.

## Run

Full ten-project evaluation:

```bash
python scripts/evaluate_ndg_nested_lopo.py --device cpu
```

One-fold pilot:

```bash
python scripts/evaluate_ndg_nested_lopo.py \
  --test-project log4j-1.1 \
  --device cpu
```

Main controls:

```text
--upstream-epochs --ndg-epochs --patience --min-delta
--ast-batch-size --cfg-batch-size --lr --weight-decay
--hidden-dim --embedding-dim --ast-layers --cfg-layers
--ndg-layers --heads --dropout --attention-dropout
--seed --device --test-project
```

The output directory is cleaned at startup. Safety checks reject broad targets
such as the repository root, filesystem root, or home directory.

## Outputs

Combined outputs:

```text
outputs/promise/final_ndg_nested_lopo/ndg_node_embeddings.npy
outputs/promise/final_ndg_nested_lopo/ndg_node_embedding_index.csv
outputs/promise/final_ndg_nested_lopo/all_test_node_predictions.csv
outputs/promise/final_ndg_nested_lopo/fold_metrics.csv
outputs/promise/final_ndg_nested_lopo/nested_lopo_summary.json
```

Per-fold outputs:

```text
folds/<test-project>/split.json
folds/<test-project>/ast_encoder.pt
folds/<test-project>/cfg_encoder.pt
folds/<test-project>/ndg_encoder.pt
folds/<test-project>/ast_selection_history.csv
folds/<test-project>/cfg_selection_history.csv
folds/<test-project>/ndg_selection_history.csv
folds/<test-project>/test_node_embeddings.npy
folds/<test-project>/test_node_predictions.csv
```

## Metrics

The summary contains pooled node metrics and unweighted macro-project mean,
standard deviation, and median for accuracy, precision, recall, F1, ROC-AUC,
and PR-AUC. It also reports a majority-class baseline fitted from each fold's
outer-training labels.

## Expected Result

- Every evaluated file appears once, in the fold where its project was held out.
- The final embedding matrix has one row per evaluated file node.
- Fold split files prove project membership at each training boundary.
- The embedding index is the authoritative mapping from matrix rows to files.

