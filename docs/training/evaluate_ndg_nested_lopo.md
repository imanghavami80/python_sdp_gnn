# 13 — Strict Nested NDG LOPO Evaluation

**Implementation:** `scripts/evaluate_ndg_nested_lopo.py`

## Purpose

This is the final research evaluation entry point. It trains AST, CFG, and NDG
models inside a strict nested Leave-One-Project-Out protocol and produces one
defect probability and final contextual embedding per held-out file node.

Precomputed global or standalone LOPO embeddings are not consumed.

AST inputs must use the complete 20-column syntax schema. NDG inputs must use
scope-aware resolution. Input hashes and representation descriptions are recorded
in the manifest and summary; alternative extractor implementations are not supported. See
[representation improvements](../features/representation_improvements.md).

Late fusion and CFG are fixed. Choose `--cluster-mode none|simple|gated`
(default none), and independently enable `--ndg-structural-features` (default
off). See [00 — README](../../README.md) for all six scenario commands.

## Inputs

The script requires completed AST, CFG, and NDG extraction:

```text
outputs/promise/ast/graph_index.csv
outputs/promise/cfg/graph_index.csv
outputs/promise/ndg/graph_index.csv
their tensor directories and vocabulary JSON files
```

Structural index and feature names are additionally required only when that
branch is enabled. CFG indexes must carry `construction=exceptional_control_flow`
and use exactly the canonical control-flow vocabulary. Incompatible inputs are rejected.

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
8. Fit metric scaling, optional clustering and risk, and optional structural
   scaling on inner-fit nodes; transform inner validation without refitting.
9. Train and select the NDG model, including its structural gate, using the
   same inner project boundary.
10. Select the F1 decision threshold on that inner-validation project.
11. Refit transformations and any selected clusterer on all outer-training nodes;
    cross-fit cluster risk by training project.
12. Retrain a fresh NDG on all outer-training projects with equal total loss
    contribution from every project.
13. Transform, predict, and encode every file node in the outer test project.

This repeats until every project has been tested once, unless `--test-project`
restricts the run.

## Leakage Controls

The outer test project is excluded from:

- AST training and epoch selection.
- CFG training and epoch selection.
- NDG training and epoch selection.
- Median imputation and standard scaling.
- NDG structural-feature normalization.
- Cluster-count selection, centroids, and cluster-feature normalization.
- Cluster defect-risk estimation.
- Class weighting and majority-baseline selection.
- Decision-threshold selection.

The threshold is selected independently inside every outer fold. The outer test
project never influences it. A threshold from the selected inner model is
transferred to a freshly retrained final model; probability-scale mismatch is
still possible. This cleanup does not change or solve that calibration issue.

## Missing-View Policy

- Metric view: always available.
- AST fallback: masked by default; `--include-ast-fallbacks` is an explicit
  ablation option.
- CFG placeholder: always masked as unavailable.

The view gate normalizes weights over only the representations available for a
file.

## Run

Full twelve-project evaluation:

```bash
python scripts/evaluate_ndg_nested_lopo.py --device cpu
```

One-fold pilot:

```bash
python scripts/evaluate_ndg_nested_lopo.py \
  --test-project log4j-1.2 \
  --output-dir outputs/promise/pilot_log4j \
  --device cpu
```

Main controls:

```text
--upstream-epochs --ndg-epochs --patience --min-delta
--ast-batch-size --cfg-batch-size --lr --weight-decay
--hidden-dim --embedding-dim --ast-layers --cfg-layers
--ndg-layers --heads --dropout --attention-dropout
--ndg-structural-features --cluster-mode --cluster-method --cluster-count
--cluster-min --cluster-max --cluster-silhouette-sample-size --cluster-n-init
--gmm-n-init --gmm-covariance-type --gmm-reg-covar
--hdbscan-min-cluster-sizes --hdbscan-min-samples --cluster-risk-smoothing
--seed --device --test-project
```

The default output is
`outputs/promise/experiments/<scenario>/seed_<seed>/`, where the scenario
includes cluster mode, active algorithm, and structural-feature on/off.
Nonempty output directories are rejected. Use `--output-dir` to separate pilots
or different hyperparameters with the same scenario and seed. The manifest
records arguments and input-index hashes; summaries and checkpoints identify
the mode, fixed fusion, and CFG construction contract.

## Runtime and Progress

This is intentionally expensive: every outer fold independently trains AST,
CFG, and NDG models and generates fold-specific embeddings. Reusing global AST
or CFG embeddings would be faster but would violate the test-project boundary.

Stage messages and periodic `ast_encoding` / `cfg_encoding` batch counters
distinguish long encoding passes from a stalled process. A pause immediately
after AST `early_stopping` normally means AST selection embeddings are being
generated.

On macOS:

- Start with the one-fold CPU pilot.
- Use MPS only if its batch counters advance faster than CPU.
- Keep `--num-workers 0` as the safe default.
- If no batch counter advances for an extended period, use `Ctrl+C` and restart
  with `--device cpu` and a separate output directory.

## Outputs

Combined outputs include `run_manifest.json` plus:

```text
outputs/promise/experiments/<scenario>/seed_<seed>/ndg_node_embeddings.npy
outputs/promise/experiments/<scenario>/seed_<seed>/ndg_node_embedding_index.csv
outputs/promise/experiments/<scenario>/seed_<seed>/all_test_node_predictions.csv
outputs/promise/experiments/<scenario>/seed_<seed>/fold_metrics.csv
outputs/promise/experiments/<scenario>/seed_<seed>/nested_lopo_summary.json
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

Enabled clustering additionally writes `folds/<project>/cluster_features.json`.
CSV gate diagnostics are missing for absent gates (including the cluster gate
in simple mode). No learned cluster-gate effect is implied in simple mode.

## Metrics

The summary contains pooled node metrics and unweighted macro-project mean,
standard deviation, and median for accuracy, balanced accuracy, precision,
recall, F1, MCC, G-Mean, ROC-AUC, PR-AUC, and Brier score. It also reports a
majority-class baseline fitted from each fold's outer-training labels.

Use macro-project results as the primary CPDP result. Pooled scores are
secondary because projects contain very different numbers of files.

## Expected Result

- Every evaluated file appears once, in the fold where its project was held out.
- The final embedding matrix has one row per evaluated file node.
- Fold split files prove project membership at each training boundary.
- Every prediction records its fold-specific decision threshold.
- Every prediction records its learned NDG structural-gate value.
- The embedding index is the authoritative mapping from matrix rows to files.
