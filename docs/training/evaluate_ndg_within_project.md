# Within-project evaluation

`scripts/evaluate_ndg_within_project.py` trains one four-view model per project.
It uses only files from that project. Source-file groups are split approximately
60% training, 20% validation, and 20% test, with both defect classes required
in each partition. The same split is used for AST, CFG, and NDG. Ineligible
projects appear in `project_eligibility.csv` with a reason.

## Training flow

1. Train AST and CFG encoders using training files; choose their checkpoints by
   validation loss. Missing AST/CFG graphs are masked.
2. Derive the NDG node input from directed, typed dependencies. Scale it and
   the dataset metrics using training nodes only. Metrics are a separate view.
3. Train the NDG encoder and gated four-view fusion on the full unlabeled graph,
   using **only training labels** in the loss. Test labels are hidden from the
   training graph.
4. Choose the NDG checkpoint by validation loss and its prediction threshold
   by validation F1. Evaluate that checkpoint once on test files.

This is **transductive** evaluation: test-file features and dependency edges are
visible, but their labels are not. It estimates prediction for known files with
withheld labels, not future releases. The AST/CFG checkpoints are frozen during
NDG and fusion training.

## Run from scratch

First run the four extraction commands in the [README](../../README.md), then:

```bash
.venv/bin/python scripts/evaluate_ndg_within_project.py \
  --no-ndg-structural-features --device cpu --seed 43 \
  --output-dir outputs/promise/within_project/four_view_reproduction_seed43
```

Use a fresh output directory for each run. `--project ant-1.7` selects one
project. `--preflight-only` checks eligibility without training, also using a
fresh directory. Optional handcrafted NDG features can be tested separately
with `--ndg-structural-features` after running
`scripts/extract_ndg_structural_features.py`.

## Read the results

- `within_project_summary.json` contains **macro averages** under
  `model_macro_project_metrics` and `majority_class_macro_project_metrics`. Every
  completed project has equal weight. It does not contain pooled metrics.
- `project_metrics.csv` contains one test-score row per project.
- `all_test_node_predictions.csv` contains held-out file predictions. It is
  saved for audit, not for computing a pooled headline result.
- `projects/<project>/file_view_embeddings.npz` contains four aligned,
  projected file embeddings, availability masks, and learned fusion weights.
  `test_node_embeddings.npy` contains the final fused test embeddings.
- Project folders also hold splits, AST/CFG/NDG checkpoints, validation
  histories, and the exact preprocessing parameters needed to replay a model.
- `run_manifest.json` records arguments, environment, seeds, and input hashes.

The `majority_class_macro_project_metrics` section means a simple
**training-majority-class predictor**. The already completed reference run
uses the older field name `baseline_macro_project_metrics` for this same dummy
predictor. The reference four-view neural model is
the completed `four_view_scratch_seed43` run, documented in the
[README](../../README.md#current-reference-baseline). Compare a new model's
`model_macro_project_metrics` with that run's model metrics. Keep the project
set, file splits, seed, and source inputs fixed. A single seed does not
establish robustness.
