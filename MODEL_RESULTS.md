# 19 — Within-Project Results

The active runner is `scripts/evaluate_ndg_within_project.py`. Each project gets
a separate model and a grouped 60/20/20 train/validation/test split. The full
unlabeled NDG is visible; only training labels contribute to the loss.

The full `baseline_seed42` experiment completed ten projects and evaluated 873
held-out files. Forrest (30 clean / 2 defective) and Xalan (1 clean / 898 defective)
cannot support both classes in three disjoint splits. Their preprocessing is not
the reason for exclusion.

| Metric | Equal-project mean | Pooled test files |
| --- | ---: | ---: |
| F1 | 0.6407 | 0.7257 |
| ROC-AUC | 0.8201 | 0.8042 |
| Average precision (PR) | 0.6848 | 0.7675 |
| Balanced accuracy | 0.6693 | 0.7754 |
| MCC | 0.3423 | 0.5409 |

The training-majority classifier's mean project F1 is 0.3368 and mean balanced
accuracy is 0.5. These results show useful signal, but only one split/seed was
evaluated. Log4j's high F1 (0.9737) comes from predicting every test file defective;
its test set has only two clean files. High F1 alone is not proof of robustness.

Saved histories verify actual training: AST ran 12–50 epochs, CFG 14–50 and NDG
11–24 per project. The original run did not record elapsed times. The updated
runner adds timing and provenance without changing the split or epoch budgets;
the completed baseline artifacts remain unchanged.

## Full experiment

The completed run used:

```bash
.venv/bin/python scripts/evaluate_ndg_within_project.py \
  --no-ndg-structural-features --device cpu --seed 42 \
  --output-dir outputs/promise/within_project/baseline_seed42
```

Read `within_project_summary.json`, `project_metrics.csv`, and
`all_test_node_predictions.csv` in that directory. Report macro-project and
per-project metrics, test class counts, and skipped projects. The majority-class
baseline in the summary uses training prevalence; it is not another neural model.
Compare optional structural features using the same seed and file split.
For new runs, use a fresh output directory such as `baseline_audited_seed42`.
Use `--preflight-only` to inspect all projects without training.

See [the evaluation guide](docs/training/evaluate_ndg_within_project.md).
