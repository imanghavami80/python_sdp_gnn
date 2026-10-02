# 18 — Within-Project Model Evaluation Strategy

Start with the [within-project baseline](../training/evaluate_ndg_within_project.md).
Use identical source-file splits, seeds and extraction artifacts when comparing
optional structural features. Fit scalers on training
nodes only; select epochs and thresholds on validation nodes only. Preserve the
full unlabeled graph and clearly describe the experiment as transductive.

Report per-project and macro-project PR-AUC, ROC-AUC, F1, MCC, balanced accuracy,
and Brier score. Report skipped projects and small test-class counts. Pooled
metrics are secondary. A single split is preliminary evidence; additional
matched seeds can establish whether differences persist when resources permit.
Do not select a favorable seed or use test results to tune a threshold.

Keep the selected checkpoint for test evaluation. Do not retrain it after
selecting the threshold. Compare only matched within-project runs.
