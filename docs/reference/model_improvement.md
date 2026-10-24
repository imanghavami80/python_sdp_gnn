# Improving the four-view baseline

The [reference run](../../README.md#current-reference-baseline) is the four-view
model at seed 43. Its primary result is the equal-project macro mean, supported
by the per-project results in `project_metrics.csv`. The majority-class scores
in its summary are only a simple sanity check.

Change one modeling choice at a time and save each candidate in a new output
directory. Compare on the same 12 projects, source-file splits, extracted
inputs, and seed. Use the saved `split.csv` files and `run_manifest.json` to
check the match. Report macro F1, MCC, balanced accuracy, ROC AUC, and PR AUC,
as well as per-project changes. Forrest's six-file test set can move the macro
mean substantially, so inspect that project separately.

Training-only scaling and validation-only checkpoint/threshold selection must
remain in place. Test labels must not influence model selection. Once a change
looks useful on the matched run, evaluate it over additional matched seeds.
