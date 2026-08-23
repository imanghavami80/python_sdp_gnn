# Model Improvement Strategy

## Implemented

The final nested LOPO pipeline now addresses two high-impact CPDP problems:

1. **Project-size imbalance:** each outer-training project contributes equal
   total NDG loss, while positive-class weighting still handles label imbalance.
2. **Cross-project calibration shift:** each fold selects its classification
   threshold only from the inner-validation project.

Both changes preserve the outer-test boundary.

The proposal's late fusion and implementation's early fusion are both exposed
through `--fusion-stage`, making the design choice a controlled ablation.

Training-only k-means++ and GMM features are implemented for the proposal's
second contribution. After direct concatenation reduced F1, clustering was
moved into a separate gated branch and enriched with cross-fitted, smoothed
defect risk. `--cluster-method` makes the geometry algorithm a controlled
ablation; `--no-cluster-features` remains the baseline. The outer-test boundary
is preserved for every variant.

## Evaluation Priority

Use metrics in this order:

1. Macro-project PR-AUC and ROC-AUC for ranking.
2. Macro-project MCC, balanced accuracy, and F1 for classification.
3. Brier score for probability calibration.
4. Pooled metrics only as secondary results.

Report mean, standard deviation, median, per-project values, and results over
multiple random seeds.

## Next Controlled Experiments

Change one factor at a time:

1. Metrics-only baseline.
2. Metrics + AST.
3. Metrics + CFG.
4. Metrics + AST + CFG without NDG message passing.
5. Full relational NDG model.
6. Full late-fusion model with versus without cluster-derived metric features.
7. Gated k-means++ versus gated GMM with every non-clustering setting fixed.

This ablation establishes which view actually improves cross-project
generalization. After that, prioritize extraction coverage and relation-quality
improvements over adding unrelated feature families.

## Multiple Seeds

Use separate output directories:

```bash
for seed in 42 43 44 45 46; do
  python scripts/evaluate_ndg_nested_lopo.py \
    --seed "$seed" \
    --output-dir "outputs/promise/final_ndg_seed_${seed}" \
    --device cpu
done
```

Never select the best seed. Aggregate all seeds.

## Training Time Is Not a Quality Metric

A 15-minute run is not evidence of under-training. Hardware, graph batching,
and model size determine wall-clock time. Diagnose learning from training and
validation curves, selected epochs, multiple-seed variance, and held-out
metrics. Increase epochs or capacity only when those measurements show
underfitting; longer training can otherwise overfit source projects.
