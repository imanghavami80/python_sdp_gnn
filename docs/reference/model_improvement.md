# 18 — Model Improvement Strategy

The active model always uses AST, CFG v3, relational NDG message passing, and
late fusion. The supported scenario factors are cluster mode (none/simple/gated)
and handcrafted NDG structural features (off/on).

## Controlled evaluation

Start with the baseline: neither clustering nor handcrafted NDG features.
Then compare simple and gated clustering using the same clustering algorithm
(default k-means++), feature construction, graph artifacts, and seeds. Repeat
the three modes with NDG structural features enabled. These six scenarios
separate cluster integration from topology-feature augmentation.

Use macro-project PR-AUC and ROC-AUC for ranking, MCC/balanced accuracy/F1 for
classification, and Brier score for probabilities. Report per-project scores
and variation across seeds. Pooled metrics are secondary because project sizes
and label distributions differ substantially.

The current inner-validation threshold is transferred to a freshly retrained
model. This preserves the outer-test boundary but does not guarantee calibrated
probabilities or a well-matched final threshold. Diagnose score distributions,
predicted-positive rates, and calibration before attributing every metric change
to a representation. Do not tune thresholds on test labels.

## Seeds and provenance

Run matched seeds for each scenario:

```bash
for seed in 42 43 44 45 46; do
  python scripts/evaluate_ndg_nested_lopo.py --seed "$seed" --device cpu
done
```

Automatic output names include scenario and seed; nonempty directories are
rejected. Never select the best seed. Compare manifest arguments, input hashes,
CFG v3 markers, and completed fold counts before aggregating results.

Historical reports document earlier decisions; they are not results from the
cleaned architecture. New experiments are required after regenerating inputs.

Training time alone does not indicate learning quality. Use selection histories,
held-out results, and variance to decide whether more training is useful.
