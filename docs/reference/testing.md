# 17 — Testing and Validation

Run `.venv/bin/python -m pytest -q` from the repository root.

The suite covers preprocessing, Java source mapping, AST/CFG extraction and
encoding, typed NDG dependencies, optional structural descriptors, masked late
fusion, source-file grouping, training-only scaling, and held-out-label isolation.
Within-project integration tests verify checkpoint prediction reproduction,
validation-controlled training, and identical predictions after test-label changes.

For a short real-data pilot:

```bash
.venv/bin/python scripts/evaluate_ndg_within_project.py \
  --project log4j-1.2 --upstream-epochs 1 --ndg-epochs 1 \
  --hidden-dim 32 --embedding-dim 32 --device cpu \
  --output-dir outputs/promise/within_project/verification_pilot
```

Use a fresh output directory. Check split membership, class counts, finite
probabilities and saved checkpoints. Pilot performance is not evidence of model
quality. See [the evaluation guide](../training/evaluate_ndg_within_project.md).
