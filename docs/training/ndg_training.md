# Training utilities

`src/thesis_project/training/within_project.py` creates grouped splits, fits
training-only scaling, and trains on labeled training nodes while the full
dependency graph is visible. Validation loss selects a checkpoint; validation
F1 selects its decision threshold.

`src/thesis_project/training/ndg.py` holds the four-view graph tensors, binary
metrics, and model-input helpers. The complete experiment is run by
`scripts/evaluate_ndg_within_project.py`; see the
[evaluation guide](evaluate_ndg_within_project.md).
