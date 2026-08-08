# Testing and Validation

## Purpose

The test suite protects tensor contracts, model output dimensions, edge-aware
behavior, source mapping, missing-view handling, and leakage boundaries.

Run all tests from the repository root:

```bash
pytest
```

Compile-check all Python modules:

```bash
python -m compileall -q src scripts tests
```

## Test Files

### `tests/test_preprocess_promise.py`

Checks strict source mapping behavior, including rejection of packaged
simple-name fallback, ambiguous qualified names, and duplicate dataset classes.

### `tests/test_ast_encoder.py`

Checks AST normalization, encoder and classifier shapes, batching, and attention
pooling behavior.

### `tests/test_cfg_encoder.py`

Checks CFG input normalization, categorical and edge-type contracts, encoder and
classifier shapes, and attention outputs.

### `tests/test_ndg_encoder.py`

Checks multi-view masking, node-level output shape, reverse relations, outer
project leakage guards, validation-project suitability, and training-only metric
transformation.

## Validation Beyond Unit Tests

After changing extraction code:

1. Run preprocessing and all three extractors.
2. Compare `(dataset_name, name)` keys across AST, CFG, and NDG outputs.
3. Verify tensor shapes against every graph index row.
4. Inspect parse failures, fallbacks, placeholders, and validation issue files.
5. Run one short nested LOPO pilot before launching the full experiment.

After changing model or training code:

1. Run unit tests and compilation checks.
2. Run one outer fold with reduced epochs.
3. Verify one prediction and embedding per held-out NDG node.
4. Inspect `split.json` to confirm the outer test project is absent from every
   training set.
5. Confirm each fold records a validation-selected decision threshold.
6. Confirm combined training projects have equal total `loss_weight`.

## Expected Warnings

Recent Python versions may produce PyTorch Geometric deprecation warnings from
type inspection. Warnings are not test failures, but dependency compatibility
should be reviewed before upgrading to a Python version where the deprecated
behavior is removed.
