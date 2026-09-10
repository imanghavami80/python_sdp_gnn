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

Checks CFG input normalization, categorical and typed-edge contracts, encoder
and classifier shapes, ECJ paths containing spaces, source-level detection,
TSV parsing, and graph-cycle loop detection.

### `tests/test_extract_promise_cfg.py`

Checks exact-release bytecode cache reuse, checksum-verified archive member
extraction, and source-build behavior for projects without a configured official
binary release.

### `tests/test_ndg_encoder.py`

Checks multi-view masking, early/late fusion behavior, node-level output shape, reverse relations, outer
project leakage guards, validation-project suitability, and training-only metric
transformation.

### `tests/test_cluster_features.py`

Checks automatic and fixed cluster counts, output dimensions, soft-membership
normalization, preservation of graph labels/topology, rejection of unprocessed
non-finite metrics, transformation without refitting on test data, and
leave-one-project-out encoding of training cluster risk. It also verifies that
differently sized projects receive equal total cluster-fitting and risk weight,
and exercises GMM with full, tied, diagonal, and spherical covariance.
It also verifies fixed-size HDBSCAN features for both discovered-cluster and
all-noise cases.

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
7. Confirm each fold's `cluster_features.json` was fitted without the outer test project.

## Expected Warnings

Recent Python versions may produce PyTorch Geometric deprecation warnings from
type inspection. Warnings are not test failures, but dependency compatibility
should be reviewed before upgrading to a Python version where the deprecated
behavior is removed.
