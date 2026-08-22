# NDG Extraction

**Implementation:** `scripts/extract_promise_ndg.py`

## Purpose

This stage creates one Network Dependency Graph for each project. An NDG node
represents a mapped Java file/class, and an edge indicates that the source file
depends on a target file from the same project.

The prediction granularity is node level, not graph level. The final model
predicts the defect label of each file node.

## Prerequisite

Run preprocessing first. AST and CFG extraction are not required to build the
raw NDG, but all three graph stages must be available before final evaluation.

## Nodes

Every mapped row in the preprocessed CSV becomes one node. Stored metadata
includes project name, qualified class name, source path, and binary label.

The initial node feature tensor contains only the 20 `log1p` software metrics.
It excludes identifiers, paths, match strategy, dataset name, and defect label.
Fold-specific AST and CFG embeddings are attached later by the final evaluator.

## Dependency Relations

A directed edge `A -> B` is created when file A depends on file B through one
of these relations:

```text
EXTENDS
IMPLEMENTS
FIELD_TYPE
PARAMETER_TYPE
RETURN_TYPE
OBJECT_CREATION
METHOD_CALL
```

Java source is parsed structurally. Type resolution uses package declarations,
explicit imports, wildcard imports, and project types. Ambiguous simple names
are not assigned arbitrarily.

Multiple relation types may connect the same pair of nodes. Edge type IDs are
categorical relations, not continuous edge feature vectors.

## Run

```bash
python scripts/extract_promise_ndg.py
```

Useful options:

```bash
python scripts/extract_promise_ndg.py --dataset-name log4j-1.2
python scripts/extract_promise_ndg.py --output-dir path/to/ndg-output
python scripts/extract_promise_ndg.py --help
```

## Outputs

```text
outputs/promise/ndg/graph_index.csv
outputs/promise/ndg/edge_type_vocab.json
outputs/promise/ndg/feature_names.json
outputs/promise/ndg/ndg_summary.json
outputs/promise/ndg/parse_failures.json
outputs/promise/ndg/parse_fallbacks.json
outputs/promise/ndg/validation_issues.json
outputs/promise/ndg/graphs/<project>.json
outputs/promise/ndg/tensors/<project>_x.npy
outputs/promise/ndg/tensors/<project>_y.npy
outputs/promise/ndg/tensors/<project>_edge_index.npy
outputs/promise/ndg/tensors/<project>_edge_type.npy
```

## Expected Result

- Exactly one NDG per selected project.
- Exactly one NDG node per mapped file in that project.
- Node labels and metric rows preserve the preprocessed dataset order recorded
  in the JSON graph.
- `edge_index` has shape `[2, num_edges]` and `edge_type` has one value per edge.
- Unresolved external dependencies do not become project nodes.

## Downstream Use

The strict nested evaluator augments every NDG node with fold-specific AST and
CFG embeddings, learns typed inverse relations for message passing, and performs
the final file-level classification.
