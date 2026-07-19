# AST Extraction

**Implementation:** `scripts/extract_promise_ast.py`

## Purpose

This stage converts each mapped Java source file into one filtered Abstract
Syntax Tree graph. The graph preserves useful syntax while avoiding a large,
sparse representation of every parser detail.

## Prerequisite

Run preprocessing first. The default input is:

```text
outputs/promise/promise_preprocessed_log1p.csv
```

Only rows with a mapped `source_path` are processed.

## Graph Contract

Each graph represents one Java file.

Node data:

- `node_type_id`: categorical ID from `node_type_vocab.json`.
- `x[:, 0]`: AST depth.
- `x[:, 1]`: node out-degree.
- `x[:, 2]`: whether the node has an identifier.

Edges are directed parent-to-child AST relations. `edge_index` has shape
`[2, num_edges]`.

Node types are embedded by the model rather than stored as manually assigned
semantic vectors. At model load time, depth is divided by the graph maximum and
out-degree receives `log1p`; `has_identifier` remains binary.

## Extraction Process

1. Load and validate mapped rows.
2. Parse each source file using `javalang`.
3. Retain the configured syntax node types.
4. Build parent-child edges and structural node features.
5. Validate node and edge dimensions.
6. Save a readable JSON graph and NumPy tensors.
7. Record one index row per file.

If normal parsing fails, the extractor builds a coarse fallback syntax tree and
sets `parser_mode=fallback`. This preserves file alignment, but the final model
masks fallback ASTs by default because they are not equivalent to parsed ASTs.

## Run

```bash
python scripts/extract_promise_ast.py
```

Useful options:

```bash
# Extract only one project
python scripts/extract_promise_ast.py --dataset-name log4j-1.1

# Use custom locations
python scripts/extract_promise_ast.py \
  --input-csv path/to/preprocessed.csv \
  --output-dir path/to/ast-output
```

The default behavior cleans graph and tensor output directories. Use
`--no-clean` only when intentionally preserving existing files.

## Outputs

```text
outputs/promise/ast/graph_index.csv
outputs/promise/ast/node_type_vocab.json
outputs/promise/ast/ast_summary.json
outputs/promise/ast/parse_failures.json
outputs/promise/ast/graphs/<file>.json
outputs/promise/ast/tensors/<file>_x.npy
outputs/promise/ast/tensors/<file>_node_type_id.npy
outputs/promise/ast/tensors/<file>_edge_index.npy
```

## Expected Result

- One indexed AST graph for every mapped input row.
- Tensor dimensions agree with `num_nodes`, `num_edges`, and `feature_dim`.
- Every graph retains its dataset, file identity, source path, and defect label.
- Fallback graphs and parse failures are explicitly reported.

## Downstream Use

The AST graph index is consumed by the AST encoder workflows and by the strict
nested NDG evaluator, which trains an AST encoder separately inside each outer
LOPO fold.
