# Thesis Project

Initial scaffold for a Python master thesis project.

## Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Run tests

```bash
pytest
```

## GNN modeling

The first model component is `thesis_project.models.ASTGINEncoder`, a GIN-based
AST graph encoder with trainable node-type embeddings and attention pooling.
Before training, AST structural features are normalized graph-by-graph:

```text
depth -> depth / max_depth_in_graph
out_degree -> log1p(out_degree)
has_identifier -> unchanged binary flag
```

Attention weights can be inspected to see which AST nodes influenced pooling
more strongly. Treat them as approximate indicators, not exact causal
explanations for a defect prediction.

Generate trained AST embeddings after running AST extraction:

```bash
python scripts/generate_ast_embeddings.py
```

Default outputs:

- `outputs/promise/embeddings/ast/ast_embeddings.npy`
- `outputs/promise/embeddings/ast/ast_embedding_index.csv`
- `outputs/promise/embeddings/ast/ast_encoder.pt`
- `outputs/promise/embeddings/ast/ast_training_history.csv`
- `outputs/promise/embeddings/ast/ast_embedding_summary.json`

The CFG model component is `thesis_project.models.CFGEdgeAwareGATEncoder`, an
edge-aware GATv2 encoder with attention pooling. CFG node features stay compact:

```text
line_position -> normalized source-line position, clamped to [0, 1]
has_source_line -> binary flag
is_synthetic -> binary flag for ENTRY/EXIT/placeholder nodes
```

CFG behavior is represented through typed directed edges. The encoder learns
trainable edge-type embeddings and passes them into GATv2 attention, so
`CFG_NEXT`, `CFG_TRUE`, `CFG_FALSE`, `CFG_RETURN`, and `CFG_EXCEPTION` can receive
different learned importance. Attention weights are useful for approximate
inspection, not exact causal explanation.

Evaluate the CFG encoder after running CFG extraction:

```bash
python scripts/evaluate_cfg_lopo.py
```

The CFG encoder uses Leave-One-Project-Out (LOPO) because this is a
cross-project SDP setting. Placeholder CFG graphs are always excluded before
splitting because they contain only trivial `ENTRY -> EXIT` structure.

Default outputs:

- `outputs/promise/embeddings/cfg_lopo/fold_metrics.csv`
- `outputs/promise/embeddings/cfg_lopo/aggregate_metrics.json`
- `outputs/promise/embeddings/cfg_lopo/all_test_predictions.csv`
- `outputs/promise/embeddings/cfg_lopo/folds/<project>/cfg_encoder.pt`
- `outputs/promise/embeddings/cfg_lopo/folds/<project>/test_embeddings.npy`
