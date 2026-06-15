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
