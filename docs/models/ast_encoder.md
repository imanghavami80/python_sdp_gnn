# AST Encoder

**Implementation:** `src/thesis_project/models/ast_encoder.py`

## Responsibility

The AST encoder converts one variable-size file AST into one fixed-size file
embedding. During supervised encoder training, a classifier head maps that
embedding to a defect logit. The final NDG pipeline uses the embedding, not the
AST classifier prediction.

## Input Contract

For a batch of AST graphs:

- `x`: numeric node features with shape `[num_nodes, 3]`.
- `node_type_id`: AST node type IDs with shape `[num_nodes]`.
- `edge_index`: directed AST child relations with shape `[2, num_edges]`.
- `batch`: graph assignment for each node with shape `[num_nodes]`.

The three numeric features are depth, out-degree, and `has_identifier`.

## Normalization

By default, normalization is performed per graph:

```text
depth = depth / maximum graph depth
out_degree = log1p(out_degree)
has_identifier = unchanged
```

This prevents raw depth or high-degree nodes from dominating trainable node-type
embeddings. Normalization can be disabled for controlled ablation experiments,
but should stay enabled in the main pipeline.

## Architecture

1. Embed `node_type_id` using a trainable lookup table.
2. Concatenate node-type embeddings with numeric structural features.
3. Project the combined vector into the hidden dimension.
4. Apply multiple GIN layers with normalization, activation, dropout, and
   residual behavior.
5. Use gated attention pooling to aggregate all AST nodes into one file vector.
6. Project the pooled vector to the configured output dimension.

GIN is used because ASTs are primarily structural, tree-like graphs. Attention
pooling allows the model to assign different aggregation weights to syntax
nodes.

## Main Classes

### `ASTEncoderConfig`

Defines vocabulary size, structural feature dimension, node-type embedding
dimension, hidden/output dimensions, number of layers, dropout, and whether
normalization is enabled.

### `GraphAttentionPooling`

Computes a learned scalar score for each node, normalizes scores within each
graph, and forms a weighted graph representation.

### `ASTGINEncoder`

Returns one embedding per AST graph. It can optionally return node attention
weights for analysis.

### `ASTGraphClassifier`

Wraps the encoder with a binary classification head for supervised embedding
training.

## Outputs

Given `B` file ASTs and output dimension `D`, the encoder returns `[B, D]`.
Attention weights, when requested, contain one value per input node.

Attention is not a causal explanation. It provides only an approximate signal
of which nodes influenced pooling more strongly.

## Consumers

- `scripts/generate_ast_embeddings.py`
- `scripts/evaluate_ast_lopo.py`
- `scripts/evaluate_ndg_nested_lopo.py`

## Extension Rules

- Add syntax information to node features rather than program behavior.
- Keep node-type IDs categorical and trainable.
- Update extraction vocabularies, config validation, loaders, and tests together
  when changing the input contract.
- Preserve one output embedding per file graph.

