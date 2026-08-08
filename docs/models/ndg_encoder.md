# NDG Encoder

**Implementation:** `src/thesis_project/models/ndg_encoder.py`

## Responsibility

The NDG encoder is the final model. It combines three file-node views and then
propagates information over typed file dependencies. It returns one contextual
embedding per file node, and the classifier returns one defect logit per node.

## Input Contract

For one project graph or disconnected training projects:

- `metrics_x`: transformed metric features `[num_files, 20]`.
- `ast_x`: fold-specific AST embeddings `[num_files, ast_dim]`.
- `cfg_x`: fold-specific CFG embeddings `[num_files, cfg_dim]`.
- `view_mask`: availability flags `[num_files, 3]`.
- `edge_index`: typed file dependency edges `[2, num_edges]`.
- `edge_type`: relation IDs `[num_edges]`.

The metric view is always available. AST fallback and CFG placeholder views are
masked by the final evaluator unless explicitly configured otherwise.

## Multi-View Fusion

Each view has an independent projection network that maps it into a common
hidden dimension. A learned gate scores the available views for each file. A
masked softmax converts those scores to weights, and unavailable views receive
zero weight.

Conceptually:

```text
file_state = gate(metrics, AST, CFG)
```

This is preferable to blind concatenation because files can legitimately lack a
reliable AST or CFG view.

## Relational Message Passing

The extracted seven forward NDG relation types are augmented with separately
typed reverse relations before training. Reverse relations permit information
flow in both directions while preserving dependency direction semantics.

Trainable relation embeddings are passed into residual GATv2 layers. Therefore,
an `EXTENDS` edge can influence attention differently from `METHOD_CALL`,
`FIELD_TYPE`, or another relation.

## Architecture

1. Project metrics, AST, and CFG independently.
2. Apply masked per-file view gating.
3. Embed typed forward and inverse NDG relations.
4. Apply stacked residual edge-aware GATv2 layers.
5. Normalize and project each contextual file state.
6. Apply the node classifier to produce one defect logit per file.

There is no graph-level pooling because this is node-level prediction.

## Main Classes

### `NDGEncoderConfig`

Defines input dimensions, relation count, hidden/output dimensions, attention
heads, layer count, and dropout.

### `ViewProjection`

Maps one raw view into the common node space.

### `GatedMultiViewFusion`

Computes normalized weights over only the views available for each node.

### `NDGMultiViewRelationalGATEncoder`

Returns one learned embedding per NDG file node.

### `NDGNodeClassifier`

Adds the binary node classification head used for final defect prediction.

## Outputs

For `N` files and output dimension `D`:

- Encoder output: `[N, D]` file-node embeddings.
- Classifier output: `[N]` defect logits.
- Sigmoid of each logit: defect probability for that file.

## Consumer

The strict final consumer is `scripts/evaluate_ndg_nested_lopo.py`.

## Extension Rules

- Preserve node-level output alignment with the NDG node index.
- Add new dependency semantics as edge types, not arbitrary ordinal values.
- Fit any transformation using only outer-training data.
- Maintain view masks whenever a new optional file representation is added.
