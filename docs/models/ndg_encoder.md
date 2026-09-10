# NDG Encoder

**Implementation:** `src/thesis_project/models/ndg_encoder.py`

## Responsibility

The NDG encoder is the final model. It combines three file-node views and then
propagates information over typed file dependencies. It returns one contextual
embedding per file node, and the classifier returns one defect logit per node.

## Input Contract

For one project graph or disconnected training projects:

- `metrics_x`: the 20 transformed metrics `[num_files, metrics_dim]`.
- `cluster_x`: an optional separate cluster representation containing geometry
  and leakage-safe risk features `[num_files, cluster_dim]`.
- `ast_x`: fold-specific AST embeddings `[num_files, ast_dim]`.
- `cfg_x`: fold-specific CFG embeddings `[num_files, cfg_dim]`.
- `view_mask`: availability flags `[num_files, 3]`.
- `edge_index`: typed file dependency edges `[2, num_edges]`.
- `edge_type`: relation IDs `[num_edges]`.

The metric view is always available. AST fallback and CFG placeholder views are
masked by the final evaluator unless explicitly configured otherwise.

## Gated Cluster Branch

Cluster features are never concatenated into `metrics_x`. The metrics and
cluster tensors have independent projection networks. A sigmoid gate produces
one weight per file and injects the cluster projection as a residual:

```text
metric_state = metric_projection(metrics)
cluster_state = cluster_projection(cluster_features)
NDG_input = LayerNorm(metric_state + gate * cluster_state)
```

The final gate bias is initialized to `-2`, so training begins near the
no-cluster baseline instead of forcing an unverified feature family into every
node. The learned per-file gate is written with predictions for diagnostics.

## Multi-View Fusion Variants

Each view has an independent projection network that maps it into a common
hidden dimension. A learned gate scores the available views for each file. A
masked softmax converts those scores to weights, and unavailable views receive
zero weight.

The default `early` variant is the implementation design:

```text
file_state = gate(metrics, AST, CFG)
file_embedding = NDG_GNN(file_state, dependencies)
```

The `late` variant implements the proposal design:

```text
ndg_embedding = NDG_GNN(metrics, dependencies)
file_embedding = gate(ndg_embedding, AST_embedding, CFG_embedding)
```

In late fusion, AST/CFG inputs cannot change the independently learned NDG
embedding. Both variants use masked gating for unavailable AST or CFG views.

## Relational Message Passing

The extracted seven forward NDG relation types are augmented with separately
typed reverse relations before training. Reverse relations permit information
flow in both directions while preserving dependency direction semantics.

Trainable relation embeddings are passed into residual GATv2 layers. Therefore,
an `EXTENDS` edge can influence attention differently from `METHOD_CALL`,
`FIELD_TYPE`, or another relation.

## Architecture

1. Project the inputs required by the selected fusion stage.
2. Optionally inject the separately projected cluster branch through its gate.
3. Embed typed forward and inverse NDG relations.
4. Apply stacked residual edge-aware GATv2 layers.
5. Fuse before message passing (`early`) or after independent NDG encoding
   (`late`).
6. Normalize and project each contextual file state.
7. Apply the node classifier to produce one defect logit per file.

There is no graph-level pooling because this is node-level prediction.

## Main Classes

### `NDGEncoderConfig`

Defines input dimensions, relation count, hidden/output dimensions, attention
heads, layer count, and dropout.

### `ViewProjection`

Maps one raw view into the common node space.

### `GatedMultiViewFusion`

Computes normalized weights over only the views available for each node.

### `GatedClusterMetricEncoder`

Projects metrics and cluster features independently and controls cluster
influence with a bounded residual gate.

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
