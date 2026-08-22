# CFG Encoder

**Implementation:** `src/thesis_project/models/cfg_encoder.py`

## Responsibility

The CFG encoder converts a variable-size file-level CFG into one fixed-size file
embedding. Unlike the AST model, control-flow semantics are carried heavily by
typed edges, so the encoder passes learned relation embeddings directly into
graph attention.

## Input Contract

For a batch of CFG graphs:

- `x`: numeric statement and structural features.
- `node_type_id`: broad CFG role IDs.
- `stmt_kind_id`: statement operation IDs.
- `invoke_kind_id`: invocation dispatch IDs.
- `edge_index`: directed control-flow edges.
- `edge_type`: one categorical control-flow relation per edge.
- `batch`: graph assignment per node.

All categorical arrays have one value per node except `edge_type`, which has one
value per edge.

## Node Representation

The encoder concatenates:

```text
node_type_embedding
statement_kind_embedding
invocation_kind_embedding
numeric CFG features
```

Numeric features include statement flags and structural roles. Expected bounded
features are clamped to stable ranges by default. Degree-derived features are
already stored in logarithmic form by extraction.

## Edge Representation

Every CFG relation ID receives a trainable edge embedding. GATv2 uses this edge
vector when computing attention, allowing normal flow, true/false branches,
returns, exceptions, loop back edges, and switch edges to affect messages
differently.

## Architecture

1. Embed node role, statement kind, and invocation kind.
2. Concatenate categorical and numeric node inputs.
3. Project nodes into the hidden space.
4. Embed CFG edge types.
5. Apply stacked residual GATv2 layers with edge features.
6. Aggregate nodes with graph-level attention pooling.
7. Project to the configured file embedding dimension.

## Main Classes

### `CFGEncoderConfig`

Defines all vocabulary sizes, numeric feature count, embedding dimensions,
attention heads, layer count, dropout, and output dimension.

### `CFGEdgeAwareGATEncoder`

Produces one embedding per input CFG. Optional outputs expose pooling attention
and layer-level edge attention for diagnostics.

### `CFGGraphClassifier`

Adds a binary classifier head for supervised CFG embedding training.

## Placeholder Policy

An `ENTRY -> EXIT` placeholder does not contain useful behavior. Placeholder
graphs are excluded from standalone CFG training. In final NDG evaluation, the
file remains an NDG node but the CFG view mask is false and its zero-filled CFG
vector cannot influence multi-view fusion.

## Outputs

For `B` real file CFGs and output dimension `D`, the encoder returns `[B, D]`.
The result is a file embedding, not a method embedding and not a project graph
embedding.

## Consumers

- `scripts/evaluate_cfg_lopo.py`
- `scripts/evaluate_ndg_nested_lopo.py`

## Extension Rules

- Represent program behavior primarily with edge types.
- Keep source-line snippets and identifiers out of the learned numeric vector
  unless a controlled semantic-view experiment requires them.
- Update extractor vocabularies, tensor loaders, config, and tests together.
- Never silently include placeholder CFGs as real training graphs.

