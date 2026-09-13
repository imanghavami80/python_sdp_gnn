# 11 — NDG Encoder

**Implementation:** `src/thesis_project/models/ndg_encoder.py`

## Contract

The encoder always performs late fusion. It returns one contextual embedding
per NDG file node; the classifier produces one defect logit per file.

Inputs are `metrics_x`, `cluster_x`, `ndg_structural_x`, independent
`ast_x` and `cfg_x` embeddings, `view_mask`, `edge_index`, and
`edge_type`. Disabled feature branches use zero-width tensors. The metrics
view is required; unavailable AST and CFG views are masked.

## Cluster integration before NDG message passing

`NDGEncoderConfig.cluster_mode` determines the metric-state construction:

| Mode | NDG input |
| --- | --- |
| `none` | projection of metrics |
| `simple` | projection of concatenated metrics and cluster features |
| `gated` | LayerNorm(metric projection + learned gate × cluster projection) |

The simple and gated modes receive the same cluster feature vector. Their
`cluster_dim` must be positive; none requires zero. The cluster gate is a
sigmoid with final bias initialized to -2. No cluster gate exists in simple
mode. The tensor `metrics_x` remains the original 20 metrics in every mode.

## Relational NDG encoding

Seven extracted dependency types receive separately typed inverse edges for
training. Relation embeddings enter residual GATv2 layers. Layer outputs are
concatenated and projected to a file embedding. AST and CFG values cannot
affect this independently computed NDG embedding.

## Optional structural augmentation

When `ndg_structural_dim > 0`, a separate MLP projects training-standardized
topology descriptors. A sigmoid gate controls their residual addition after
message passing. This branch is independent of the cluster mode.

## Fixed late fusion

The learned NDG embedding (optionally enriched with structural features) is
fused with AST and CFG embeddings through independent projections, masked
softmax view weights, and a final projection. There is no early-fusion module,
configuration field, or execution path.

`NDGNodeClassifier` applies dropout and a linear binary head to each file
embedding. No project-level pooling is used.

## Diagnostics

Attention output includes `ndg_embeddings`, `view_weights`,
`edge_attention`, `cluster_gate`, and `ndg_structural_gate`.
Disabled gates return zeros internally; prediction CSVs use missing values for
gates that do not exist in the scenario. Gate values are not causal feature
importance.

The architecture cleanup changes parameter initialization and checkpoint
structure. Historical checkpoints are not compatible with the new configuration;
rerun matched scenarios with the current code.

See [08 — Cluster features](../features/cluster_features.md) and
[07 — NDG structural features](../features/ndg_structural_features.md).
