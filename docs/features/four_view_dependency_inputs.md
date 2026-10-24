# Four-view model flow

Each project is trained separately. Every Java file gets four vectors:

| View | How it is built |
| --- | --- |
| AST | A file's syntax graph passes through the AST encoder. |
| CFG | A file's control-flow graph passes through the CFG encoder. |
| NDG | Its typed dependency counts initialize a node in the project dependency graph; a relational GNN processes the graph. |
| Metrics | Its PROMISE metrics pass through a separate learned projection. |

The NDG node input has 19 values: total incoming/outgoing counts, an isolate
flag, incoming/outgoing counts for each of seven dependency types, and two
relation-diversity values. Counts use `log1p`; only original forward edges are
counted. Training files determine the scaling parameters for both dependency
inputs and metrics. No defect labels are used to build these inputs.

```text
AST graph ── AST encoder ──────────────────┐
CFG graph ── CFG encoder ──────────────────┤
NDG edges + dependency counts ── NDG GNN ─┤── gated late fusion ── defect score
PROMISE metrics ── metrics projection ─────┘
```

## Fusion

Each view is independently projected to the same width. A learned gate scores
the four views and a softmax turns those scores into attention weights that sum
to one **for each file**. Missing AST or CFG
views receive exactly zero weight. The weighted vectors remain in separate
slots, so the final projection can use both view identity and combinations of
views. A classifier then predicts the file's defect probability.

This is gated attention over the four views, rather than Transformer-style
self-attention between views. It is a practical choice for unequal view quality
and missing source graphs. It is the implemented fusion method, not a claim
that it is universally optimal; its value must be established by evaluation.
Metrics do not enter NDG message passing, so the graph embedding reflects
dependencies while the metrics view remains independently available.

For every project, `file_view_embeddings.npz` saves the four projected vectors
in NDG node order: `ndg`, `metrics`, `ast`, and `cfg`, with `node_names`,
`view_mask`, and `fusion_weights`. Missing view vectors are zero. These are the
vectors presented to fusion after projection and view-identity encoding; they
are not four independent defect predictions.
