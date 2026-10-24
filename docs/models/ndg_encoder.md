# NDG encoder and fusion

The NDG encoder starts each file node with a 19-value, label-free dependency
signature. Relation-aware graph attention passes information along directed,
typed edges and produces one NDG embedding per file. Dataset metrics do not
enter these graph updates.

AST, CFG, NDG, and metrics embeddings are projected to a common width. A
masked softmax gate learns one weight per available view and file. The four
weighted vectors keep separate positions in the fusion layer, followed by a
small projection and binary classifier. Missing AST or CFG embeddings receive
zero weight. The gate weights describe model behavior; they are not causal
feature importance.

Optional handcrafted NDG structural descriptors use a gated residual addition
to the NDG embedding. The default four-view experiment leaves them disabled.

See [the model flow](../features/four_view_dependency_inputs.md) for the
diagram and [the evaluation guide](../training/evaluate_ndg_within_project.md)
for training and saved outputs.
