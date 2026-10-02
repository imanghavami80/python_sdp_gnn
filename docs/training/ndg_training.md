# 12 — Within-Project Training Utilities

`src/thesis_project/training/ndg.py` provides the project graph container, typed
inverse relations, model inputs, binary metrics and validation F1 threshold
selection. `training/within_project.py` implements grouped file splits,
training-only preprocessing and label-masked NDG training.

A full project graph contains metric, AST and CFG tensors, a view-availability
mask, dependency edges, and optional structural features. Test labels are hidden
from the trainer. Message passing uses the full unlabeled graph; only training
indices contribute to BCE. Validation loss selects the checkpoint and validation
probabilities select its decision threshold. That same checkpoint is evaluated.

`scripts/ast_training.py` and `scripts/cfg_training.py` supply file-view datasets,
loaders, training and embedding extraction. `scripts/ndg_common.py` centralizes
input loading and reporting. Only `scripts/evaluate_ndg_within_project.py` runs
the complete experiment. See [the evaluation guide](evaluate_ndg_within_project.md).
