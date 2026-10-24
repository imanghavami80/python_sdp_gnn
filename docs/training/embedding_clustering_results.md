# 24 — Embedding Clustering Results

**Historical experiment.** The clustering training option has been removed
from the active pipeline. Saved results are retained for comparison.

## Method

`--embedding-cluster-loss` applies clustering **during NDG model training**, after
the AST and CFG encoders have produced file embeddings. Following five warmup
epochs, it clusters the learned, pre-fusion NDG, AST, and CFG views of all files
in each project. Missing views are masked. The number of groups (2–8) is chosen
from the embeddings using a silhouette score; assignments and view-specific
prototypes are refreshed every five epochs. The auxiliary prototype loss pulls
each available view toward its assigned group and separates it from other groups:

`L = L_defect + 0.05 × L_cluster` (default settings).

This changes the learned representations before fusion; it does not append a
cluster ID to the input features. The upstream AST and CFG encoders are frozen
at this stage, while their trainable fusion projections and the NDG encoder
receive the auxiliary gradient. Clustering uses no defect labels. It can see
unlabeled validation and test files under the declared transductive protocol.
These settings describe the completed historical runs; the option is no longer
available in the evaluator.

## Matched experiment: seed 43

The comparison uses the same extracted inputs, source code, train/validation/test
splits, and 1,054 held-out files from 12 projects. Neither run used optional NDG
structural features. Results are from:

- `outputs/promise/within_project/baseline_embedding_comparison_seed43/`
- `outputs/promise/within_project/embedding_cluster_loss_seed43/`

| Equal-project mean | Baseline | Embedding clustering | Change |
| --- | ---: | ---: | ---: |
| F1 | 0.6283 | 0.6255 | −0.0029 |
| PR-AUC | 0.7592 | 0.7542 | −0.0050 |
| ROC-AUC | 0.8462 | 0.8536 | +0.0074 |
| MCC | 0.4009 | 0.3895 | −0.0114 |
| Balanced accuracy | 0.6918 | 0.7009 | +0.0091 |

Clustering found one more defective test file but incorrectly flagged 16 more
clean files: true positives changed from 346 to 347 and false positives from
153 to 169. Project effects varied: Ivy's F1 rose by 0.144, while JEdit's fell
by 0.064 and Synapse's fell by 0.071. The selected checkpoint used clustering in
11 projects, with 2–4 groups and a median silhouette of 0.325. Forrest's selected
checkpoint had no active clustering.

## Interpretation

**This run does not show an overall defect-prediction improvement.** Clusters
sometimes separated files with different defect rates, but a geometrically
clear group need not help the classifier distinguish defective files. The small
ROC-AUC increase did not translate into better mean F1, PR-AUC, or MCC.

The comparison also has a training-rule limitation: clustering starts at epoch
6, and the clustering scenario selects checkpoints only from epoch 6 onward.
The baseline selected an epoch 1–5 checkpoint in six projects. Those projects'
differences therefore combine the auxiliary loss with a change in checkpoint
selection. One seed cannot establish a reliable gain or loss. A fair next
ablation should give the baseline the same warmup/selection rule, then compare
several preset seeds without choosing settings from test results.
