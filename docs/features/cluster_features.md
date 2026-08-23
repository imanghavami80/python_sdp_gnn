# Cluster-Based NDG Features

**Implementation:** `src/thesis_project/training/clustering.py`

## Purpose

The proposal requires cluster-derived features from the handcrafted PROMISE
metrics. They form a separate gated branch for each NDG file node. AST and CFG
graphs are not clustered: they already have independent learned
representations.

Cluster geometry is unsupervised and never receives defect labels. A separate
smoothed risk feature uses training labels after clustering, with cross-fitting
that prevents a project from being encoded by its own labels.

## Selected Method

The implementation uses k-means with k-means++ initialization. This method was
selected because the inputs are a small, fixed set of numeric metrics and the
fitted centroids can transform unseen files into a fixed-dimensional distance
space. It is deterministic under the configured seed, fast enough to fit
inside every nested fold, and supports a clean train/transform boundary.

The implementation does not use a raw integer cluster ID as a model feature.
IDs are arbitrary labels and would incorrectly suggest that cluster 2 is
numerically greater than cluster 1. Instead, for `K` clusters it creates:

1. `K` standardized `log1p` distances to all centroids;
2. `K` soft memberships computed from Gaussian distance weights, scaled by
   each training cluster's typical within-cluster distance;
3. one standardized nearest-centroid distance as an outlier feature;
4. one smoothed cluster-conditioned defect-risk feature.

This creates `2K + 2` values in `cluster_x` while retaining the 20 original
metrics unchanged in `metrics_x`. The hard cluster ID, maximum soft-membership
confidence, outlier value, risk, and learned gate are written to prediction
CSVs for diagnostics.

## Separate Gated Branch

Metrics and cluster features have independent projection networks. A sigmoid
gate controls the cluster residual before NDG message passing:

```text
metric_state = metric_projection(metrics_x)
cluster_state = cluster_projection(cluster_x)
ndg_input = LayerNorm(metric_state + gate * cluster_state)
```

The gate starts near zero through a `-2` output bias. The model therefore begins
near the established no-cluster baseline and must learn evidence before giving
the cluster branch substantial influence.

## Cluster Defect Risk

Soft component memberships weight the training defect counts. For component
`j`, the risk is shrunk toward the relevant training-set defect rate:

```text
risk_j = (soft_defects_j + alpha * global_rate) /
         (soft_count_j + alpha)
file_risk = sum(membership_j * risk_j)
```

`alpha` defaults to 20 and is controlled by `--cluster-risk-smoothing`.
Each training project receives equal total sample weight when fitting k-means
and estimating these rates. Consequently, a large project cannot dominate the
cluster centers or risk merely because it contains more files; the weights are
normalized to retain a mean node weight of one, so `alpha` keeps a consistent
interpretation.
For every training node, rates exclude the node's complete project. Validation
and test nodes use all labels from their corresponding training partition.

## Selecting the Number of Clusters

By default, candidate counts from 2 through 10 are fitted on the inner-fit
nodes. Mean silhouette score selects `K`; ties prefer the smaller model. The
score uses at most 2,000 training nodes to control its quadratic distance cost.
Defect labels and the inner-validation project are not used for this choice.

A fixed predeclared count can be supplied with `--cluster-count K`. This is
mainly useful for a controlled sensitivity experiment.

## Nested Leakage Boundary

For each outer test project:

1. Metrics are imputed and standardized from inner-fit nodes only.
2. Candidate clusterers and `K` are selected from those inner-fit metrics.
3. Each inner-fit project's training risk excludes that project's labels.
4. The inner-validation project is transformed using fixed inner-fit
   centroids and risk fitted from inner-fit labels.
5. Metrics are independently refitted on all outer-training nodes.
6. A fresh clusterer with the already selected `K` is fitted on those
   outer-training nodes.
7. Each outer-training project's risk excludes its own labels.
8. The held-out project is transformed using fixed final centroids and risk
   fitted from all outer-training labels.

The test project therefore cannot influence scaling, `K`, centroids, distance
normalization, soft-membership scales, or defect risk.

## Why Not the Other Common Methods?

- Gaussian mixtures add covariance estimation and distribution assumptions
  that are fragile for correlated metrics and differently sized projects.
- Spectral and agglomerative clustering do not naturally provide the required
  fitted transform for unseen projects.
- DBSCAN/HDBSCAN can discover noise and irregular shapes, but produce a
  variable number of clusters and do not provide the same stable centroid
  feature space.
- A single hard k-means label discards similarity to the other clusters and is
  an arbitrary categorical code.

These alternatives can be studied later, but they are not a stronger default
for a leakage-safe cross-project feature transformer.

## Run and Ablation

Cluster features are enabled by default. For the proposal's late-fusion model:

```bash
python scripts/evaluate_ndg_nested_lopo.py \
  --fusion-stage late \
  --cluster-features \
  --output-dir outputs/promise/nested_lopo_late_cluster_gate_kmeans \
  --device cpu
```

Compare it with the same architecture and seed without cluster features:

```bash
python scripts/evaluate_ndg_nested_lopo.py \
  --fusion-stage late \
  --no-cluster-features \
  --output-dir outputs/promise/nested_lopo_late \
  --device cpu
```

Do not claim an improvement until this paired ablation is complete across all
projects and preferably multiple seeds.

## Saved Evidence

Each fold writes `cluster_features.json`, containing candidate silhouette
scores, selected `K`, centroids, feature normalization values, membership
scales, smoothed defect rates, and feature names. `fold_metrics.csv` records the
selected count, branch dimension, and test gate statistics. The overall summary
records selected counts by fold. Prediction files include `cluster_defect_risk`
and `cluster_gate`.

## References

- Arthur and Vassilvitskii, [k-means++: The Advantages of Careful
  Seeding](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf), 2007.
- Rousseeuw, *Silhouettes: a Graphical Aid to the Interpretation and
  Validation of Cluster Analysis*, 1987; formula and API summarized in the
  [scikit-learn silhouette documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html).
- The scikit-learn [`KMeans.transform`
  documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
  defines the centroid-distance feature space used here.
