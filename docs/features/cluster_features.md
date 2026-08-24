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

## Available Methods

`--cluster-method kmeans` uses k-means with k-means++ initialization. It is the
default so existing experiments remain reproducible. Fitted centroids provide
a deterministic out-of-sample distance space.

`--cluster-method gmm` uses a Gaussian mixture model. It represents overlapping
components through posterior probabilities and measures distance using each
component's covariance. Full covariance is the default because software metrics
are correlated; `tied`, `diag`, and `spherical` are available as sensitivity
options. Positive covariance regularization limits singular and ill-conditioned
fits.

`--cluster-method hdbscan` uses hierarchical density-based clustering. It is
the density-based comparison because it can find non-spherical, variable-density
groups and mark atypical files as noise. The external `hdbscan` package is used
instead of scikit-learn's built-in implementation because it supports frozen
training-model prediction, soft membership vectors, and outlier scores for
unseen validation/test files.

The implementation does not use a raw integer cluster ID as a model feature.
IDs are arbitrary labels and would incorrectly suggest that cluster 2 is
numerically greater than cluster 1. Instead, for `K` clusters it creates:

1. `K` standardized `log1p` distances to all components;
2. `K` soft memberships: scaled distance weights for k-means or exact posterior
   responsibilities for GMM;
3. one standardized outlier feature: nearest-centroid distance for k-means or
   negative mixture log-likelihood for GMM;
4. one smoothed cluster-conditioned defect-risk feature.

HDBSCAN discovers a variable number of clusters, so a vector with one feature
per cluster would change size between inner selection and final refitting. It
instead produces a fixed six-value density representation: maximum soft
membership, membership entropy, noise probability, approximate assignment
strength, standardized GLOSH outlier score, and smoothed defect risk. This lets
the same gated GNN architecture be selected and finally retrained even when the
number of density clusters changes.

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
Each training project receives equal total influence. K-means accepts exact
sample weights. Scikit-learn GMM does not, so it is fitted on a deterministic
equal-project resample with the same total number of nodes. Risk estimation
always uses exact project weights. Consequently, a large project cannot
dominate merely because it contains more files; risk weights retain a mean of
one, so `alpha` keeps a consistent interpretation.

HDBSCAN deliberately fits density geometry on unique training files. Repeating
small-project files to emulate sample weights would create artificial density
peaks and invalid clusters. Its defect-risk estimate still uses exact equal
total project weights, and no validation/test files or labels influence either
geometry or risk.
For every training node, rates exclude the node's complete project. Validation
and test nodes use all labels from their corresponding training partition.

## Selecting the Number of Clusters

By default, candidate counts from 2 through 10 are fitted on the inner-fit
nodes. K-means selects the maximum silhouette score, using at most 2,000 nodes.
GMM selects the minimum Bayesian information criterion (BIC) on its balanced
training sample. Ties prefer the smaller model. Defect labels and the
inner-validation project are never used for this choice.

HDBSCAN selects from `--hdbscan-min-cluster-sizes` using maximum relative DBCV,
an unsupervised density-cluster validity measure. `--hdbscan-min-samples`
controls the common density-neighborhood size. It may legitimately label some
or all files as noise; the fixed branch and global-risk fallback remain valid.

Metadata and console output flag a selection on either search boundary. An
upper-bound selection means the configured range should be reported explicitly
and checked in a later sensitivity analysis; it does not authorize changing the
range after looking at held-out-project performance.

A fixed predeclared count can be supplied with `--cluster-count K`. This is
mainly useful for a controlled sensitivity experiment.

## Nested Leakage Boundary

For each outer test project:

1. Metrics are imputed and standardized from inner-fit nodes only.
2. Candidate clusterers and `K` are selected from those inner-fit metrics using
   the criterion belonging to the chosen method.
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

## Design Boundaries

- GMM is an explicit algorithm ablation, not a replacement for k-means. Its
  covariance regularization limits unstable full-covariance estimation.
- HDBSCAN is the appropriate density-based ablation because ordinary DBSCAN
  requires a single global radius and has no comparable frozen soft-prediction
  interface. Its number of clusters and noise fraction are diagnostics, not
  model inputs with arbitrary ordinal meaning.
- Spectral and agglomerative clustering do not naturally provide the required
  fitted transform for unseen projects.
- DBSCAN/HDBSCAN can discover noise and irregular shapes, but produce a
  variable number of clusters and do not provide the same stable centroid
  feature space.
- A single hard k-means label discards similarity to the other clusters and is
  an arbitrary categorical code.

The density-based alternatives can be studied later, but do not provide the
same fixed, stable component space for unseen projects.

## Run and Ablation

Cluster features are enabled by default. For the proposal's late-fusion model:

```bash
python scripts/evaluate_ndg_nested_lopo.py \
  --fusion-stage late \
  --cluster-features \
  --cluster-method kmeans \
  --output-dir outputs/promise/nested_lopo_late_cluster_gate_kmeans \
  --device cpu
```

Run the controlled GMM alternative separately:

```bash
python scripts/evaluate_ndg_nested_lopo.py \
  --fusion-stage late \
  --cluster-features \
  --cluster-method gmm \
  --output-dir outputs/promise/nested_lopo_late_cluster_gate_gmm \
  --device cpu
```

Run the density-based HDBSCAN alternative separately:

```bash
python scripts/evaluate_ndg_nested_lopo.py \
  --fusion-stage late \
  --cluster-features \
  --cluster-method hdbscan \
  --output-dir outputs/promise/nested_lopo_late_cluster_gate_hdbscan \
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

Each fold writes `cluster_features.json`, containing the method, candidate
silhouette, BIC, or relative-DBCV scores, selected parameter, fitted parameters, normalization values,
smoothed defect rates, and feature names. `fold_metrics.csv` records the selected
count, branch dimension, and test gate statistics. The overall summary records
selected counts by fold. Prediction files include `cluster_defect_risk` and
`cluster_gate`.

For HDBSCAN, the metadata additionally records the selected `min_cluster_size`,
relative DBCV, persistence, and training noise fraction. Its `cluster_id` is
`-1` for predicted noise and must never be used as a numeric model feature.

## References

- Arthur and Vassilvitskii, [k-means++: The Advantages of Careful
  Seeding](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf), 2007.
- Rousseeuw, *Silhouettes: a Graphical Aid to the Interpretation and
  Validation of Cluster Analysis*, 1987; formula and API summarized in the
  [scikit-learn silhouette documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html).
- The scikit-learn [`KMeans.transform`
  documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
  defines the centroid-distance feature space used here.
- The scikit-learn [`GaussianMixture`
  documentation](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)
  defines posterior responsibilities, covariance options, and BIC.
- The [HDBSCAN prediction API](https://hdbscan.readthedocs.io/en/latest/prediction_tutorial.html)
  documents training-only approximate prediction for unseen data; its
  [API reference](https://hdbscan.readthedocs.io/en/latest/api.html) documents
  soft memberships, outlier scores, and relative DBCV.
