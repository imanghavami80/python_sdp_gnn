# 07 — Handcrafted NDG Structural Features

**Implementations:** `scripts/extract_ndg_structural_features.py` and
`src/thesis_project/features/ndg_structural.py`

## Objective

This view describes the structural role of every file in its project dependency
graph. It complements the learned relational GNN: message passing learns from
neighbor content, while the handcrafted branch makes established local and
global network properties directly available to the predictor.

The extractor uses only NDG topology and typed dependency relations. It never
reads defect labels or source-project statistics from another project.

## Features

The default seven NDG relations produce 38 features per file:

- incoming, outgoing, unique, and total dependency degree;
- an isolate indicator;
- incoming and outgoing counts for each typed relation;
- relation diversity and normalized relation entropy;
- PageRank in both dependency directions;
- directed betweenness and incoming/outgoing closeness;
- directed clustering coefficient and normalized core number;
- reciprocal-neighbor and two-hop reachability measures;
- mean neighbor degree; and
- weakly and strongly connected component size fractions.

Counts use `log1p`, bounded quantities are normalized, and betweenness uses at
most 128 deterministic pivots so extraction remains practical on an M4 Mac.
Other global features are exact. With the current 5,303-node, 12-project data,
feature extraction takes about one second; neural training remains the dominant
cost.

The direction follows the NDG edge convention: an outgoing edge represents a
dependency used by the source file, while an incoming edge represents another
file depending on the current file.

These choices are motivated by prior defect-prediction work using dependency
network measures ([Zimmermann and Nagappan, 2008](https://doi.org/10.1145/1368088.1368161))
and later evidence that ego and global network measures can complement source
code metrics ([Tantithamthavorn et al., 2022](https://arxiv.org/abs/2202.06145)).

## Model Integration

For every nested LOPO split, feature means and standard deviations are learned
from training nodes only. The standardized values pass through their own MLP.
A sigmoid gate then adds them as a residual to the learned NDG embedding:

```text
ndg_embedding = relational_NDG_GNN(metrics, typed_dependencies)
structural_state = structural_MLP(structural_features)
enriched_ndg = ndg_embedding + gate * structural_state
prediction_state = late_fusion(enriched_ndg, AST_embedding, CFG_embedding)
```

The gate starts near zero (bias `-2`) so the model initially behaves close to
the existing NDG model and increases structural influence only when training
supports it. Each held-out file's gate value is saved as
`ndg_structural_gate` for diagnostics.

The branch is disabled by default. It is independent of clustering: enable it
with none, simple, or gated cluster mode. When disabled, extraction outputs are
not required.

## Commands

Extract the features after creating the NDGs:

```bash
.venv/bin/python scripts/extract_ndg_structural_features.py
```

Run the proposed late-fusion experiment on CPU:

```bash
.venv/bin/python scripts/evaluate_ndg_nested_lopo.py \
  --ndg-structural-features \
  --cluster-mode none \
  --device cpu
```

Run the directly comparable ablation with the same evaluator and seed:

```bash
.venv/bin/python scripts/evaluate_ndg_nested_lopo.py \
  --no-ndg-structural-features \
  --cluster-mode none \
  --device cpu
```

Compare macro-project metrics across multiple seeds before claiming an
improvement. The feature extractor is deterministic; neural training remains
stochastic.
