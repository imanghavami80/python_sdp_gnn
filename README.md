# 00 — Multi-View Software Defect Prediction

The model predicts defects for Java files using software metrics, AST, CFG v3,
and a project-level Network Dependency Graph (NDG). Late fusion is always used:
the NDG GNN first processes metric-based file states and dependencies, and its
embeddings are then fused with independently learned AST and CFG embeddings.

## Setup

Use Python 3.11+ and a Java JDK with `java` and `javac` on PATH.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
pytest
```

Inputs are PROMISE CSVs and Java source trees under `projects_new/`.
The preprocessing script discovers the benchmark projects and maps qualified
class names to source files. `outputs/` contains generated data only and is
excluded from Git. The cleanup removed the previous artifacts; rebuild them
using the commands below. Verified CFG release JARs in `build/` remain cached.

## Rebuild the data

Run these from the repository root in order:

```bash
python scripts/preprocess_promise.py
python scripts/extract_promise_ast.py
python scripts/extract_promise_cfg.py
python scripts/extract_promise_ndg.py
```

CFG v3 is the only behavioral extractor. It uses Soot exceptional control flow,
rejects compiler-error method bodies, and uses checksum-verified exact-release
bytecode for configured Camel/Synapse versions. Missing executable CFGs remain
masked views; their file nodes are retained. There is no CFG/PDG selector.

Run this additional step only if an experiment will use handcrafted NDG features:

```bash
python scripts/extract_ndg_structural_features.py
```

This creates 38 topology descriptors per file with the current seven dependency
relations. Features describe directed degree, relation types, centrality,
reachability, reciprocity, and component membership. Extraction is label-free.

## Experiment scenarios

The two scenario switches are independent:

| Switch | Values | Default |
| --- | --- | --- |
| `--cluster-mode` | `none`, `simple`, `gated` | `none` |
| `--ndg-structural-features` / `--no-ndg-structural-features` | on / off | off |

Simple clustering concatenates the cluster feature vector with metrics before
NDG message passing. Gated clustering projects the same vector separately and
injects a learned residual. Both use the same training-only cluster geometry
and cross-fitted defect-risk construction. Neither clusters AST/CFG embeddings.

Optional NDG structural features enter through a separate residual gate after
NDG message passing, before fusion with AST and CFG. They can accompany any
clustering mode. The late-fusion gate exists in every scenario; “gated cluster”
refers specifically to the additional cluster gate.

All six scenarios use CFG v3 and late fusion:

```bash
# 1. Baseline
python scripts/evaluate_ndg_nested_lopo.py --device cpu

# 2. Simple clustering
python scripts/evaluate_ndg_nested_lopo.py --cluster-mode simple --device cpu

# 3. Gated clustering
python scripts/evaluate_ndg_nested_lopo.py --cluster-mode gated --device cpu

# 4. NDG structural features
python scripts/evaluate_ndg_nested_lopo.py --ndg-structural-features --device cpu

# 5. Simple clustering + NDG structural features
python scripts/evaluate_ndg_nested_lopo.py --cluster-mode simple --ndg-structural-features --device cpu

# 6. Gated clustering + NDG structural features
python scripts/evaluate_ndg_nested_lopo.py --cluster-mode gated --ndg-structural-features --device cpu
```

K-means++ is the default clustering algorithm. Existing `--cluster-method gmm`
and `--cluster-method hdbscan` remain algorithm settings within simple/gated
mode. Keep the algorithm and hyperparameters fixed when comparing the six
scenarios. Run `--help` for training, input-path, and algorithm settings.

The previous fusion selector and cluster-feature boolean switches have been
removed. No fusion argument is needed.

## Outputs and comparison

Default directories identify the scenario, algorithm when active, and seed:

```text
outputs/promise/experiments/cluster_none__ndg_structural_off/seed_42/
outputs/promise/experiments/cluster_gated_kmeans__ndg_structural_on/seed_42/
```

Each contains `run_manifest.json`, `nested_lopo_summary.json`,
`fold_metrics.csv`, `all_test_node_predictions.csv`, embeddings, and fold
checkpoints. The manifest records arguments and input-index hashes. CFG v3
markers and the exact control-flow vocabulary are checked before evaluation.

Existing nonempty experiment directories are rejected. For a pilot, changed
hyperparameters, or another run with the same seed, use a new `--output-dir`.

```bash
python scripts/evaluate_ndg_nested_lopo.py \
  --test-project log4j-1.2 --upstream-epochs 1 --ndg-epochs 1 \
  --device cpu --output-dir outputs/promise/pilot_log4j
```

Nested LOPO retrains AST, CFG, and NDG inside every outer project fold. Scaling,
clustering, epoch selection, and threshold selection exclude the held-out
project. It uses one inner validation project; it does not average over all
possible inner project folds. Missing AST/CFG views are masked.

Compare macro-project PR-AUC, ROC-AUC, MCC, balanced accuracy, F1, and Brier
score across matched seeds and graph artifacts. The majority-class baseline in
each summary is not the full model's no-cluster/no-structural baseline.
Thresholds selected on the inner model are transferred to the final retrained
model; this existing calibration limitation remains and can affect results.
The cleanup does not establish improved predictive performance.

## Documentation

Read [01 — Documentation map](docs/README.md) for the numbered reading order.
Historical measurements remain in [19 — Results record](MODEL_RESULTS.md) and
the CFG progress reports; their raw outputs were deleted during cleanup.

Standalone AST/CFG evaluation and global AST embedding scripts remain diagnostic
tools. Their embeddings must not replace fold-specific training in the final
evaluation. Keep `--num-workers 0` on macOS; CPU is a practical starting device.
