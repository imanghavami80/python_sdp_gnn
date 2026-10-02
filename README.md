# 00 — Multi-View Software Defect Prediction

The primary scenario is **within-project defect prediction**: train and test on
disjoint files from the same project, using a separate model for each project.
The model predicts defects for Java files using software metrics, AST, CFG,
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
excluded from Git. Rebuild inputs using the commands below when extraction changes.
Verified CFG release JARs in `build/` remain cached.

## Rebuild the data

There is one AST, CFG, and NDG implementation. AST preserves syntax information;
NDG uses scope-aware dependency resolution. See
[22 — Representation improvements](docs/features/representation_improvements.md).
The commands below generate these representations at the default paths.

Run these from the repository root in order:

```bash
python scripts/preprocess_promise.py
python scripts/extract_promise_ast.py
python scripts/extract_promise_cfg.py
python scripts/extract_promise_ndg.py
```

CFG is the only behavioral extractor. It uses Soot exceptional control flow,
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

## Run within-project prediction

Existing extracted inputs can be reused. Audit all projects without training:

```bash
.venv/bin/python scripts/evaluate_ndg_within_project.py \
  --preflight-only --device cpu --seed 42 \
  --output-dir outputs/promise/within_project/preflight_seed42
```

Run the baseline in a new directory:

```bash
.venv/bin/python scripts/evaluate_ndg_within_project.py \
  --no-ndg-structural-features \
  --device cpu --seed 42 \
  --output-dir outputs/promise/within_project/baseline_audited_seed42
```

Each project uses a reproducible 60/20/20 training/validation/test split, grouped
by Java source file and stratified by defect status. The full unlabeled NDG is
visible, while training loss uses only training labels: **transductive
within-project evaluation**. Validation selects checkpoints and the threshold;
the same checkpoint is tested without final retraining.

Forrest and Xalan cannot provide both classes in three disjoint splits and are
reported as ineligible in the preflight report: Forrest has only two defective
files and Xalan only one clean file. All 12 are audited; ten support evaluation.
Use `--require-all-projects` to stop before training if any project is ineligible.
Use `--project ant-1.7` for a single project. One seed is the default to keep the
local workload manageable.

Optional structural features are enabled with `--ndg-structural-features`.
Use a new output directory and the same seed for comparisons.

See [23 — Within-project evaluation](docs/training/evaluate_ndg_within_project.md)
for the protocol, leakage boundaries, and output schema.
The first completed baseline is documented in [MODEL_RESULTS.md](MODEL_RESULTS.md).

## Results and documentation

Results live under `outputs/promise/within_project/`. Each experiment saves
`within_project_summary.json`, `project_metrics.csv`, test predictions, exact
file splits, validation histories, and reusable model checkpoints. Existing
nonempty output directories are rejected.
Runs also record per-stage durations, actual epoch counts, environment versions,
and source/input fingerprints. Deterministic PyTorch algorithms are required by
default. Early stopping and small per-project training sets can make runs fast.

[01 — Documentation map](docs/README.md) links the extraction, models, training,
and result interpretation guides.

Keep `--num-workers 0` on macOS. CPU is a practical starting device.
