# Multi-View Software Defect Prediction

This project predicts whether a Java file is defective by combining four
sources of software information:

- **Software metrics:** file-level measurements from PROMISE, such as LOC, WMC,
  CBO, RFC, and complexity.
- **AST:** the syntactic structure of each Java file.
- **CFG:** typed normal and exceptional execution flow inside methods in each
  Java file.
- **NDG:** typed dependencies between files in the same project.

The final model performs **node-level prediction on each project NDG**. Every
NDG node represents one Java file. Two controlled fusion variants are available:

- `early` (implementation design): metrics, AST, and CFG initialize each NDG
  node before dependency message passing.
- `late` (proposal design): the NDG learns from metrics and dependencies first;
  its node embedding is then fused with independent AST and CFG embeddings.

Both variants output one defect probability and one learned embedding per file.

## Pipeline Overview

```text
PROMISE CSV + Java source code
              |
              v
       Preprocess and map files
              |
       +-------------+-------------+
       |             |             |
       v             v             v
   Extract AST   Extract CFG    Software metrics
       |          with Soot          |
       v             |               v
   AST encoder       v        Training-only k-means++ / GMM / HDBSCAN
                 CFG encoder   cluster geometry + risk
       |             |               |
       +-------------+---------------+
              |
Gated metrics/cluster branch + AST embedding + CFG embedding
              |
              v
     Project-level typed NDG
              |
              v
  Relational GNN node classification
              |
              v
 Defect probability and embedding for every file
```

The project uses strict nested Leave-One-Project-Out (LOPO) as its primary
leakage-safe evaluation. One complete project is held out for testing, so its
files cannot influence training, normalization, or model selection.

## Repository Layout

```text
projects_new/                 12-project PROMISE benchmark and Java source code
projects_old/                 Previous dataset releases; not used by default
scripts/                      Preprocessing, extraction, and evaluation commands
docs/                         File-by-file developer documentation
src/thesis_project/models/    AST, CFG, and NDG model implementations
src/thesis_project/training/  Shared leakage-safe training utilities
tests/                        Automated tests
tools/                        Java analysis dependencies used by CFG extraction
outputs/                      Generated datasets, graphs, tensors, and results
```

`outputs/` is generated locally and is excluded from Git.

Detailed developer documentation for every important implementation file is
available in [`docs/README.md`](docs/README.md).

## Setup

Requirements:

- Python 3.11 or newer
- A Java JDK available through the `java` and `javac` commands

Create the environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
```

Run the tests to verify the installation:

```bash
pytest
```

## Input Data

Each project directory under `projects_new/` contains one PROMISE CSV and the
corresponding Java source tree. For example:

```text
projects_new/
  Log4j/
    log4j-1.2.csv
    <Java source directories>
```

The CSV must contain `name`, `bug`, and the expected 20 software metric
columns. `name` should identify the package-qualified Java class. The
preprocessor automatically discovers CSV files under `projects_new/*/*.csv`.
It handles the benchmark layout that repeats the `name` header for project
metadata and the qualified Java class.

## Run the Complete Pipeline

Run these commands in order from the repository root.

Before running a stage, use `--help` to view that script's available inputs and
options. Replace `<script>` with the script filename:

```bash
python scripts/<script>.py --help
```

For example:

```bash
python scripts/preprocess_promise.py --help
```

### 1. Preprocess and Map the Data

```bash
python scripts/preprocess_promise.py
```

This step:

- Converts the defect count into a binary label: `0` for clean and `1` for
  defective.
- Applies `log1p` to the non-negative software metrics.
- Maps each PROMISE class to its Java source file using its qualified name.
- Combines all project datasets into one CSV.

Main output:

```text
outputs/promise/promise_preprocessed_log1p.csv
```

Standard scaling is intentionally not fitted here. It is fitted only on the
training projects inside each final LOPO fold to prevent information leakage.

### 2. Extract AST Graphs

```bash
python scripts/extract_promise_ast.py
```

This creates one filtered AST graph per mapped Java file. Each graph contains
node type IDs, structural features, AST edges, tensors, and the file defect
label. Numeric structural features are normalized when the encoder loads them.

Main outputs:

```text
outputs/promise/ast/graph_index.csv
outputs/promise/ast/graphs/
outputs/promise/ast/tensors/
outputs/promise/ast/ast_summary.json
```

Files that cannot be parsed normally receive a tagged fallback AST. The final
model masks these fallback views by default.

### 3. Extract CFG Graphs

```bash
python scripts/extract_promise_cfg.py
```

This compiles each complete project-version source tree with partial-error
recovery, then builds a statement-level Soot CFG for every mapped file. Each
edge has one precise execution meaning: entry, fall-through, true/false branch,
goto, switch case/default, return, explicit throw, caught exception, or escaping
exception. Loop features are derived from graph cycles rather than source order.
For Camel 1.6 and Synapse 1.2, checksum-verified bytecode from the matching
official Apache release takes precedence over invalid compiler-error stubs and
is cached after the first run. A missing archive or checksum failure stops the
extractor so an experiment cannot silently revert to lower CFG coverage.

Main outputs:

```text
outputs/promise/cfg/graph_index.csv
outputs/promise/cfg/graphs/
outputs/promise/cfg/tensors/
outputs/promise/cfg/cfg_summary.json
```

If Soot cannot recover a useful CFG, the file receives a tagged placeholder
graph. Placeholder CFGs are treated as an unavailable view by the final model.

### 4. Extract Project NDGs

```bash
python scripts/extract_promise_ndg.py
```

This creates one directed, typed dependency graph per project. Every node is a
mapped Java file. Edges represent relations such as inheritance,
implementation, field types, parameter or return types, object creation, and
method calls.

Main outputs:

```text
outputs/promise/ndg/graph_index.csv
outputs/promise/ndg/graphs/
outputs/promise/ndg/tensors/
outputs/promise/ndg/ndg_summary.json
```

At extraction time, NDG node tensors contain the 20 preprocessed metrics. The
final evaluator adds training-only cluster features and fold-specific AST and
CFG embeddings. See [cluster-based features](docs/features/cluster_features.md).

### 5. Train and Evaluate the Final Model

```bash
python scripts/evaluate_ndg_nested_lopo.py --device cpu
```

Use `--device mps` on a compatible Apple Silicon environment or `--device cuda`
on a CUDA system.

For every outer LOPO fold, the evaluator:

1. Holds out one complete project for testing.
2. Uses another training project for inner epoch selection.
3. Fits metric imputation and scaling only on training files.
4. Selects and fits k-means++, GMM, or HDBSCAN using only the appropriate training files,
   then builds a separate cluster view from component distances, soft
   memberships, outlier evidence, and cross-fitted cluster defect risk.
5. Trains the AST and CFG encoders without the outer test project.
6. Generates fold-specific AST and CFG embeddings.
7. Injects the cluster view through a learnable residual gate, then combines
   the resulting NDG view with the available AST and CFG embeddings.
8. Gives every training project equal total loss weight, preventing large
   projects from dominating optimization.
9. Selects the decision threshold using only the inner-validation project.
10. Trains the relational NDG model and predicts the held-out project.
11. Repeats the process until every project has been tested once.

Use `--fusion-stage early` for the implementation design or
`--fusion-stage late` for the proposal design. Write them to separate output
directories and report both as a predeclared ablation.

Cluster features are enabled by default. Use `--no-cluster-features` only for
the required with/without-clustering ablation. Automatic selection considers
`K=2..10`; `--cluster-count K` uses a predeclared fixed count. Select
`--cluster-method kmeans` for silhouette-selected k-means++ or
`--cluster-method gmm` for BIC-selected Gaussian mixtures, or
`--cluster-method hdbscan` for relative-DBCV-selected density clustering.
`--cluster-risk-smoothing` controls shrinkage of training-only cluster defect
rates. Keep its default unless it is tuned entirely inside the nested protocol.

Main results:

```text
outputs/promise/final_ndg_nested_lopo/all_test_node_predictions.csv
outputs/promise/final_ndg_nested_lopo/ndg_node_embeddings.npy
outputs/promise/final_ndg_nested_lopo/ndg_node_embedding_index.csv
outputs/promise/final_ndg_nested_lopo/fold_metrics.csv
outputs/promise/final_ndg_nested_lopo/nested_lopo_summary.json
```

`all_test_node_predictions.csv` contains the label, probability, selected
threshold, and prediction for every held-out file. `nested_lopo_summary.json`
contains pooled and macro-project classification, ranking, and calibration
metrics together with majority-class baseline results.

Report macro-project metrics as the primary cross-project result. In addition to
accuracy and F1, inspect balanced accuracy, MCC, ROC-AUC, PR-AUC, and Brier
score. Pooled metrics can be dominated by the largest project.

To run a quick pilot on one held-out project before the complete experiment:

```bash
python scripts/evaluate_ndg_nested_lopo.py \
  --test-project log4j-1.2 \
  --output-dir outputs/promise/pilot_log4j \
  --device cpu
```

Use a separate pilot directory because the selected output directory is cleaned
when evaluation starts.

### Long Silent Stages

Nested LOPO retrains and re-encodes AST and CFG graphs inside every outer fold.
After an `early_stopping` message, the next operation may be an embedding pass
over thousands of files rather than another epoch. Progress is reported as:

```text
fold=<project> stage=<stage> status=started
ast_encoding batch=<current>/<total>
cfg_encoding batch=<current>/<total>
```

MPS is not always faster for many small, irregular graphs. If MPS stops making
batch progress, interrupt it and run the one-fold pilot with `--device cpu`.
Keep `--num-workers 0` on macOS unless multiprocessing has been tested.

## Optional Standalone Experiments

These commands help inspect individual encoders, but their saved embeddings are
not used by the strict final NDG evaluator:

```bash
# Exploratory AST embeddings from a global model
python scripts/generate_ast_embeddings.py

# Project-held-out AST evaluation
python scripts/evaluate_ast_lopo.py

# Legacy project-held-out behavioral-graph evaluation
python scripts/evaluate_cfg_lopo.py
```

The final evaluator retrains both upstream encoders inside every outer fold. Do
not substitute globally trained embeddings in the final cross-project results,
because that would leak information from held-out projects.

## Model Summary

- **AST encoder:** GIN with node-type embeddings and attention pooling.
- **CFG encoder:** one edge-aware GATv2 stack over typed control-flow relations,
  followed by attention pooling.
- **NDG encoder:** relational GATv2 with selectable early or late multi-view
  fusion at file-node level.

Attention weights can provide an approximate indication of influential nodes,
but they should not be treated as exact causal explanations.

## Adding Another Project

Place its PROMISE CSV and Java source code under one project directory, then
rerun the pipeline from preprocessing:

```text
projects_new/<project-name>/<dataset-name>.csv
projects_new/<project-name>/<source-tree>/...
```

For a source tree stored elsewhere, pass it explicitly:

```bash
python scripts/preprocess_promise.py \
  --dataset-name my-project-1.0 \
  --csv-path path/to/my-project.csv \
  --source-root path/to/java/source
```
