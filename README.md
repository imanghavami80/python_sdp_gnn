# Multi-View Software Defect Prediction

This project predicts whether a Java file is defective by combining four
sources of software information:

- **Software metrics:** file-level measurements from PROMISE, such as LOC, WMC,
  CBO, RFC, and complexity.
- **AST:** the syntactic structure of each Java file.
- **CFG:** the control-flow behavior of methods in each Java file.
- **NDG:** typed dependencies between files in the same project.

The final model performs **node-level prediction on each project NDG**. Every
NDG node represents one Java file and receives its metrics, AST embedding, and
CFG embedding. The output is one defect probability and one learned embedding
per file.

## Pipeline Overview

```text
PROMISE CSV + Java source code
              |
              v
       Preprocess and map files
              |
       +------+------+
       |             |
       v             v
   Extract AST   Extract CFG with Soot
       |             |
       v             v
   AST encoder    CFG encoder
       |             |
       +------+------+
              |
Software metrics + AST embedding + CFG embedding
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

The final evaluation uses strict nested Leave-One-Project-Out (LOPO). One whole
project is held out for testing, so files from that project cannot influence
training, normalization, or epoch selection.

## Repository Layout

```text
projects/                  PROMISE CSV files and Java project source code
scripts/                   Preprocessing, extraction, and evaluation commands
docs/                      File-by-file developer documentation
src/thesis_project/models/ AST, CFG, and NDG model implementations
src/thesis_project/training/ Shared leakage-safe training utilities
tests/                     Automated tests
tools/                     Java analysis dependencies used by CFG extraction
outputs/                   Generated datasets, graphs, tensors, and results
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

Each project directory under `projects/` contains one PROMISE CSV and the
corresponding Java source tree. For example:

```text
projects/
  log4j/
    log4j-1.1.csv
    <Java source directories>
```

The CSV must contain `name`, `bug`, and the expected 20 software metric
columns. `name` should identify the package-qualified Java class. The
preprocessor automatically discovers CSV files under `projects/*/*.csv`.

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

This compiles the Java projects with partial-error recovery and uses Soot to
build method-level statement CFGs. Method CFGs are aggregated into one CFG for
each file. Nodes describe statements and operations; typed edges describe
normal, branch, return, exception, loop, and switch flow.

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
final evaluator adds fold-specific AST and CFG embeddings during training.

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
4. Trains the AST and CFG encoders without the outer test project.
5. Generates fold-specific AST and CFG embeddings.
6. Combines metrics and available embeddings at each NDG file node.
7. Trains the relational NDG model and predicts the held-out project.
8. Repeats the process until every project has been tested once.

Main results:

```text
outputs/promise/final_ndg_nested_lopo/all_test_node_predictions.csv
outputs/promise/final_ndg_nested_lopo/ndg_node_embeddings.npy
outputs/promise/final_ndg_nested_lopo/ndg_node_embedding_index.csv
outputs/promise/final_ndg_nested_lopo/fold_metrics.csv
outputs/promise/final_ndg_nested_lopo/nested_lopo_summary.json
```

`all_test_node_predictions.csv` contains the final label, probability, and
prediction for every held-out file. `nested_lopo_summary.json` contains pooled
and macro-project accuracy, precision, recall, F1, ROC-AUC, and PR-AUC, together
with majority-class baseline results.

To run a quick pilot on one held-out project before the complete experiment:

```bash
python scripts/evaluate_ndg_nested_lopo.py \
  --test-project log4j-1.1 \
  --device cpu
```

## Optional Standalone Experiments

These commands help inspect individual encoders, but their saved embeddings are
not used by the strict final NDG evaluator:

```bash
# Exploratory AST embeddings from a global model
python scripts/generate_ast_embeddings.py

# Project-held-out AST evaluation
python scripts/evaluate_ast_lopo.py

# Project-held-out CFG evaluation, excluding placeholder CFGs
python scripts/evaluate_cfg_lopo.py
```

The final evaluator retrains both upstream encoders inside every outer fold. Do
not substitute globally trained embeddings in the final cross-project results,
because that would leak information from held-out projects.

## Model Summary

- **AST encoder:** GIN with node-type embeddings and attention pooling.
- **CFG encoder:** edge-aware GATv2 with node, statement, invocation, and CFG
  edge types, followed by attention pooling.
- **NDG encoder:** multi-view relational GATv2 operating at file-node level.

Attention weights can provide an approximate indication of influential nodes,
but they should not be treated as exact causal explanations.

## Adding Another Project

Place its PROMISE CSV and Java source code under one project directory, then
rerun the pipeline from preprocessing:

```text
projects/<project-name>/<dataset-name>.csv
projects/<project-name>/<source-tree>/...
```

For a source tree stored elsewhere, pass it explicitly:

```bash
python scripts/preprocess_promise.py \
  --dataset-name my-project-1.0 \
  --csv-path path/to/my-project.csv \
  --source-root path/to/java/source
```
