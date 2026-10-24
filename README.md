# Four-View Software Defect Prediction

This project predicts defective Java files **within each project**. Each file has
four views: an AST embedding, a CFG embedding, an NDG embedding, and a learned
embedding of its PROMISE software metrics. The NDG uses features derived from
the file's typed dependencies, while metrics remain a separate view.
The four views meet only at the final fusion layer. See the
[model flow](docs/features/four_view_dependency_inputs.md) for a short explanation.

## Setup

Use Python 3.11+ and a Java JDK with `java` and `javac` on `PATH`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e . --no-deps
```

The input projects are read from `projects_new/`. Run the following commands
from the repository root, in order. The first four rebuild preprocessing and
all three graph representations; the last command trains AST, CFG, NDG, and
fusion models from scratch for each eligible project.

```bash
.venv/bin/python scripts/preprocess_promise.py --projects-root projects_new
.venv/bin/python scripts/extract_promise_ast.py
.venv/bin/python scripts/extract_promise_cfg.py
.venv/bin/python scripts/extract_promise_ndg.py
.venv/bin/python scripts/evaluate_ndg_within_project.py \
  --no-ndg-structural-features --device cpu --seed 43 \
  --output-dir outputs/promise/within_project/four_view_reproduction_seed43
```

The evaluator does not overwrite a nonempty result directory. Choose another
name when repeating an experiment. For a data check without training, run the
evaluator with `--preflight-only` and a separate output directory.

## Current reference baseline

The first full four-view run is saved in
`outputs/promise/within_project/four_view_scratch_seed43/`. It used seed 43,
no optional structural features, and completed all 12 projects (1,054 held-out
files). The model's **equal-project macro means** are:

| Metric | Baseline |
| --- | ---: |
| F1 | 0.6898 |
| MCC | 0.4443 |
| Balanced accuracy | 0.7307 |
| ROC AUC | 0.8562 |
| PR AUC | 0.7558 |
| Precision | 0.6241 |
| Recall | 0.8222 |

Use `model_macro_project_metrics` in that run's `within_project_summary.json`
as the reference for future changes. Its `baseline_macro_project_metrics` field
is only a training-majority-class dummy predictor, not an earlier neural model;
new runs label that field `majority_class_macro_project_metrics`.
Forrest has just six test files and F1 = 1.0, so its equal contribution makes
the macro average sensitive to a very small test set. A candidate improvement
should use the same project set, file splits, source inputs, and seed, then
report its per-project changes alongside the macro mean. One seed is preliminary
evidence; repeat matched seeds before making a strong performance claim.

Each project uses a grouped, approximately 60/20/20 training/validation/test
split. The full unlabeled dependency graph is visible, but only training labels
enter the loss. Validation selects the checkpoints and decision threshold.
Projects without sufficient class support are reported as ineligible.

The main result is `within_project_summary.json` under the chosen output
directory. Its `model_macro_project_metrics` gives every completed project one
equal vote. `project_metrics.csv` contains each project's test scores; the
summary does not compute pooled scores. Every project also saves
`file_view_embeddings.npz`, with aligned AST, CFG, NDG, and metrics embeddings
for all its files, plus the view mask and fusion weights.

See the [evaluation guide](docs/training/evaluate_ndg_within_project.md) for
output details and the [documentation map](docs/README.md) for extraction details.
