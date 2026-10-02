# 23 — Within-Project AST + CFG + NDG Evaluation

**Runner:** `scripts/evaluate_ndg_within_project.py`  
**Training:** `src/thesis_project/training/within_project.py`

This is the primary experiment protocol. A separate model is trained for each
project. No labeled files from other projects are used, and global exploratory
embeddings are not reused. Existing AST, CFG, and NDG extraction artifacts are
compatible; changing the evaluation scenario does not require re-extraction.

## What is being predicted

This is **transductive within-project file classification** on one project
snapshot. The full NDG, including test-node metrics and dependency edges, is
available during message passing. Only training-node labels enter the loss.
It evaluates withheld labels of known files, not future releases, future files,
or a previously unseen project. Unlabeled graph visibility is intentional and
must be reported in the thesis; it is not an inductive or temporal benchmark.
The graph-learning distinction follows the standard
[transductive GNN setting](https://arxiv.org/abs/1609.02907).

## Splitting and rare classes

Default split is approximately **60% train / 20% validation / 20% test**.
Rows pointing to the same resolved Java source path stay together. A file group
with any defective row is stratified as defective; groups are shuffled within
each class using the seed. Rounded validation/test counts are bounded to retain
at least one group of each class in each split. Consequently, realized node
fractions may differ slightly from the requested group fractions.

The same saved split is used by AST, CFG, and NDG. Hyperparameters,
early stopping, and threshold selection never use test labels. Labels are used
to construct a stratified split, after which the split is fixed. Both class
supports must be nonzero in every partition. Projects with fewer than three
source-file groups per class are explicitly skipped, without resampling their
minority class across partitions. In the current corpus, Forrest has only two
defective files and Xalan only one clean file, so neither supports this protocol.
The summary lists skipped projects and reasons. Available AST and CFG train/val
graphs must also contain both classes; otherwise that project is reported skipped.

Every selected project is audited **before any training** and appears in
`project_eligibility.csv`, including class counts and the exact exclusion reason.
Valid preprocessing and sufficient labeled examples for evaluation are separate
requirements. Duplicating a rare file across splits would contaminate evaluation.
Changing split percentages cannot solve Xalan's single clean example.

This grouping follows the general principle of
[non-overlapping groups in model evaluation](https://scikit-learn.org/stable/modules/cross_validation.html).
It prevents the same source file appearing in multiple partitions; identical
code copies under different paths are not automatically deduplicated. Random
splits can still benefit from similarity between related files.

## Training and leakage boundaries

1. Train AST and CFG separately using only that project's training files.
   Select each checkpoint by validation loss. Missing graph views remain masked.
   Test labels are replaced with placeholders before upstream loaders are used.
2. Freeze these checkpoints and encode every available file in the project.
   Graph-local AST/CFG normalizations remain deterministic. The extraction-time
   categorical vocabularies are shared, label-free artifacts.
3. Fit median imputation and scaling on training nodes only. All-training-missing
   metric coordinates are set to zero for every node; constant training scales
   are protected against division by zero. Structural-feature scaling, if enabled,
   also uses training nodes only. Persist all fitted parameters.
4. Run NDG message passing over the full project graph. Compute BCE **only on
   training indices**, with the positive weight calculated from training labels.
   Test labels are NaN in the graph supplied to the trainer; they are never part
   of the loss, validation, or class-weight calculation.
5. Select the NDG checkpoint using validation-node loss and select an F1 threshold
   from that checkpoint's validation probabilities. Evaluate that same checkpoint
   on test nodes once. There is **no final retraining or threshold transfer**.

Validation is reused for early stopping and threshold selection, as in an ordinary
train/validation/test setup. Test results remain the evaluation evidence. A single
split is not proof of robustness; matched additional seeds can be run later.

## Optional structural features

Optional structural descriptors are calculated from the full unlabeled graph;
that is consistent with the declared transductive protocol. Enable them only
with matching extracted structural artifacts.

## Run

From the repository root:

Check the data and environment without training first (use an empty directory):

```bash
.venv/bin/python scripts/evaluate_ndg_within_project.py \
  --preflight-only --device cpu --seed 42 \
  --output-dir outputs/promise/within_project/preflight_seed42
```

Run training:

```bash
.venv/bin/python scripts/evaluate_ndg_within_project.py \
  --no-ndg-structural-features \
  --device cpu --seed 42 \
  --output-dir outputs/promise/within_project/baseline_audited_seed42
```

`--project ant-1.7` limits execution to a project and can be repeated.
`--validation-fraction` and `--test-fraction` default to 0.2 each. Standard
training/device/batch-size options are available through `--help`. Existing
nonempty result directories are rejected. Keep the seed, split fractions,
inputs, and model settings fixed when comparing methods.

Use `--require-all-projects` to fail before training if any requested project
cannot support the protocol. This does not force invalid splits. Default runs
evaluate the eligible projects and explicitly report the rest.

### Runtime and reproducibility

Epoch limits (50 for AST/CFG and 100 for NDG) are **maximums**, not fixed budgets.
Each stage stops after 10 epochs without sufficient validation-loss improvement
and restores the best checkpoint. Selected epoch and total executed epochs are
different; both are now recorded. AST/CFG timings include fitting and embedding
extraction; NDG timing covers fitting and validation threshold selection. Project
elapsed time also includes preprocessing, evaluation and output writing. Overall
elapsed time includes input loading, preflight and fingerprinting, ending just
before aggregate report generation.

Each model learns from roughly 60% of **one project's** files. A fast run is
therefore plausible and is not a reason to increase epochs. The completed
`baseline_seed42` histories contain 12–50 AST, 14–50 CFG and 11–24 NDG epochs;
that older runner did not save wall-clock durations.

The runner records Python/package versions, actual device and thread counts,
stage seeds, source hashes, and input hashes including referenced graph/tensor
contents. PyTorch deterministic algorithms are required by default; unsupported
operations fail instead of silently compromising this setting. The CPU command
is the locally tested path. `--no-deterministic` explicitly relaxes this setting
and is recorded. Reproducibility across package versions or hardware is not
guaranteed ([PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)).
Training-only normalization follows
[scikit-learn's leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).

Keep existing completed runs immutable. Compare methods with the same seed,
inputs, settings and saved file memberships; a single seed remains preliminary
evidence. Do not choose changes by repeatedly optimizing the reported test scores.

## Outputs

- `within_project_summary.json`: completed/skipped projects, macro-project and
  pooled test metrics, majority-class baseline metrics, and protocol description.
- `project_metrics.csv` and `all_test_node_predictions.csv`: only held-out tests.
- `projects/<project>/split.csv` and `split.json`: reproducible node/file membership.
- Per-project selection histories, validation probabilities, test embeddings,
  and AST/CFG/NDG checkpoints.
- NDG checkpoint: selected threshold, preprocessing parameters, node ordering,
  split indices, and exact model-input tensors for reproducing predictions.
- `project_eligibility.csv`: every selected project, class support, split sizes,
  available upstream views for eligible projects, and reasons for exclusions.
- `project_metrics.csv`: also includes executed/selected epochs and stage timings.
- `run_manifest.json`: arguments, stage seeds, environment, source hashes and
  hashes of indices/vocabularies plus selected projects' referenced graph/tensor
  artifacts. Keep inputs unchanged while a run is executing.

Macro metrics weight completed projects equally. Pooled metrics weight files
and combine probabilities from different models. The majority classifier uses
training prevalence only. Compare runs with identical file splits and input artifacts.

The one-epoch integration pilot completed all ten eligible projects; two were
skipped as described above. This verifies execution, not predictive improvement.
