# PROMISE Preprocessing

**Implementation:** `scripts/preprocess_promise.py`

## Purpose

This is the first pipeline stage. It validates PROMISE metric CSV files,
converts defect counts to binary labels, transforms metrics, maps dataset rows
to Java source files, writes per-project datasets, and creates one combined
multi-project dataset.

## Inputs

By default, projects are discovered from:

```text
projects/<project>/<dataset>.csv
projects/<project>/<Java source tree>
```

Each CSV must contain a unique, non-empty `name`, a `bug` count, and these 20
metrics:

```text
wmc dit noc cbo rfc lcom ca ce npm lcom3 loc dam moa mfa cam ic cbm amc max_cc avg_cc
```

## Processing

1. Discover all project CSV files or read explicit project arguments.
2. Validate column names and reject duplicate class names.
3. Convert every metric to numeric form.
4. Convert `bug` to `0` when the count is zero and `1` otherwise.
5. Apply `log1p` to non-negative metrics.
6. Index Java files by package-qualified class name.
7. Match each PROMISE class to exactly one source file.
8. Write per-project and combined datasets and mapping reports.

Packaged classes are never resolved using only a simple class name. Ambiguous
qualified or simple names remain unmapped instead of being assigned to an
arbitrary source file.

## Leakage Policy

The default `--scaler none` is required for final nested LOPO evaluation. It
applies `log1p` but does not fit imputation or scaling over all projects. The
final evaluator fits median imputation and standard scaling using only the
training nodes of each fold.

`--scaler standard` and `--scaler minmax` are available for descriptive or
standalone analysis. They must not be used as input to the final cross-project
experiment because fitted preprocessing would see held-out projects.

## Run

```bash
python scripts/preprocess_promise.py
```

Useful options:

```bash
# Process one explicitly located project
python scripts/preprocess_promise.py \
  --dataset-name log4j-1.1 \
  --csv-path projects/log4j/log4j-1.1.csv \
  --source-root projects/log4j

# Inspect all options
python scripts/preprocess_promise.py --help
```

## Outputs

Combined outputs:

```text
outputs/promise/promise_preprocessed_log1p.csv
outputs/promise/promise_preprocess_summary.json
```

Per-project outputs:

```text
outputs/<dataset>/<dataset>_preprocessed_log1p.csv
outputs/<dataset>/<dataset>_name_to_source_mapping.csv
outputs/<dataset>/<dataset>_name_to_source_mapping.json
outputs/<dataset>/<dataset>_preprocess_summary.json
```

The combined CSV contains `dataset_name`, `name`, the 20 transformed metrics,
binary `bug`, `source_path`, and `match_strategy`.

## Expected Result

- Every input row is retained for dataset accounting.
- Mapped rows have a unique `source_path`.
- Unmapped or ambiguous rows have no source path and are not passed to graph
  extraction.
- The summary reports mapped and unmapped counts per project.

## Downstream Use

AST, CFG, and NDG extraction all consume
`outputs/promise/promise_preprocessed_log1p.csv` by default.

