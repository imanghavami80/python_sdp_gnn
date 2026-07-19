# Developer Documentation

This directory explains the implementation file by file. Start with the root
[`README.md`](../README.md) for installation and the shortest path to running
the pipeline. Use this documentation when changing, reviewing, or extending a
specific stage.

## Execution Order

| Order | Stage | Implementation | Documentation |
| --- | --- | --- | --- |
| 1 | PROMISE preprocessing | `scripts/preprocess_promise.py` | [Preprocessing](pipeline/preprocess_promise.md) |
| 2 | AST extraction | `scripts/extract_promise_ast.py` | [AST extraction](pipeline/extract_promise_ast.md) |
| 3 | CFG extraction | `scripts/extract_promise_cfg.py` | [CFG extraction](pipeline/extract_promise_cfg.md) |
| 3a | Soot backend | `scripts/soot_cfg_extractor.java` | [Soot backend](reference/soot_cfg_extractor.md) |
| 4 | NDG extraction | `scripts/extract_promise_ndg.py` | [NDG extraction](pipeline/extract_promise_ndg.md) |
| 5 | Final evaluation | `scripts/evaluate_ndg_nested_lopo.py` | [Nested LOPO](training/evaluate_ndg_nested_lopo.md) |

## Models

| Model | Implementation | Documentation |
| --- | --- | --- |
| AST GIN encoder | `src/thesis_project/models/ast_encoder.py` | [AST model](models/ast_encoder.md) |
| Edge-aware CFG GAT | `src/thesis_project/models/cfg_encoder.py` | [CFG model](models/cfg_encoder.md) |
| Multi-view relational NDG GAT | `src/thesis_project/models/ndg_encoder.py` | [NDG model](models/ndg_encoder.md) |

## Supporting Training Workflows

| Purpose | Implementation | Documentation |
| --- | --- | --- |
| Exploratory global AST embeddings | `scripts/generate_ast_embeddings.py` | [Global AST workflow](training/generate_ast_embeddings.md) |
| Standalone AST LOPO | `scripts/evaluate_ast_lopo.py` | [AST LOPO](training/evaluate_ast_lopo.md) |
| Standalone CFG LOPO | `scripts/evaluate_cfg_lopo.py` | [CFG LOPO](training/evaluate_cfg_lopo.md) |
| Shared NDG operations | `src/thesis_project/training/ndg.py` | [NDG training utilities](training/ndg_training.md) |
| Automated verification | `tests/` | [Testing](reference/testing.md) |

## Data Granularity

- PROMISE rows represent Java files/classes and contain one binary defect label.
- AST and CFG encoders produce one graph embedding per file.
- An NDG represents one complete project and each NDG node represents one file.
- The final task is node classification: one probability and embedding per file.

## Leakage Boundary

Only `evaluate_ndg_nested_lopo.py` is the final research evaluation. It trains
AST, CFG, and NDG models inside every outer project fold. Outputs from the
standalone AST and CFG scripts are useful for diagnostics, but must not replace
fold-specific embeddings in the final experiment.

