# 01 — Documentation Reading Map

Every maintained Markdown document has a two-digit reading number in its title.
Filenames remain stable so links and IDE tabs continue to work. Read 00–13 for
the core pipeline, 14–18 for diagnostics and evaluation guidance, and 19–21 for
historical results.

The current system always uses CFG and late fusion. Its scenario controls are
cluster mode (none/simple/gated) and NDG structural features (off/on).
K-means++, GMM, and HDBSCAN remain algorithm settings within active clustering.
Defaults are no clustering and no structural features.

| Read | Document |
| --- | --- |
| 00 | [Multi-View Software Defect Prediction](../README.md) |
| 01 | [Documentation Reading Map](README.md) |
| 02 | [PROMISE Preprocessing](pipeline/preprocess_promise.md) |
| 03 | [AST Extraction](pipeline/extract_promise_ast.md) |
| 04 | [CFG Extraction](pipeline/extract_promise_cfg.md) |
| 05 | [Soot CFG Backend](reference/soot_cfg_extractor.md) |
| 06 | [NDG Extraction](pipeline/extract_promise_ndg.md) |
| 07 | [Handcrafted NDG Structural Features](features/ndg_structural_features.md) |
| 08 | [Cluster-Based NDG Features](features/cluster_features.md) |
| 09 | [AST Encoder](models/ast_encoder.md) |
| 10 | [CFG Encoder](models/cfg_encoder.md) |
| 11 | [NDG Encoder](models/ndg_encoder.md) |
| 12 | [Shared NDG Training Utilities](training/ndg_training.md) |
| 13 | [Strict Nested NDG LOPO Evaluation](training/evaluate_ndg_nested_lopo.md) |
| 14 | [Exploratory Global AST Embeddings](training/generate_ast_embeddings.md) |
| 15 | [Standalone AST LOPO Evaluation](training/evaluate_ast_lopo.md) |
| 16 | [Standalone CFG LOPO Evaluation](training/evaluate_cfg_lopo.md) |
| 17 | [Testing and Validation](reference/testing.md) |
| 18 | [Model Improvement Strategy](reference/model_improvement.md) |
| 19 | [Model Results and Historical Record](../MODEL_RESULTS.md) |
| 20 | [Short Report: Improvements to the CFG View](reports/cfg_view_progress_report_short.md) |
| 21 | [Progress Report: Development of the Behavioral Graph View](reports/cfg_view_progress_report.md) |
| 22 | [AST and NDG representation improvements](features/representation_improvements.md) |

The generated `outputs/` directory was removed during cleanup. Rebuild data
using document 00 before experiments. Historical reports retain earlier values;
their raw artifacts are absent. All final models are retrained inside the
project-held-out protocol; diagnostic embeddings must not be reused there.
