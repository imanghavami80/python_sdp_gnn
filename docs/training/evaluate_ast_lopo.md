# Standalone AST LOPO Evaluation

**Implementation:** `scripts/evaluate_ast_lopo.py`

## Purpose

This script evaluates the AST encoder in a cross-project setting and creates
one project-held-out AST embedding per file. It answers whether AST structure
alone learns a defect signal that transfers to another project.

These embeddings are diagnostic. The final NDG evaluator retrains the AST model
inside its own nested folds and does not load this output.

## Protocol

For each project:

1. Hold out every AST from that project.
2. Split files from the remaining projects into training and validation sets.
3. Train the AST classifier and select the best epoch using validation loss.
4. Encode the untouched project's files.
5. Place those embeddings into a combined matrix at their original index rows.
6. Repeat until every project has been held out.

The outer test project never participates in encoder training. The inner
validation split is file-level rather than project-level, so this workflow is
less strict than the final nested NDG protocol.

## Run

```bash
python scripts/evaluate_ast_lopo.py
```

Important options include model dimensions, epoch and patience controls,
`--seed`, `--device`, and `--no-clean`. Run the script with `--help` for the full
interface.

## Outputs

```text
outputs/promise/embeddings/ast_lopo/ast_embeddings.npy
outputs/promise/embeddings/ast_lopo/ast_embedding_index.csv
outputs/promise/embeddings/ast_lopo/fold_metrics.csv
outputs/promise/embeddings/ast_lopo/ast_embedding_summary.json
outputs/promise/embeddings/ast_lopo/folds/<project>/test_embeddings.npy
outputs/promise/embeddings/ast_lopo/folds/<project>/test_embedding_index.csv
outputs/promise/embeddings/ast_lopo/folds/<project>/training_history.csv
outputs/promise/embeddings/ast_lopo/folds/<project>/ast_encoder.pt
```

## Expected Result

- Every file receives an embedding from a model that did not train on its
  project.
- Fold metadata identifies the held-out project and selected checkpoint.
- Matrix rows remain aligned through the saved embedding index.

## Limitations

- It tests only the AST view.
- It is not the final multi-view node classifier.
- Fitted outputs from this workflow must not be reused as if they were nested
  inside final NDG model selection.

