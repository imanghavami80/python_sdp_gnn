# Exploratory Global AST Embeddings

**Implementation:** `scripts/generate_ast_embeddings.py`

## Purpose

This workflow trains the AST GIN classifier over the complete AST dataset and
saves one fixed-size embedding per file. It is useful for encoder debugging,
attention inspection, visualization, and exploratory analysis.

It is not a valid source of features for final cross-project evaluation because
the random train/validation split can contain files from every project.

## Inputs

```text
outputs/promise/ast/graph_index.csv
outputs/promise/ast/node_type_vocab.json
outputs/promise/ast/tensors/
```

The lazy dataset loads each graph tensor only when requested, avoiding the need
to keep every AST in memory.

## Training Process

1. Validate index columns and tensor paths.
2. Build a stratified random file-level train/validation split.
3. Train `ASTGraphClassifier` with binary cross-entropy.
4. Monitor validation loss and stop after the configured patience.
5. Restore the best checkpoint.
6. Encode every indexed AST in deterministic index order.
7. Save embeddings, index alignment, history, split, model, and summary.

Structural normalization is enabled by default. Use
`--no-normalize-structural-features` only for an ablation.

## Run

```bash
python scripts/generate_ast_embeddings.py
```

Common controls:

```text
--epochs --batch-size --lr --weight-decay
--patience --hidden-dim --output-dim --num-layers
--dropout --seed --device --num-workers
```

## Outputs

```text
outputs/promise/embeddings/ast/ast_embeddings.npy
outputs/promise/embeddings/ast/ast_embedding_index.csv
outputs/promise/embeddings/ast/ast_encoder.pt
outputs/promise/embeddings/ast/ast_training_history.csv
outputs/promise/embeddings/ast/ast_train_val_split.json
outputs/promise/embeddings/ast/ast_embedding_summary.json
```

The embedding matrix row order is defined only by
`ast_embedding_index.csv`. Never join embeddings to files by assuming another
CSV has the same order.

## Expected Result

- One embedding row for every indexed AST graph.
- A checkpoint containing model configuration and learned parameters.
- Training and validation loss for every completed epoch.
- Summary metadata describing dimensions, split sizes, and best epoch.

## Appropriate Use

Use these embeddings for visualization or architecture debugging. Use strict
nested LOPO for final defect-prediction claims.

