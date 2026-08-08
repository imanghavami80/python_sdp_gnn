# Multi-View Software Defect Prediction: Model Results

## Short summary

This project predicts whether a Java source file is defective. The final model
uses three graph views of each file:

1. **AST (Abstract Syntax Tree):** represents the syntax and structure of the
   source code.
2. **CFG (Control Flow Graph):** represents how the program statements can be
   executed.
3. **NDG (Network/Dependency Graph):** represents dependencies between files
   in the same software project.

The AST and CFG are first processed by their own Graph Neural Networks (GNNs).
Each of those GNNs creates one learned vector, called an *embedding*, for every
Java file. The final NDG GNN then uses the software metrics, AST embedding, and
CFG embedding as the features of each file node. It predicts the probability
that each file is defective.

## Data used

- 10 Java software projects from the PROMISE dataset
- 4,071 dataset records in total
- 3,758 records successfully matched to Java source files and evaluated
- 20 traditional software metrics per file, such as code size and complexity
- 3,758 AST graphs
- 3,758 CFG graphs, including 3,647 real CFGs extracted with Soot
- 10 project-level dependency graphs containing 25,392 typed dependency edges

## How the model was trained and tested

The evaluation used **strict nested Leave-One-Project-Out (LOPO)** testing.

This means the experiment was repeated 10 times. In each repetition:

1. One complete project was kept aside as the test project.
2. The other nine projects were used for training and model selection.
3. The AST GNN and CFG GNN were trained without seeing the test project.
4. Their file embeddings were added to the NDG nodes together with the software
   metrics.
5. The NDG GNN was trained without seeing the test project.
6. The trained model predicted defects for every file in the unseen test
   project.

Therefore, no files from the test project were used to train the AST, CFG, or
NDG models, choose training epochs, normalize metrics, or choose the final
classification threshold. This makes the evaluation suitable for measuring
cross-project prediction performance.

## Final result

### Accuracy

Across all 3,758 tested files, the model achieved an accuracy of **67.5%**.
In simple terms, it correctly classified about 68 out of every 100 files.

Because the projects have different sizes, the more meaningful cross-project
result is the average accuracy across the 10 projects: **62.2%**.

### Main metrics

| Metric | Result | Plain-language meaning |
| --- | ---: | --- |
| Accuracy across all files | **67.5%** | Percentage of all files classified correctly. |
| Average project accuracy | **62.2%** | Average accuracy when every project has equal importance. |
| Balanced accuracy | **56.0%** | Accuracy that gives equal importance to defective and clean files. |
| Precision | **54.0%** | When the model reports a defect, this is how often it is correct on average across projects. |
| Recall | **80.4%** | The model finds about 80 out of every 100 truly defective files. |
| F1-score | **62.2%** | A combined measure of precision and recall. |
| MCC | **0.137** | A correlation-style measure; 0 means no useful relationship, 1 means perfect prediction. |
| ROC-AUC | **0.706** | The model has a reasonable ability to rank defective files above clean files. |
| PR-AUC | **0.684** | Ranking quality focused on finding defective files. |
| Brier score | **0.242** | Probability-quality measure; lower is better. |

## What these results mean

The model is especially useful for **finding defective files**: its recall is
80.4%, so it identifies most files that really contain defects. This is useful
when the goal is to prioritize code inspection or testing.

Its performance changes between projects, which is expected in cross-project
software defect prediction because software systems have different coding
styles, sizes, histories, and defect distributions. The ROC-AUC of 0.706 and
PR-AUC of 0.684 show that the model has meaningful ranking ability, but the
moderate balanced accuracy and MCC show that there is still room for
improvement.

## One-sentence description for presentation

> I trained a multi-view GNN model in which AST and CFG GNNs create code
> embeddings for each Java file, and a final dependency-graph GNN combines
> those embeddings with software metrics to predict defects. Using strict
> cross-project testing on 10 PROMISE projects, the model achieved 67.5% overall
> accuracy, 62.2% average project accuracy, 80.4% recall, and 0.706 ROC-AUC.

## Result files

The detailed machine-generated results are stored in:

- `outputs/promise/final_ndg_nested_lopo/nested_lopo_summary.json`
- `outputs/promise/final_ndg_nested_lopo/fold_metrics.csv`
- `outputs/promise/final_ndg_nested_lopo/all_test_node_predictions.csv`
