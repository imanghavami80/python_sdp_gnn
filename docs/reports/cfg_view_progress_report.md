# Progress Report: Development of the Behavioral Graph View

**Thesis context:** Multi-view cross-project software defect prediction using
software metrics, AST, a behavioral graph, project-level NDG message passing,
and gated clustering features.

**Report date:** 10 September 2026

## Purpose

The thesis model requires three principal views: AST, a behavioral code graph,
and the project-level NDG. This work investigated how the behavioral view could
be made more informative while preserving the original thesis design. Five
versions were examined:

1. the original CFG;
2. CFG enriched with def-use edges;
3. a PDG experiment;
4. a redesigned, exception-aware CFG (CFG v2); and
5. the same rigorous CFG with improved bytecode coverage (CFG v3).

All completed prediction experiments used the same strict nested
Leave-One-Project-Out protocol, late fusion, gated k-means++ features, random
seed 42, and all 5,303 mapped files. A missing behavioral graph is represented
by a tagged placeholder and masked from the behavioral view; the corresponding
file remains in the AST, metrics, and NDG views.

## Graph Extraction Coverage

| Behavioral view | Raw extractor count | Verified valid coverage | Important interpretation |
| --- | ---: | ---: | --- |
| Original CFG | 4,945 / 5,303 reported | Not reliably measurable from that saved run | The extractor counted compiler-error stubs as successes, while Log4j coverage was 0 / 194. This value is not comparable with later verified counts. |
| CFG + def-use | 3,870 / 5,303 | 72.98% | First strict and credible coverage baseline; Log4j was repaired to 178 / 194. |
| PDG | 3,870 / 5,303 | 72.98% | Built from the same valid method bodies, so coverage was unchanged. |
| CFG v2 | 4,046 / 5,303 | 76.30% | Complete-project compilation recovered 176 additional valid files. |
| CFG v3 | 4,573 / 5,303 | 86.23% | Exact official release bytecode recovered another 527 valid files. |

The table does not indicate that valid coverage decreased. The original 93.25%
was a mixture of real graphs and false successes and must not be used as the
starting point of a coverage trend. Some ECJ-generated classes contained
methods whose only behavior was to throw an `Unresolved compilation problem`
error. They were structurally readable by Soot but were not faithful
implementations of the source program. Once this was detected, the first
comparable strict baseline was 3,870 graphs. Verified coverage then increased
monotonically: **3,870 -> 4,046 -> 4,573**.

## 1. Original CFG

### Design

The initial extractor partially compiled the mapped Java sources and used Soot
Jimple `ExceptionalUnitGraph` bodies. It generated statement nodes with
synthetic method `ENTRY` and `EXIT` nodes. Its relations were:

`CFG_NEXT`, `CFG_TRUE`, `CFG_FALSE`, `CFG_RETURN`, `CFG_EXCEPTION`,
`CFG_BACK`, `CFG_SWITCH_CASE`, and `CFG_SWITCH_DEFAULT`.

An edge-aware GATv2 encoder created one embedding for each available file CFG.

### Problems discovered

- Log4j had **0 / 194** available CFGs. Legacy source-level detection treated
  Javadoc tags as Java annotations and selected an unsuitable compiler mode;
  paths containing spaces were another reliability concern.
- The generic `CFG_NEXT` relation mixed fall-through and unconditional jumps.
- `CFG_EXCEPTION` did not distinguish a caught exception from an exception
  escaping the method.
- `CFG_BACK` was inferred from statement order. A backward bytecode edge is not
  by itself a sound definition of a natural loop.
- Most importantly, ECJ compiler-error method bodies were counted as genuine
  CFGs. Therefore, the high reported coverage was misleading.

### Prediction outcome

This version produced macro-project F1 **0.5160**, MCC **0.1387**, G-Mean
**0.3818**, ROC-AUC **0.6901**, and PR-AUC **0.5838**. It was a useful initial
baseline, but its extraction defect prevents the coverage figure from being
used as evidence of representation quality.

## 2. CFG with Def-Use Edges

### Motivation and implementation

The next experiment retained the CFG and added intraprocedural data-dependence
information. A definition of a Jimple local variable was connected to reachable
statements that used that value. The new relation was kept separate from normal
control-flow edges, allowing the GNN to learn a different representation for
data flow.

This change moved the view toward a code-property/program-dependence
representation without removing execution order or branches. At the same time,
the extraction pipeline became stricter: invalid compiler-error bodies were
excluded and the Log4j compilation problem was repaired.

### Coverage and limitations

The extractor produced **3,870 / 5,303** credible graphs. Log4j improved from
0 to **178 / 194**. The lower total compared with the original extractor mainly
reflects removal of falsely accepted compiler stubs, not a loss of valid
program behavior.

The def-use analysis was intraprocedural and dependent on successful bytecode
generation. Missing legacy dependencies still caused low coverage in Camel
(384 / 935) and Synapse (121 / 256). Mixing control-flow and data-flow edges in
one encoder also increased graph density and semantic heterogeneity.

### Prediction outcome

Compared with the original experiment, ranking changed slightly: ROC-AUC rose
from 0.6901 to **0.7020** and PR-AUC from 0.5838 to **0.5882**. However, F1 fell
to **0.4753**, MCC to **0.0965**, G-Mean to **0.3609**, and the Brier score
worsened from 0.2602 to **0.2886**. A single-seed result cannot prove the cause,
but it did not justify retaining the added complexity as the main view.

## 3. PDG Experiment

### Motivation and implementation

The CFG was temporarily replaced with a Program Dependence Graph to test a
stronger behavioral representation. The PDG combined:

- the existing execution-flow relations;
- def-use data-dependence edges; and
- control-dependence relations derived from branch and switch structure.

The encoder used separate edge-aware GATv2 branches for control and data
relations, followed by a learned node-wise gate. This was more principled than
treating every relation identically in one message-passing stack.

### Coverage and limitations

PDG coverage remained **3,870 / 5,303 (72.98%)**, because PDGs could only be
built for the same valid method bodies available to the stricter CFG + def-use
extractor. The PDG therefore added semantic relations but did not solve the
underlying compilation and coverage problem.

The representation was also substantially more complex, while CFG—not PDG—is
the required behavioral view in the thesis design. Maintaining two graph
families, two encoders, and additional extraction logic made the codebase harder
to audit and made attribution of model changes more difficult.

### Prediction outcome and decision

The PDG experiment achieved macro-project F1 **0.4753**, MCC **0.0870**,
G-Mean **0.2963**, ROC-AUC **0.6946**, PR-AUC **0.5852**, and Brier score
**0.3025**. It did not outperform the simpler original CFG overall. The PDG
implementation was therefore removed from the active pipeline, while its saved
experiment results were retained as a negative ablation.

## 4. Canonical Exception-Aware CFG (CFG v2)

### Redesign

After the PDG experiment, the behavioral view was simplified back to one
logically well-defined CFG. Def-use and PDG relations were removed from the
active model. CFG v2 retained only control-flow semantics and introduced 11
precise edge types:

`CFG_ENTRY`, `CFG_FALLTHROUGH`, `CFG_BRANCH_TRUE`, `CFG_BRANCH_FALSE`,
`CFG_GOTO`, `CFG_SWITCH_CASE`, `CFG_SWITCH_DEFAULT`, `CFG_RETURN`, `CFG_THROW`,
`CFG_EXCEPTION_HANDLER`, and `CFG_EXCEPTION_EXIT`.

The important correctness improvements were:

- separate fall-through and unconditional-goto relations;
- separate caught-exception and escaping-exception relations;
- explicit throw-to-exit semantics;
- exactly one semantic type for each transfer rather than duplicate back-edge
  labels;
- loop features derived from graph strongly connected components rather than
  statement ordering;
- rejection of ECJ unresolved-compilation method bodies;
- compilation of the complete project-version source tree so mapped classes
  can resolve sibling source dependencies;
- safer legacy Java source-level detection and ECJ argument-file path quoting;
  and
- a current, modular-JDK-compatible ECJ compiler while preserving legacy source
  modes.

### Coverage

CFG v2 produced **4,046 / 5,303** validated graphs, an increase of 176 over the
strict CFG + def-use/PDG extraction. It retained **178 / 194** Log4j graphs and
reported zero graph-validation errors. Its largest remaining coverage gaps were
Camel (384 / 935), Synapse (121 / 256), and Xerces (339 / 543).

### Prediction outcome

CFG v2 achieved F1 **0.5134**, close to the original result of 0.5160, while
balanced accuracy improved from 0.5580 to **0.5692** and G-Mean from 0.3818 to
**0.4237**. MCC was **0.1129**, ROC-AUC **0.6830**, PR-AUC **0.5707**, and Brier
score **0.2754**. Thus, v2 was not the best single-seed result on every metric,
but it was the most defensible representation: simpler than PDG, semantically
precise, validated, and aligned with the thesis.

## 5. Coverage-Improved Canonical CFG (CFG v3)

### Motivation and implementation

An audit showed that most remaining Camel and Synapse failures were caused by
missing legacy build dependencies. Retrying partial compilation could not
recover additional valid methods. Accepting ECJ error stubs or constructing an
approximate source graph would have inflated coverage without recovering real
behavior.

CFG v3 therefore keeps **the same node/edge schema, 11 relation types, feature
definitions, and GATv2 encoder as CFG v2**. The only controlled change is
bytecode recovery:

- for Camel 1.6 and Synapse 1.2, the extractor obtains bytecode from the matching
  official Apache binary releases;
- archive and JAR SHA-256 checksums are fixed and verified;
- exact-release classes take precedence over incomplete source-compiled stubs;
- extracted project JARs are cached, so subsequent runs add negligible setup
  time; and
- the source build remains the fallback for classes not supplied by a release
  JAR and for all other projects.

The release sources are the official [Apache Camel 1.6.0
archive](https://archive.apache.org/dist/camel/apache-camel/1.6.0/) and [Apache
Synapse 1.2 archive](https://archive.apache.org/dist/synapse/1.2/).

This is still a Soot/Jimple CFG extracted from real bytecode. It does not add a
second parser or fabricate control flow.

### Coverage improvement

CFG v3 produced **4,573 / 5,303** validated graphs with zero validation errors:

- total credible coverage increased from 76.30% to **86.23%**;
- Camel increased from 384 to **804 / 935**;
- Synapse increased from 121 to **228 / 256**;
- Log4j remained correctly covered at **178 / 194**; and
- placeholders fell from 1,257 to **730**.

This recovered **527 additional genuine CFGs** without changing graph semantics.
The 204 unavailable Xerces files are interfaces with no executable method body,
so generating behavioral CFGs for them would be conceptually incorrect. Many
other remaining placeholders are likewise interfaces, annotations, or files
without recoverable executable code.

The final v3 corpus contains 898,127 nodes, 1,753,339 typed control-flow edges,
and 50,842 methods.

### Prediction experiment result

The CFG-v3 experiment completed all 12 held-out-project folds and evaluated all
5,303 mapped files. It achieved macro-project recall **0.8480**, F1 **0.5183**,
ROC-AUC **0.6936**, PR-AUC **0.5777**, and Brier score **0.2465**. The result is
mixed: v3 improved ranking and calibration over v2 and produced slightly higher
F1, but its balanced accuracy, MCC, and G-Mean decreased. The model favored
sensitivity: its mean recall was high, while the specificity implied by mean
recall and balanced accuracy was only approximately 0.2173.

## Final Predictive Comparison

The following values are unweighted means and standard deviations across the 12
held-out projects. Bold values are the best mean in each row. Brier score is the
only metric for which lower is better.

| Metric | Original CFG | CFG + def-use | PDG | CFG v2 | CFG v3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Accuracy | **0.5530 ± 0.2506** | 0.5087 ± 0.2111 | 0.4633 ± 0.2107 | 0.5198 ± 0.1951 | 0.5428 ± 0.2401 |
| Balanced accuracy | 0.5580 ± 0.0726 | 0.5638 ± 0.1126 | 0.5490 ± 0.0889 | **0.5692 ± 0.1030** | 0.5326 ± 0.1031 |
| Precision | **0.5133 ± 0.3099** | 0.4987 ± 0.3338 | 0.4428 ± 0.3448 | 0.4851 ± 0.3062 | 0.4464 ± 0.3361 |
| Recall | 0.7816 ± 0.3051 | 0.7317 ± 0.3018 | 0.8032 ± 0.3203 | 0.7571 ± 0.2313 | **0.8480 ± 0.3011** |
| F1 | 0.5160 ± 0.2699 | 0.4753 ± 0.2491 | 0.4753 ± 0.2685 | 0.5134 ± 0.2115 | **0.5183 ± 0.3002** |
| MCC | **0.1387 ± 0.1138** | 0.0965 ± 0.1075 | 0.0870 ± 0.1134 | 0.1129 ± 0.1157 | 0.0785 ± 0.1272 |
| G-Mean | 0.3818 ± 0.1951 | 0.3609 ± 0.2705 | 0.2963 ± 0.2499 | **0.4237 ± 0.2341** | 0.2307 ± 0.2586 |
| ROC-AUC | 0.6901 ± 0.1442 | **0.7020 ± 0.1338** | 0.6946 ± 0.1312 | 0.6830 ± 0.1206 | 0.6936 ± 0.1493 |
| PR-AUC | 0.5838 ± 0.2749 | **0.5882 ± 0.2731** | 0.5852 ± 0.2711 | 0.5707 ± 0.2768 | 0.5777 ± 0.2827 |
| Brier score ↓ | 0.2602 ± 0.1155 | 0.2886 ± 0.0999 | 0.3025 ± 0.1068 | 0.2754 ± 0.0941 | **0.2465 ± 0.0937** |

### Focused comparison: CFG v3 versus CFG v2

This is the cleanest comparison because both versions use the same CFG schema
and encoder. The change is the higher valid bytecode coverage in v3.

| Metric | v3 − v2 mean | Projects better/tied/worse with v3 | Paired Wilcoxon p-value |
| --- | ---: | ---: | ---: |
| Accuracy | +0.0230 | 8 / 0 / 4 | 0.5186 |
| Balanced accuracy | −0.0366 | 4 / 0 / 8 | 0.1514 |
| Precision | −0.0387 | 4 / 0 / 8 | 0.2036 |
| Recall | +0.0909 | 7 / 2 / 3 | 0.2754 |
| F1 | +0.0049 | 7 / 0 / 5 | 0.4697 |
| MCC | −0.0345 | 4 / 0 / 8 | 0.2334 |
| G-Mean | −0.1930 | 4 / 0 / 8 | 0.1099 |
| ROC-AUC | +0.0107 | 9 / 0 / 3 | 0.3013 |
| PR-AUC | +0.0070 | 7 / 0 / 5 | 0.7334 |
| Brier score ↓ | −0.0288 | 8 / 0 / 4 | 0.4697 |

The paired tests are exploratory and uncorrected. None of the v3-versus-v2
differences reached p < 0.05. With only one training seed and 12 heterogeneous
projects, they should not be interpreted as proof of superiority or equivalence.

## Current Conclusion

No version dominates every metric. The original CFG has the highest accuracy,
precision, and MCC, but its extraction coverage was contaminated by invalid
compiler stubs, so it is not an acceptable final implementation. CFG + def-use
has the highest ROC-AUC and PR-AUC, but lower F1, MCC, and calibration quality.
The PDG adds substantial complexity without a convincing benefit. CFG v2 gives
the best balanced accuracy and G-Mean and remains the strongest threshold-based
balanced classifier in this single run.

CFG v3 is best in recall, F1, and Brier score and provides by far the strongest
verified extraction coverage. Relative to v2, it improves accuracy, recall, F1,
ROC-AUC, PR-AUC, and calibration, but reduces balanced accuracy, precision, MCC,
and G-Mean. Therefore, higher coverage improved several aspects of the model but
did **not** produce a universal predictive improvement.

CFG v3 should remain the current behavioral representation because it is the
most complete semantically valid CFG and does not add PDG complexity. However,
the modeling or validation-threshold behavior should be investigated before
claiming it is the best predictor. The next evidence should come from multiple
seeds and aggregate mean/variance, not selection of the most favorable run.
