# Short Report: Improvements to the CFG View

## Objective

The thesis model combines software metrics, AST, CFG, and project-level NDG
information for cross-project defect prediction. The behavioral code view was
revised several times to improve its correctness, information content, and
coverage without changing the main thesis architecture.

All completed model comparisons used strict nested Leave-One-Project-Out
evaluation, late fusion, gated k-means++, random seed 42, and the same 5,303
mapped files.

## Important Coverage Clarification

The original extractor reported 4,945 available CFGs, but this was not verified
coverage. It mistakenly accepted some ECJ-generated methods whose only behavior
was an `Unresolved compilation problem` error. These class files could be read
by Soot, but they did not represent the real program behavior. At the same time,
the extractor failed on every Log4j file.

Therefore, the original 4,945 count cannot be compared directly with the later
strict counts. After false successes were rejected, valid coverage improved as
follows:

| Version | Verified graphs | Coverage | Main change |
| --- | ---: | ---: | --- |
| CFG + def-use | 3,870 / 5,303 | 72.98% | First strict baseline; Log4j repaired to 178 / 194. |
| PDG | 3,870 / 5,303 | 72.98% | Added data and control dependence but used the same valid bytecode. |
| CFG v2 | 4,046 / 5,303 | 76.30% | Correct exception-aware CFG and improved complete-project compilation. |
| CFG v3 | 4,573 / 5,303 | 86.23% | Recovered Camel and Synapse using verified exact-release bytecode. |

Thus, credible coverage increased from **3,870 to 4,046 and then to 4,573**.

## Versions Tested

### Original CFG

**Why it was used:** It provided the first practical behavioral view required by
the thesis and represented execution flow using Soot bytecode analysis.

**Difference from the initial thesis design:** This was the first implemented
CFG, so it established the baseline rather than replacing an earlier version.

The original Soot/Jimple CFG represented execution order, branches, returns,
exceptions, loops, and switch flow. However, some relations were too general:
fall-through and unconditional jumps were mixed, caught and escaping exceptions
were not distinguished, and loop edges were inferred from statement order.
Log4j had 0/194 coverage, and compiler-error stubs inflated the overall count.

It achieved macro F1 0.5160, MCC 0.1387, ROC-AUC 0.6901, and PR-AUC 0.5838, but
its extraction coverage was not trustworthy.

### CFG with Def-Use

**Why we moved to it:** A CFG shows where execution may go, but not how values
produced by one statement are consumed by later statements. Def-use edges were
tested to provide this missing data-flow information.

**Difference from the original CFG:** It retained the original control-flow
relations but added definition-to-use edges as a separate relation. The
extraction pipeline also began rejecting compiler-error bodies and repaired the
Log4j compilation problem.

Definition-to-use edges were added as a separate relation so the graph included
local variable data flow as well as control flow. Compiler-error bodies were
rejected and Log4j coverage increased to 178/194.

Def-use slightly improved ROC-AUC to 0.7020 and PR-AUC to 0.5882, but F1 fell to
0.4753 and MCC to 0.0965. The added complexity was therefore not clearly
beneficial.

### PDG

**Why we moved to it:** The def-use experiment only added data flow to a CFG. A
PDG was tested to model both data dependence and the statements whose execution
depends on conditions, producing a more complete dependence representation.

**Difference from CFG + def-use:** It added explicit control-dependence edges
and replaced the single mixed encoder with separate control and data GATv2
branches combined by a learned gate.

The PDG combined execution flow, def-use edges, and control-dependence edges. A
more complex encoder processed control and data relations separately and fused
them through a learned gate. Coverage remained 3,870 files because PDG extraction
depended on the same compiled method bodies.

The PDG achieved F1 0.4753 and MCC 0.0870 and did not outperform the simpler CFG
overall. It also increased code and model complexity, so it was removed from the
active pipeline and retained only as a historical ablation.

### CFG v2

**Why we moved to it:** PDG increased complexity but did not improve the main
prediction results, and CFG is the behavioral view required by the thesis. A
smaller and more semantically correct CFG was therefore preferable.

**Difference from PDG:** All def-use and control-dependence relations and the
two-branch PDG encoder were removed. They were replaced by one CFG encoder over
11 precise normal and exceptional control-flow relations, together with stricter
compilation and validation.

The behavioral view was simplified back to one precise CFG. CFG v2 used 11
distinct relations for entry, fall-through, true/false branches, goto, switch,
return, throw, caught exceptions, and escaping exceptions. Loop features were
computed from graph cycles, compiler-error bodies were rejected, and the whole
project source tree was compiled to resolve sibling classes.

Coverage increased to 4,046 files with zero validation errors. It achieved F1
0.5134, balanced accuracy 0.5692, and G-Mean 0.4237. It was selected because its
semantics were clearer and more defensible than the def-use and PDG versions.

### CFG v3

**Why we moved to it:** CFG v2 was logically correct, but missing legacy
dependencies still left major coverage gaps in Camel and Synapse. The next goal
was to recover valid graphs without weakening CFG correctness.

**Difference from CFG v2:** The graph schema, features, and model are unchanged.
Only bytecode acquisition changed: exact, checksum-verified official binaries
are preferred for Camel and Synapse, with source-compiled bytecode as fallback.

CFG v3 keeps the same graph schema and model as v2. Its only change is improved
bytecode recovery. For Camel 1.6 and Synapse 1.2, exact official Apache release
JARs are downloaded, checksum-verified, and placed before incomplete compiler
stubs. No approximate or fabricated graph is created.

Coverage increased to 4,573/5,303 files: Camel improved from 384 to 804 graphs,
Synapse from 121 to 228, and Log4j remained at 178/194. The remaining 204 Xerces
placeholders are interfaces without executable method bodies. All generated v3
graphs passed validation.

CFG v3 achieved recall 0.8480, F1 0.5183, ROC-AUC 0.6936, PR-AUC 0.5777, and
Brier score 0.2465. It improved several results over v2, but reduced balanced
accuracy, precision, MCC, and G-Mean.

## Final Results Comparison

The table reports the unweighted mean across 12 held-out projects. Bold denotes
the best mean; lower Brier score is better.

| Metric | Original CFG | CFG + def-use | PDG | CFG v2 | CFG v3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Accuracy | **0.5530** | 0.5087 | 0.4633 | 0.5198 | 0.5428 |
| Balanced accuracy | 0.5580 | 0.5638 | 0.5490 | **0.5692** | 0.5326 |
| Precision | **0.5133** | 0.4987 | 0.4428 | 0.4851 | 0.4464 |
| Recall | 0.7816 | 0.7317 | 0.8032 | 0.7571 | **0.8480** |
| F1 | 0.5160 | 0.4753 | 0.4753 | 0.5134 | **0.5183** |
| MCC | **0.1387** | 0.0965 | 0.0870 | 0.1129 | 0.0785 |
| G-Mean | 0.3818 | 0.3609 | 0.2963 | **0.4237** | 0.2307 |
| ROC-AUC | 0.6901 | **0.7020** | 0.6946 | 0.6830 | 0.6936 |
| PR-AUC | 0.5838 | **0.5882** | 0.5852 | 0.5707 | 0.5777 |
| Brier score ↓ | 0.2602 | 0.2886 | 0.3025 | 0.2754 | **0.2465** |

These are single-seed results. The original CFG metrics remain useful as a
historical baseline, but its unreliable graph coverage means it cannot be the
final extractor.

## Current Conclusion

The PDG and def-use experiments showed that adding relations does not
automatically improve prediction. CFG v2 provided the clearest control-flow
semantics, and CFG v3 retained that design while substantially improving valid
coverage.

No method won every metric. CFG v2 retained the best balanced accuracy and
G-Mean. CFG v3 obtained the best recall, slightly best F1, and best Brier score,
but lower MCC and G-Mean. Thus, the additional valid CFGs improved ranking,
recall, F1, and calibration relative to v2, but did not improve every
classification measure. CFG v3 remains the current behavioral view because it
combines the strongest verified coverage with the most defensible CFG design;
multiple-seed evaluation is required before making a final performance claim.
