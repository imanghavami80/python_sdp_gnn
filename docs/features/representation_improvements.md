# 22 — AST and NDG Representation Improvements

## Scope

CFG, late fusion, and the six clustering/structural-feature scenarios remain
unchanged. Each view has one supported implementation, without numbered names.
The active inputs are `ast/`, `cfg/`, and `ndg/` under `outputs/promise/`.
Historical experiment results are preserved but do not select alternative extractors.

During naming cleanup, previous input directories were moved to
`outputs/archive/representation_cleanup/` for recovery, and active AST, NDG and
structural features were regenerated at their default paths. CFG tensors were
unchanged; only its construction metadata was renamed. Archived indexes retain
original absolute paths and are not directly runnable training inputs.

## AST

The extractor retains all parser node categories, including control wrappers and
array selectors, instead of removing intermediate nodes. Class initializer lists
become explicit block nodes. Each child carries its incoming parent attribute
(for example condition, then/else statement, left/right operand, arguments), its
index within that attribute, and its sibling count. Thus child roles remain
unambiguous in the tree. The role is encoded on the child because it has one
parent; this does not introduce a separate relation-aware GNN.

The compact 20-column tensor contains:

- depth, degree, and identifier-presence flag;
- operator, literal-category, and incoming-child-role IDs;
- log sibling index and relative sibling index;
- log counts for six prefix and six postfix unary operators.

The three categorical IDs are embedded separately (eight dimensions each), not
treated as ordinal measurements. Numeric normalization leaves these IDs intact.
Literal categories distinguish null, boolean, zero, one, other integer, floating
point, string, and character values. Raw names/strings do not form vocabularies.
Unary operator counts retain categories and multiplicity, but not arbitrary
ordering of mixed prefix sequences. This is still an abstraction of source code.

`syntax_schema.json` records column names and fixed vocabularies. Loaders require
20 columns and the matching node vocabulary. The encoder always uses syntax
embeddings together with the GIN layers and attention pooling. Older three-column
inputs and their checkpoints are not supported; regenerate inputs and retrain.

## NDG

`scripts/ndg_resolution.py` indexes declared project types, member types, fields,
method arities, return types, and project inheritance before walking bodies.
Lexical scope stacks replace the previous whole-method symbol dictionary.
Parameters and unknown local types still shadow fields. Local declarations take
effect at their initializer and expire at their enclosing block; loops, catches,
resources, lambdas, and synchronized blocks have explicit scope handling.

The scope rules follow the Java Language Specification's
[scope and shadowing definitions](https://docs.oracle.com/javase/specs/jls/se7/html/jls-6.html).
Explicit `this`, indexed field chains, static imports, project inheritance, and
unambiguous declared method-return chains resolve receivers. A generic argument
is never substituted for an unresolved container type, and an array is not
treated as its component instance. Arbitrary globally unique simple-name and
dotted-prefix guesses have been removed. Member types map back to file nodes.

This is conservative source analysis, not compiler-equivalent symbol binding or
whole-program points-to analysis. It does not fully resolve Java overloads by
argument types, external hierarchies, generic substitutions, local/anonymous
class bodies, or qualified inner construction. Methods are matched by name and
arity (including varargs); an ambiguous return type stops chain propagation.
Unresolved types/calls and unsupported bodies appear in each graph's
`unresolved_dependencies`, with counts in the summary. Many records legitimately
refer to libraries outside the mapped project. They are not inserted as guessed
project edges and are not counted as graph-validation errors.

## Reproducible extraction

Run from the repository root; these commands refresh the default input directories:

```bash
.venv/bin/python scripts/extract_promise_ast.py
.venv/bin/python scripts/extract_promise_cfg.py
.venv/bin/python scripts/extract_promise_ndg.py
.venv/bin/python scripts/extract_ndg_structural_features.py
```

The structural-feature step is needed only for scenarios that enable it.
Structural indexes fingerprint their source graph JSON; the evaluator rejects
missing or mismatched fingerprints. CFG construction and NDG resolution are
recorded as descriptive contracts, not numbered variants. Input-index hashes
remain in experiment manifests for traceability.

## Validation and experiments

### Verification on the current corpus

- AST: 5,303 mapped files, with the same 11 masked fallback graphs; 1,724,239
  nodes versus 1,662,267 before (about 3.7% more). Extraction took about 40 seconds.
- NDG: 12 project graphs containing the same 5,303 file nodes; 38,051 typed edges
  versus 36,986 before. Of the old/new edge sets, 3,208 old edges were removed and
  4,273 new edges were added. Extraction took about 18 seconds, with no parse
  failures or graph-validation issues. The 150,970 unresolved dependency records
  include external-library references and unsupported constructs, not just errors.
- All AST tensors passed finite-value, shape, categorical-range and tree-index
  checks. NDG file ordering, defect labels and metric inputs match the old graphs.
- Recomputed 38-column NDG structural features took about one second; this
  step uses already extracted project topology, not Java parsing.
- The test suite covers these contracts, including lexical-scope counterexamples, AST syntax
  propagation into the encoder, and rejection of stale structural features.
- A one-epoch CPU nested-LOPO pilot completed for Log4j using both new views,
  gated k-means clustering and structural features. This is an integration check,
  not a performance comparison. Full training runtime and predictive improvement
  remain to be measured; extraction timings are local observations, not guarantees.

### Experiment commands

Every scenario uses the same AST, CFG, and NDG implementations with late fusion.
Only clustering mode and handcrafted NDG features vary. For example, run without
either optional feature:

```bash
.venv/bin/python scripts/evaluate_ndg_nested_lopo.py --cluster-mode none --no-ndg-structural-features --device cpu --seed 42 --output-dir outputs/promise/experiments/current_inputs_no_optional_features
```

Then test gated clustering with handcrafted NDG features:

```bash
.venv/bin/python scripts/evaluate_ndg_nested_lopo.py --cluster-mode gated --cluster-method kmeans --ndg-structural-features --device cpu --seed 42 --output-dir outputs/promise/experiments/current_inputs_gated_structural
```

See [the main README](../../README.md#experiment-scenarios) for all six scenarios.
Use a fresh output directory for each run; completed results are never overwritten.
The two examples change both optional features, so use the remaining scenarios
to isolate the contribution of each. Previous single-view representation ablation
commands are no longer supported by the current model.

Compare macro-project metrics, per-project changes, and multiple seeds. Historical
results remain historical; do not relabel them as runs of the current inputs.
Extraction validity and pilot completion do not establish a predictive improvement.
The transferred validation threshold remains a calibration limitation.
