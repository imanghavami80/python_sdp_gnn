# Soot CFG Backend

**Implementation:** `scripts/soot_cfg_extractor.java`

## Purpose

This Java helper is the semantic backend for CFG extraction. The Python CFG
script compiles it, invokes it once per project, parses its tab-separated output,
adds file-level features, validates graphs, and writes the final dataset.

The helper is not intended to be run directly during normal pipeline use.

## Inputs

At runtime it receives:

- A directory containing partially compiled project classes.
- The project classpath.
- The mapped class names that need CFG extraction.
- A path for its tab-separated output.

Soot is configured to tolerate missing external dependencies through phantom
references, which is necessary for legacy PROMISE projects. Its first backend
argument is a bytecode classpath rather than a single class directory. This
lets the Python stage place checksum-verified, exact-release Camel/Synapse JARs
before partial ECJ output, so an invalid compiler-error method is not accepted
when valid official bytecode is available.

## Method Graph Construction

For every concrete method or constructor body:

1. Load the Soot Jimple body.
2. Build an `ExceptionalUnitGraph`.
3. Create explicit synthetic ENTRY and EXIT nodes.
4. Create one node for each Jimple unit/statement.
5. Connect ENTRY to graph heads.
6. Convert normal and exceptional successors to typed edges.
7. Connect returns and terminal paths to EXIT.
8. Distinguish caught exception flow from exceptions escaping the method.
9. Emit method, node, and edge records.

The Python wrapper aggregates all emitted methods for one class/file.

## Statement Semantics

The helper assigns:

- Broad node types such as condition, switch, return, throw, and statement.
- Statement kinds such as assignment, invocation, identity, branch, return,
  monitor, NOP, and throw.
- Invocation kinds such as static, virtual, interface, special, and dynamic.
- Binary instruction flags for calls, field and array access, allocation, casts,
  arithmetic, comparisons, nulls, strings, and numeric constants.

These values are derived from Soot/Jimple objects, not from regular-expression
inspection of source text.

## Edge Semantics

The backend distinguishes entry, fall-through, true and false branches, goto,
switch targets, returns, explicit throws, caught exceptions, and escaping
exceptions. Python maps emitted names to stable IDs in `edge_type_vocab.json`.

## Interchange Format

The helper emits escaped tab-separated records. Record families include method,
node, edge, class failure, and method failure records. This format keeps the Java
component independent from Python JSON and NumPy dependencies.

If the format changes, update `parse_tsv` in `extract_promise_cfg.py` and add or
update behavioral graph tests in the same change.

## Failure Handling

- A class load/body failure is emitted rather than terminating the project run.
- Method-level failures remain associated with their class and signature.
- Compiler-generated unresolved-problem method bodies are rejected.
- The Python wrapper decides whether a real graph is sufficient or a tagged
  placeholder is required.

## Extension Rules

- Use Soot IR types for new semantic flags.
- Preserve ENTRY and EXIT invariants.
- Keep record fields tab-safe through the escaping helper.
- Maintain backward agreement between Java output columns and Python parsing.
- Add new relation names to the Python vocabulary before emitting them.
