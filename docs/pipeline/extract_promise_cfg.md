# CFG Extraction

**Implementation:** `scripts/extract_promise_cfg.py`

**Java backend:** `scripts/soot_cfg_extractor.java`

## Purpose

This stage builds statement-level Control Flow Graphs using compiled Java
bytecode and Soot. It does not infer control flow from regular expressions or
construct an approximate graph from the source AST. This is the project's only
behavioral code graph; it contains control-flow relations only.

Each method and constructor receives explicit `ENTRY` and `EXIT` nodes. Method
graphs are aggregated into one file-level CFG so they align with PROMISE
file-level labels.

## Prerequisites

- Run preprocessing first.
- Install a Java JDK so `java` and `javac` are available.
- Keep the bundled ECJ and Soot JARs under `tools/`.

## Extraction Architecture

```text
Complete project-version source tree
  -> ECJ 3.32 partial compilation with debug line information
  -> checksum-verified exact Apache release bytecode where configured
  -> Soot Jimple bodies
  -> method statement CFGs
  -> Python feature enrichment and validation
  -> one aggregated graph per file
```

ECJ uses partial-error recovery because the PROMISE projects are old and may
have missing build dependencies. A nonzero ECJ status is logged. Soot can still
recover real CFGs from the class files that were emitted.

Camel 1.6 and Synapse 1.2 need a stronger recovery path because missing legacy
dependencies cause ECJ to emit methods that only throw an unresolved-compilation
error. The extractor downloads the matching official Apache binary release,
verifies fixed SHA-256 checksums, caches only the required project JARs under
`build/promise_cfg_release_cache`, and gives those exact-version classes
precedence over partial ECJ output. This remains bytecode-derived CFG extraction;
no approximate source parser or fabricated control flow is used. Other projects
continue to use their source build.

## Node Data

Categorical IDs:

- CFG node role: `ENTRY`, `EXIT`, `STATEMENT`, `CONDITION`, `RETURN`, `THROW`,
  `LOOP`, `SWITCH`, `CATCH`, or `FINALLY`.
- Statement kind: assignment, invocation, identity, branch, switch, return,
  throw, monitor, NOP, and related categories.
- Invocation kind: static, virtual, interface, special, dynamic, or no call.

The numeric tensor contains source-position information, operation flags, and
structural roles such as degrees, branch/join/terminal status, normalized method
position, method size, and loop membership.

## Edge Types

```text
CFG_ENTRY
CFG_FALLTHROUGH
CFG_BRANCH_TRUE
CFG_BRANCH_FALSE
CFG_GOTO
CFG_SWITCH_CASE
CFG_SWITCH_DEFAULT
CFG_RETURN
CFG_THROW
CFG_EXCEPTION_HANDLER
CFG_EXCEPTION_EXIT
```

Each possible transfer has one typed edge. Caught exceptions target a `CATCH`
node; exceptions that leave a method target its synthetic `EXIT`. Loop
membership is computed from strongly connected components, avoiding duplicate
edges and the incorrect assumption that every source-order backward jump is a
loop.

## Run

```bash
python scripts/extract_promise_cfg.py
```

Useful options:

```bash
python scripts/extract_promise_cfg.py --dataset-name log4j-1.2
python scripts/extract_promise_cfg.py --build-dir path/to/build-cache
python scripts/extract_promise_cfg.py --help
```

The extractor cleans its generated graph, tensor, log, and build directories by
default. The separate verified-release cache is retained. `--no-clean` should
only be used deliberately. Exact-release bytecode recovery is mandatory for
configured projects. If a required archive is unavailable or fails checksum
validation, extraction stops instead of silently reverting to lower coverage.

## Outputs

```text
outputs/promise/cfg/graph_index.csv
outputs/promise/cfg/node_type_vocab.json
outputs/promise/cfg/stmt_kind_vocab.json
outputs/promise/cfg/invoke_kind_vocab.json
outputs/promise/cfg/edge_type_vocab.json
outputs/promise/cfg/feature_names.json
outputs/promise/cfg/cfg_summary.json
outputs/promise/cfg/parse_failures.json
outputs/promise/cfg/validation_issues.json
outputs/promise/cfg/graphs/<file>.json
outputs/promise/cfg/tensors/<file>_*.npy
outputs/promise/cfg/logs/
```

## Validation

The Python stage checks tensor dimensions and bounds plus edge-specific
invariants for entry, branch, goto, switch, return, throw, and exception flow.
Problems are written to `validation_issues.json` rather than silently ignored.

ECJ source paths are quoted in its argument file. This is required for legacy
projects such as Log4j that contain source directories with spaces.
The bundled ECJ was upgraded to a modular-JDK-aware release so legacy sources
can resolve standard-library classes when extraction runs on current JDKs.
Javadoc tags are ignored when detecting whether a project needs Java 5 source
syntax, preserving valid identifiers such as `enum` in older Java sources.
The complete version tree is compiled so mapped classes can resolve sibling
sources; Soot still receives only the mapped benchmark classes.
The summary records release-cache status, selected JAR paths, and bytecode
precedence for every dataset.

## Placeholder Policy

If no useful Soot CFG can be recovered for a mapped file, the extractor writes
a tagged `ENTRY -> EXIT` placeholder to preserve file alignment. A placeholder
contains no meaningful behavior and must not be treated as a real CFG. The
standalone CFG evaluator excludes placeholders, while the final NDG evaluator
retains the file node but masks its CFG view.

## Expected Result

- One indexed file-level graph per mapped input row.
- Real and placeholder counts are reported separately.
- CFG edge counts are reported per project.
- Node, statement, invocation, and edge tensors match graph metadata.
- Java compilation and Soot diagnostics remain available for auditing.
