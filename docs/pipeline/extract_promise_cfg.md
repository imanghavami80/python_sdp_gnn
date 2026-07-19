# CFG Extraction

**Implementation:** `scripts/extract_promise_cfg.py`

**Java backend:** `scripts/soot_cfg_extractor.java`

## Purpose

This stage builds statement-level Control Flow Graphs using compiled Java
bytecode and Soot. It does not infer control flow from regular expressions or
construct an approximate CFG from the source AST.

Each method and constructor receives explicit `ENTRY` and `EXIT` nodes. Method
graphs are aggregated into one file-level CFG so they align with PROMISE
file-level labels.

## Prerequisites

- Run preprocessing first.
- Install a Java JDK so `java` and `javac` are available.
- Keep the bundled ECJ and Soot JARs under `tools/`.

## Extraction Architecture

```text
Mapped Java source
  -> ECJ partial compilation with debug line information
  -> Soot Jimple bodies
  -> method statement CFGs
  -> Python feature enrichment and validation
  -> one aggregated graph per file
```

ECJ uses partial-error recovery because the PROMISE projects are old and may
have missing build dependencies. A nonzero ECJ status is logged. Soot can still
recover real CFGs from the class files that were emitted.

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
CFG_NEXT
CFG_TRUE
CFG_FALSE
CFG_RETURN
CFG_EXCEPTION
CFG_BACK
CFG_SWITCH_CASE
CFG_SWITCH_DEFAULT
```

The CFG encoder consumes these edge types directly through learned edge
embeddings.

## Run

```bash
python scripts/extract_promise_cfg.py
```

Useful options:

```bash
python scripts/extract_promise_cfg.py --dataset-name log4j-1.1
python scripts/extract_promise_cfg.py --build-dir path/to/build-cache
python scripts/extract_promise_cfg.py --help
```

The extractor cleans its generated graph, tensor, log, and build directories by
default. `--no-clean` should only be used deliberately.

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

The Python stage checks graph dimensions, method ENTRY/EXIT nodes, return-to-exit
flow, condition branches, loop back edges, and vocabulary bounds. Problems are
written to `validation_issues.json` rather than silently ignored.

## Placeholder Policy

If no useful Soot CFG can be recovered for a mapped file, the extractor writes
a tagged `ENTRY -> EXIT` placeholder to preserve file alignment. A placeholder
contains no meaningful behavior and must not be treated as a real CFG. The
standalone CFG evaluator excludes placeholders, while the final NDG evaluator
retains the file node but masks its CFG view.

## Expected Result

- One indexed file-level graph per mapped input row.
- Real and placeholder counts are reported separately.
- Node, statement, invocation, and edge tensors match graph metadata.
- Java compilation and Soot diagnostics remain available for auditing.
