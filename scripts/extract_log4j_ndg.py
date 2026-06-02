#!/usr/bin/env python3
"""Extract a file-level Network Dependency Graph (NDG) for Log4j 1.0.

The graph has one node per PROMISE row with a mapped Java source file. Typed
edges represent static dependencies extracted from Java source ASTs.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import javalang
import numpy as np
import pandas as pd
from javalang.ast import Node

EDGE_TYPES = [
    "EXTENDS",
    "IMPLEMENTS",
    "FIELD_TYPE",
    "PARAMETER_TYPE",
    "RETURN_TYPE",
    "OBJECT_CREATION",
    "METHOD_CALL",
]
EDGE_TYPE_TO_ID = {edge_type: idx for idx, edge_type in enumerate(EDGE_TYPES)}
NON_FEATURE_COLUMNS = {"name", "bug", "source_path", "match_strategy"}
LEGACY_IDENTIFIER_PATTERN = re.compile(r"\b(enum|assert)\b")


def normalize_legacy_identifiers(code: str) -> str:
    """Rename pre-Java-5 identifiers that modern Java parsers reserve."""
    return LEGACY_IDENTIFIER_PATTERN.sub(lambda match: f"legacy_{match.group(1)}", code)


def package_name(tree: Node) -> str:
    package = getattr(tree, "package", None)
    return str(package.name) if package is not None else ""


def source_fqcn(source_root: Path, source_path: Path) -> str:
    return ".".join(source_path.relative_to(source_root).with_suffix("").parts)


def iter_reference_type_names(type_node: Any) -> Iterable[str]:
    if type_node is None:
        return
    if isinstance(type_node, str):
        yield type_node
        return
    name = getattr(type_node, "name", None)
    if name:
        parts = [str(name)]
        sub_type = getattr(type_node, "sub_type", None)
        while sub_type is not None:
            sub_name = getattr(sub_type, "name", None)
            if sub_name:
                parts.append(str(sub_name))
            sub_type = getattr(sub_type, "sub_type", None)
        yield ".".join(parts)
    for argument in getattr(type_node, "arguments", None) or []:
        yield from iter_reference_type_names(getattr(argument, "type", None))


@dataclass
class TypeResolver:
    package: str
    explicit_imports: dict[str, str]
    wildcard_imports: list[str]
    known_source_types: set[str]
    simple_to_types: dict[str, set[str]]

    def resolve(self, raw_name: str | None) -> str | None:
        if not raw_name:
            return None
        name = str(raw_name).strip()
        if not name:
            return None
        name = name.replace("[]", "").replace("$", ".")

        candidates: list[str] = []
        if name in self.known_source_types:
            candidates.append(name)
        if name in self.explicit_imports:
            candidates.append(self.explicit_imports[name])
        if self.package:
            candidates.append(f"{self.package}.{name}")
        candidates.extend(f"{prefix}.{name}" for prefix in self.wildcard_imports)
        candidates.extend(sorted(self.simple_to_types.get(name.split(".")[-1], set())))

        for candidate in candidates:
            current = candidate
            while current:
                if current in self.known_source_types:
                    return current
                if "." not in current:
                    break
                current = current.rsplit(".", 1)[0]
        return None


@dataclass
class DependencyCollector:
    node_names: set[str]
    relations: set[tuple[str, str, str]] = field(default_factory=set)

    def add(self, source_name: str, target_name: str | None, edge_type: str) -> None:
        if target_name is None:
            return
        if target_name not in self.node_names:
            return
        if source_name == target_name:
            return
        self.relations.add((source_name, target_name, edge_type))


def parse_java_source(source_path: Path) -> tuple[Node, str, str | None]:
    code = source_path.read_text(encoding="utf-8", errors="ignore")
    try:
        return javalang.parse.parse(code), "javalang", None
    except Exception as strict_error:  # noqa: BLE001
        normalized_code = normalize_legacy_identifiers(code)
        try:
            return javalang.parse.parse(normalized_code), "legacy_normalized", repr(strict_error)
        except Exception as fallback_error:  # noqa: BLE001
            raise RuntimeError(f"strict={strict_error!r}; legacy_normalized={fallback_error!r}") from fallback_error


def build_type_resolver(
    tree: Node,
    known_source_types: set[str],
    simple_to_types: dict[str, set[str]],
) -> TypeResolver:
    explicit_imports: dict[str, str] = {}
    wildcard_imports: list[str] = []
    for imported in getattr(tree, "imports", None) or []:
        path = str(imported.path)
        if imported.wildcard:
            wildcard_imports.append(path)
        else:
            explicit_imports[path.split(".")[-1]] = path
    return TypeResolver(
        package=package_name(tree),
        explicit_imports=explicit_imports,
        wildcard_imports=wildcard_imports,
        known_source_types=known_source_types,
        simple_to_types=simple_to_types,
    )


def emit_type_dependencies(
    collector: DependencyCollector,
    resolver: TypeResolver,
    source_name: str,
    type_node: Any,
    edge_type: str,
) -> None:
    for type_name in iter_reference_type_names(type_node):
        collector.add(source_name, resolver.resolve(type_name), edge_type)


def collect_symbols(resolver: TypeResolver, declaration: Node) -> dict[str, str]:
    symbols: dict[str, str] = {}
    for field_decl in getattr(declaration, "fields", None) or []:
        resolved_names = [resolver.resolve(name) for name in iter_reference_type_names(field_decl.type)]
        resolved = next((name for name in resolved_names if name), None)
        if resolved:
            for declarator in field_decl.declarators:
                symbols[str(declarator.name)] = resolved
    return symbols


def add_scoped_symbols(resolver: TypeResolver, executable: Node, symbols: dict[str, str]) -> None:
    for parameter in getattr(executable, "parameters", None) or []:
        resolved = next((resolver.resolve(name) for name in iter_reference_type_names(parameter.type) if resolver.resolve(name)), None)
        if resolved:
            symbols[str(parameter.name)] = resolved
    for _, local_decl in executable.filter(javalang.tree.LocalVariableDeclaration):
        resolved = next((resolver.resolve(name) for name in iter_reference_type_names(local_decl.type) if resolver.resolve(name)), None)
        if resolved:
            for declarator in local_decl.declarators:
                symbols[str(declarator.name)] = resolved


def invocation_target(qualifier: str | None, symbols: dict[str, str], resolver: TypeResolver) -> str | None:
    if not qualifier:
        return None
    text = str(qualifier)
    if text.startswith("this."):
        text = text[5:]
    if text in symbols:
        return symbols[text]
    first = text.split(".", 1)[0]
    if first in symbols:
        return symbols[first]
    return resolver.resolve(text)


def extract_file_dependencies(
    source_name: str,
    tree: Node,
    resolver: TypeResolver,
    collector: DependencyCollector,
) -> None:
    type_declaration_classes = (
        javalang.tree.ClassDeclaration,
        javalang.tree.InterfaceDeclaration,
        javalang.tree.EnumDeclaration,
    )
    for _, declaration in tree:
        if not isinstance(declaration, type_declaration_classes):
            continue
        base_symbols = collect_symbols(resolver, declaration)

        extends = getattr(declaration, "extends", None)
        extends_nodes = extends if isinstance(extends, list) else [extends]
        for type_node in extends_nodes:
            emit_type_dependencies(collector, resolver, source_name, type_node, "EXTENDS")
        for type_node in getattr(declaration, "implements", None) or []:
            emit_type_dependencies(collector, resolver, source_name, type_node, "IMPLEMENTS")

        for field_decl in getattr(declaration, "fields", None) or []:
            emit_type_dependencies(collector, resolver, source_name, field_decl.type, "FIELD_TYPE")

        executables = list(getattr(declaration, "constructors", None) or []) + list(getattr(declaration, "methods", None) or [])
        for executable in executables:
            symbols = dict(base_symbols)
            add_scoped_symbols(resolver, executable, symbols)

            for parameter in getattr(executable, "parameters", None) or []:
                emit_type_dependencies(
                    collector,
                    resolver,
                    source_name,
                    parameter.type,
                    "PARAMETER_TYPE",
                )
            emit_type_dependencies(
                collector,
                resolver,
                source_name,
                getattr(executable, "return_type", None),
                "RETURN_TYPE",
            )

            for _, invocation in executable.filter(javalang.tree.MethodInvocation):
                target = invocation_target(getattr(invocation, "qualifier", None), symbols, resolver)
                collector.add(
                    source_name,
                    target,
                    "METHOD_CALL",
                )

    for _, creator in tree.filter(javalang.tree.ClassCreator):
        emit_type_dependencies(
            collector,
            resolver,
            source_name,
            creator.type,
            "OBJECT_CREATION",
        )


def mermaid_preview(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], limit: int = 16) -> str:
    degree = Counter()
    for edge in edges:
        degree[edge["source"]] += 1
        degree[edge["target"]] += 1
    selected = {node_id for node_id, _ in degree.most_common(limit)}
    selected_edges = [edge for edge in edges if edge["source"] in selected and edge["target"] in selected][:45]
    node_by_id = {node["id"]: node for node in nodes}
    lines = ["graph LR"]
    for node_id in sorted(selected):
        label = node_by_id[node_id]["name"].split(".")[-1]
        lines.append(f'  N{node_id}["{label}#{node_id}"]')
    for edge in selected_edges:
        lines.append(f'  N{edge["source"]} -->|{edge["edge_type"]}| N{edge["target"]}')
    return "\n".join(lines)


def validate_graph(
    x: np.ndarray,
    y: np.ndarray,
    edge_index: np.ndarray,
    edge_type: np.ndarray,
    edges: list[dict[str, Any]],
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    num_nodes = x.shape[0]
    if y.shape != (num_nodes,):
        issues.append({"issue": f"Expected y shape {(num_nodes,)}, received {y.shape}"})
    if edge_index.shape != (2, len(edges)):
        issues.append({"issue": f"Expected edge_index shape {(2, len(edges))}, received {edge_index.shape}"})
    if edge_type.shape != (len(edges),):
        issues.append({"issue": f"Expected edge_type shape {(len(edges),)}, received {edge_type.shape}"})
    if not np.isfinite(x).all():
        issues.append({"issue": "Node feature matrix contains non-finite values"})
    if set(np.unique(y)) - {0, 1}:
        issues.append({"issue": "Classification labels are not binary"})
    if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= num_nodes):
        issues.append({"issue": "edge_index contains out-of-bounds node ids"})
    if edge_type.size and (edge_type.min() < 0 or edge_type.max() >= len(EDGE_TYPES)):
        issues.append({"issue": "edge_type contains unsupported ids"})
    if any(edge["source"] == edge["target"] for edge in edges):
        issues.append({"issue": "File-level self edge found"})
    typed_keys = {(edge["source"], edge["target"], edge["edge_type"]) for edge in edges}
    if len(typed_keys) != len(edges):
        issues.append({"issue": "Duplicate typed edge found"})
    return issues


def build_report(
    summary: dict[str, Any],
    feature_names: list[str],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    parse_fallbacks: list[dict[str, str]],
    parse_failures: list[dict[str, str]],
) -> str:
    edge_counts = Counter(edge["edge_type"] for edge in edges)
    edge_rows = [f"| `{name}` | {EDGE_TYPE_TO_ID[name]} | {edge_counts.get(name, 0)} |" for name in EDGE_TYPES]
    feature_rows = [f"| {index} | `{name}` |" for index, name in enumerate(feature_names)]
    fallback_rows = [
        f"| `{item['name']}` | `{item['source_path']}` |"
        for item in parse_fallbacks
    ] or ["| None | None |"]
    failure_rows = [
        f"| `{item['name']}` | `{item['error']}` |"
        for item in parse_failures
    ] or ["| None | None |"]

    lines = [
        "# NDG Extraction Report for Log4j 1.0",
        "",
        "## 1. Objective",
        "This report describes the file-level Network Dependency Graph (NDG) extraction step for the Log4j 1.0 "
        "PROMISE dataset. The NDG captures static dependencies between mapped Java files so the software defect "
        "prediction model can learn project structure alongside per-file metrics, AST structure, and CFG behavior.",
        "",
        "## 2. Graph Granularity",
        "- The NDG is one project-level directed heterogeneous graph.",
        "- Each node is one PROMISE dataset class with a mapped Java source file.",
        "- Each directed edge `A -> B` means file `A` statically depends on file `B`.",
        "- Dependencies to standard-library classes, external libraries, and source files without mapped PROMISE nodes "
        "are excluded because they cannot be represented as nodes in this file-level dataset graph.",
        "",
        "## 3. Input Data",
        "- Dataset: `outputs/log4j/log4j_preprocessed_standard.csv`",
        "- Source root: `projects/log4j/logging-log4j1-v_1_0/src/java`",
        f"- Dataset rows: {summary['dataset_rows']}",
        f"- Rows with mapped source files: {summary['num_nodes']}",
        f"- Rows without mapped source files: {summary['unmapped_rows']}",
        "",
        "## 4. Extraction Pipeline",
        "- Each mapped Java file is parsed with `javalang` into a source AST.",
        "- Legacy Log4j identifiers named `enum` or `assert` are deterministically renamed only when strict parsing fails.",
        "- Imports, package declarations, and the source-tree class index resolve referenced types to project files.",
        "- Dependencies are deduplicated by `(source, target, edge_type)`.",
        "- Dependencies are stored as compact typed edges.",
        "",
        "```text",
        "Java source -> javalang AST -> project type resolution -> typed file dependencies -> project-level NDG tensors",
        "```",
        "",
        "## 5. Dependency Edge Types",
        "| Edge type | ID | Exported edges |",
        "| --- | ---: | ---: |",
        *edge_rows,
        "",
        "| Edge type | Rule for creating `A -> B` |",
        "| --- | --- |",
        "| `EXTENDS` | Class `A` extends class `B`. |",
        "| `IMPLEMENTS` | Class `A` implements interface `B`. |",
        "| `FIELD_TYPE` | Class `A` declares a field whose type is `B`. |",
        "| `PARAMETER_TYPE` | A method in class `A` accepts a parameter whose type is `B`. |",
        "| `RETURN_TYPE` | A method in class `A` returns type `B`. |",
        "| `OBJECT_CREATION` | A method in class `A` creates an object of type `B`. |",
        "| `METHOD_CALL` | A method in class `A` calls a method through an expression statically resolved to `B`. |",
        "",
        "The graph is multi-relational: a file pair can have multiple edges when different dependency rules apply.",
        "",
        "## 6. Node Feature Vector",
        "Each node uses the 20 standardized file metrics from the preprocessed PROMISE dataset:",
        "",
        "```text",
        "x(file) = [wmc, dit, noc, cbo, rfc, lcom, ca, ce, npm, lcom3, loc, dam, moa, mfa, cam, ic, cbm, amc, max_cc, avg_cc]",
        "```",
        "",
        "The original `bug` count is excluded from `x` to prevent target leakage. For binary defect prediction, "
        "`y.npy` stores `1` when `bug > 0`, otherwise `0`. Metadata columns `name`, "
        "`source_path`, and `match_strategy` are also excluded from `x`.",
        "",
        "| Feature index | Metric |",
        "| ---: | --- |",
        *feature_rows,
        "",
        "## 7. Tensor Files and Semantics",
        "| File | Shape | Meaning |",
        "| --- | --- | --- |",
        f"| `tensors/log4j_x.npy` | `[{summary['num_nodes']}, {summary['node_feature_dim']}]` | Standardized file-metric node features. |",
        f"| `tensors/log4j_y.npy` | `[{summary['num_nodes']}]` | Binary defective/non-defective labels (`bug > 0`). |",
        f"| `tensors/log4j_edge_index.npy` | `[2, {summary['num_edges']}]` | Directed file dependency connectivity. |",
        f"| `tensors/log4j_edge_type.npy` | `[{summary['num_edges']}]` | Typed relation id aligned with each edge column. |",
        "",
        "`edge_index[:, i]` and `edge_type[i]` describe the same directed relation.",
        "",
        "## 8. How to Use the NDG in a GNN",
        "The NDG already has one node per Java file, matching the target prediction granularity. Use a relational GNN "
        "such as `RGCNConv` so each file representation is updated from typed dependency edges. Later, fuse each NDG "
        "file embedding with pooled AST and CFG representations joined by fully-qualified class name.",
        "",
        "```python",
        "import numpy as np",
        "",
        "root = 'outputs/log4j/ndg/tensors'",
        "x = np.load(f'{root}/log4j_x.npy')",
        "y = np.load(f'{root}/log4j_y.npy')",
        "edge_index = np.load(f'{root}/log4j_edge_index.npy')",
        "edge_type = np.load(f'{root}/log4j_edge_type.npy')",
        "",
        "# x.shape          == [num_files, 20]",
        "# y.shape          == [num_files]",
        "# edge_index.shape == [2, num_edges]",
        "# edge_type.shape  == [num_edges]",
        "```",
        "",
        "## 9. Output Files",
        "- `outputs/log4j/ndg/graph_index.csv`: index row for the generated project-level graph.",
        "- `outputs/log4j/ndg/graphs/log4j.json`: readable NDG nodes and typed edges.",
        "- `outputs/log4j/ndg/tensors/log4j_x.npy`: node metric feature matrix.",
        "- `outputs/log4j/ndg/tensors/log4j_y.npy`: binary defect label vector.",
        "- `outputs/log4j/ndg/tensors/log4j_edge_index.npy`: directed edge connectivity.",
        "- `outputs/log4j/ndg/tensors/log4j_edge_type.npy`: typed edge ids.",
        "- `outputs/log4j/ndg/feature_names.json`: stable node feature order.",
        "- `outputs/log4j/ndg/edge_type_vocab.json`: stable edge type vocabulary.",
        "- `outputs/log4j/ndg/parse_fallbacks.json`: strict parser failures recovered by legacy normalization.",
        "- `outputs/log4j/ndg/parse_failures.json`: unrecovered source parse failures.",
        "- `outputs/log4j/ndg/validation_issues.json`: graph validation results.",
        "- `outputs/log4j/ndg/ndg_summary.json`: global extraction statistics.",
        "",
        "## 10. Results",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Dataset rows | {summary['dataset_rows']} |",
        f"| NDG nodes | {summary['num_nodes']} |",
        f"| Unmapped dataset rows excluded | {summary['unmapped_rows']} |",
        f"| Typed NDG edges | {summary['num_edges']} |",
        f"| Distinct connected file pairs | {summary['distinct_file_pairs']} |",
        f"| Strict parses | {summary['strict_parses']} |",
        f"| Legacy-normalized fallback parses | {summary['fallback_parses']} |",
        f"| Unrecovered parse failures | {summary['parse_failures']} |",
        f"| Node feature dimension | {summary['node_feature_dim']} |",
        f"| Defective nodes (`bug > 0`) | {summary['defective_nodes']} |",
        f"| Validation issues | {summary['validation_issues']} |",
        "",
        "## 11. Validation",
        "The generated tensors are checked for shape consistency, finite node features, binary labels, valid edge "
        "bounds, supported relation ids, unique typed edges, and absence of file-level self edges.",
        "",
        f"- Validation issues found: {summary['validation_issues']}",
        "",
        "## 12. Notes and Limitations",
        "- Dependencies are static and file-level. They do not prove runtime execution.",
        "- Method-call resolution is intentionally conservative. Calls without a resolvable receiver type are omitted.",
        "- Dependencies to non-PROMISE files are filtered because the final graph nodes must align with mapped rows.",
        "- Legacy normalization changes only parser input identifiers; it does not alter source files.",
        "",
        "### 12.1 Legacy-Normalized Parses",
        "| Class | Source path |",
        "| --- | --- |",
        *fallback_rows,
        "",
        "### 12.2 Unrecovered Parse Failures",
        "| Class | Failure |",
        "| --- | --- |",
        *failure_rows,
        "",
        "## 13. Sample Visualization",
        "The diagram below shows a compact high-connectivity subset of the generated file-level NDG.",
        "",
        "```mermaid",
        mermaid_preview(nodes, edges),
        "```",
        "",
        "## 14. Future Integration",
        "The NDG uses the same fully-qualified class names as the AST and CFG indexes. This keeps syntax in AST nodes, "
        "control behavior in CFG edges, and file dependencies in NDG edges while allowing class-level multi-view fusion.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source_root = repo_root / "projects/log4j/logging-log4j1-v_1_0/src/java"
    dataset_path = repo_root / "outputs/log4j/log4j_preprocessed_standard.csv"
    out_dir = repo_root / "outputs/log4j/ndg"
    graphs_dir = out_dir / "graphs"
    tensors_dir = out_dir / "tensors"
    out_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    tensors_dir.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_csv(dataset_path)
    mapped = dataset[dataset["source_path"].notna() & (dataset["match_strategy"] != "not_found")].copy()
    feature_names = [column for column in dataset.columns if column not in NON_FEATURE_COLUMNS]
    mapped_names = mapped["name"].astype(str).tolist()
    node_name_to_id = {name: idx for idx, name in enumerate(mapped_names)}
    node_names = set(node_name_to_id)

    source_types = {
        source_fqcn(source_root, source_path)
        for source_path in source_root.rglob("*.java")
    }
    simple_to_types: dict[str, set[str]] = defaultdict(set)
    for type_name in source_types:
        simple_to_types[type_name.split(".")[-1]].add(type_name)

    x = mapped[feature_names].to_numpy(dtype=np.float32)
    bug_count = mapped["bug"].to_numpy(dtype=np.int64)
    y = (bug_count > 0).astype(np.int64)
    nodes = [
        {
            "id": node_id,
            "name": str(row["name"]),
            "source_path": str(row["source_path"]),
            "bug": int(row["bug"]),
            "label": int(row["bug"] > 0),
            "features": {feature: float(row[feature]) for feature in feature_names},
        }
        for node_id, (_, row) in enumerate(mapped.iterrows())
    ]

    collector = DependencyCollector(node_names=node_names)
    parse_fallbacks: list[dict[str, str]] = []
    parse_failures: list[dict[str, str]] = []
    strict_parses = 0
    for _, row in mapped.iterrows():
        name = str(row["name"])
        source_path = Path(str(row["source_path"]))
        try:
            tree, parser_mode, strict_error = parse_java_source(source_path)
            if parser_mode == "javalang":
                strict_parses += 1
            else:
                parse_fallbacks.append(
                    {"name": name, "source_path": str(source_path), "strict_error": str(strict_error)}
                )
            resolver = build_type_resolver(tree, source_types, simple_to_types)
            extract_file_dependencies(name, tree, resolver, collector)
        except Exception as error:  # noqa: BLE001
            parse_failures.append({"name": name, "source_path": str(source_path), "error": str(error)})

    edges = []
    for source_name, target_name, edge_type in sorted(collector.relations):
        edges.append(
            {
                "source": node_name_to_id[source_name],
                "target": node_name_to_id[target_name],
                "source_name": source_name,
                "target_name": target_name,
                "edge_type": edge_type,
                "edge_type_id": EDGE_TYPE_TO_ID[edge_type],
            }
        )

    if edges:
        edge_index = np.array([[edge["source"] for edge in edges], [edge["target"] for edge in edges]], dtype=np.int64)
        edge_type = np.array([edge["edge_type_id"] for edge in edges], dtype=np.int64)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_type = np.zeros((0,), dtype=np.int64)

    validation_issues = validate_graph(x, y, edge_index, edge_type, edges)
    summary = {
        "dataset_rows": int(len(dataset)),
        "num_nodes": int(len(nodes)),
        "unmapped_rows": int(len(dataset) - len(nodes)),
        "num_edges": int(len(edges)),
        "distinct_file_pairs": int(len({(edge["source"], edge["target"]) for edge in edges})),
        "node_feature_dim": int(x.shape[1]),
        "edge_type_vocab_size": len(EDGE_TYPE_TO_ID),
        "defective_nodes": int(y.sum()),
        "non_defective_nodes": int((y == 0).sum()),
        "strict_parses": strict_parses,
        "fallback_parses": len(parse_fallbacks),
        "parse_failures": len(parse_failures),
        "validation_issues": len(validation_issues),
    }
    graph = {
        "graph_type": "file_level_network_dependency_graph",
        "project": "log4j-1.0",
        "directed": True,
        "multi_relational": True,
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "node_feature_dim": int(x.shape[1]),
        "nodes": nodes,
        "edges": edges,
    }

    graph_path = graphs_dir / "log4j.json"
    x_path = tensors_dir / "log4j_x.npy"
    y_path = tensors_dir / "log4j_y.npy"
    edge_index_path = tensors_dir / "log4j_edge_index.npy"
    edge_type_path = tensors_dir / "log4j_edge_type.npy"
    np.save(x_path, x)
    np.save(y_path, y)
    np.save(edge_index_path, edge_index)
    np.save(edge_type_path, edge_type)
    graph_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    (out_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    (out_dir / "edge_type_vocab.json").write_text(json.dumps(EDGE_TYPE_TO_ID, indent=2), encoding="utf-8")
    (out_dir / "parse_fallbacks.json").write_text(json.dumps(parse_fallbacks, indent=2), encoding="utf-8")
    (out_dir / "parse_failures.json").write_text(json.dumps(parse_failures, indent=2), encoding="utf-8")
    (out_dir / "validation_issues.json").write_text(json.dumps(validation_issues, indent=2), encoding="utf-8")
    (out_dir / "ndg_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    pd.DataFrame(
        [
            {
                "name": "log4j",
                "num_nodes": len(nodes),
                "num_edges": len(edges),
                "feature_dim": int(x.shape[1]),
                "graph_json": str(graph_path),
                "x_npy": str(x_path),
                "y_npy": str(y_path),
                "edge_index_npy": str(edge_index_path),
                "edge_type_npy": str(edge_type_path),
            }
        ]
    ).to_csv(out_dir / "graph_index.csv", index=False)
    (out_dir / "ndg_report.md").write_text(
        build_report(summary, feature_names, nodes, edges, parse_fallbacks, parse_failures),
        encoding="utf-8",
    )

    print(
        "NDG extraction finished. "
        f"nodes={summary['num_nodes']} edges={summary['num_edges']} "
        f"fallback_parses={summary['fallback_parses']} validation_issues={summary['validation_issues']}"
    )
    print(f"summary={out_dir / 'ndg_summary.json'}")
    print(f"report={out_dir / 'ndg_report.md'}")


if __name__ == "__main__":
    main()
