#!/usr/bin/env python3
"""Extract one typed file-level dependency graph per PROMISE project."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
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

METRIC_COLUMNS = [
    "wmc",
    "dit",
    "noc",
    "cbo",
    "rfc",
    "lcom",
    "ca",
    "ce",
    "npm",
    "lcom3",
    "loc",
    "dam",
    "moa",
    "mfa",
    "cam",
    "ic",
    "cbm",
    "amc",
    "max_cc",
    "avg_cc",
]
LEGACY_IDENTIFIER_PATTERN = re.compile(r"\b(enum|assert)\b")


def normalize_legacy_identifiers(code: str) -> str:
    """Rename pre-Java-5 identifiers that modern Java parsers reserve."""
    return LEGACY_IDENTIFIER_PATTERN.sub(lambda match: f"legacy_{match.group(1)}", code)


def sanitize_filename(value: str, max_len: int = 180) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    if len(safe) <= max_len:
        return safe
    import hashlib

    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"{safe[: max_len - 13]}_{digest}"


def package_name(tree: Node) -> str:
    package = getattr(tree, "package", None)
    return str(package.name) if package is not None else ""


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
        simple_matches = self.simple_to_types.get(name.split(".")[-1], set())
        if len(simple_matches) == 1:
            candidates.extend(simple_matches)

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
            message = f"strict={strict_error!r}; legacy_normalized={fallback_error!r}"
            raise RuntimeError(message) from fallback_error


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


def first_resolved_type(resolver: TypeResolver, type_node: Any) -> str | None:
    for type_name in iter_reference_type_names(type_node):
        resolved = resolver.resolve(type_name)
        if resolved:
            return resolved
    return None


def collect_symbols(resolver: TypeResolver, declaration: Node) -> dict[str, str]:
    symbols: dict[str, str] = {}
    for field_decl in getattr(declaration, "fields", None) or []:
        resolved = first_resolved_type(resolver, field_decl.type)
        if resolved:
            for declarator in field_decl.declarators:
                symbols[str(declarator.name)] = resolved
    return symbols


def add_scoped_symbols(resolver: TypeResolver, executable: Node, symbols: dict[str, str]) -> None:
    for parameter in getattr(executable, "parameters", None) or []:
        resolved = first_resolved_type(resolver, parameter.type)
        if resolved:
            symbols[str(parameter.name)] = resolved
    for _, local_decl in executable.filter(javalang.tree.LocalVariableDeclaration):
        resolved = first_resolved_type(resolver, local_decl.type)
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
                emit_type_dependencies(collector, resolver, source_name, parameter.type, "PARAMETER_TYPE")
            emit_type_dependencies(collector, resolver, source_name, getattr(executable, "return_type", None), "RETURN_TYPE")

            for _, invocation in executable.filter(javalang.tree.MethodInvocation):
                target = invocation_target(getattr(invocation, "qualifier", None), symbols, resolver)
                collector.add(source_name, target, "METHOD_CALL")

    for _, creator in tree.filter(javalang.tree.ClassCreator):
        emit_type_dependencies(collector, resolver, source_name, creator.type, "OBJECT_CREATION")


def prepare_output_dirs(output_dir: Path, clean: bool) -> tuple[Path, Path]:
    graphs_dir = output_dir / "graphs"
    tensors_dir = output_dir / "tensors"
    output_dir.mkdir(parents=True, exist_ok=True)
    if clean:
        shutil.rmtree(graphs_dir, ignore_errors=True)
        shutil.rmtree(tensors_dir, ignore_errors=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    tensors_dir.mkdir(parents=True, exist_ok=True)
    return graphs_dir, tensors_dir


def load_source_roots(summary_path: Path) -> dict[str, Path]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    roots: dict[str, Path] = {}
    for project in summary.get("project_summaries", []):
        roots[str(project["dataset_name"])] = Path(project["source_root"])
    return roots


def load_mapped_rows(input_csv: Path, dataset_filter: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(input_csv)
    required = {"dataset_name", "name", "bug", "source_path", "match_strategy"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required NDG input columns: {missing}")
    if dataset_filter:
        df = df[df["dataset_name"].astype(str) == dataset_filter].copy()
    mapped = df[df["source_path"].notna()].copy()
    mapped = mapped[mapped["match_strategy"].astype(str) != "not_found"].copy()
    mapped = mapped[mapped["match_strategy"].astype(str) != "ambiguous_simple_name"].copy()
    return df.reset_index(drop=True), mapped.reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    missing_metrics = [column for column in METRIC_COLUMNS if column not in df.columns]
    if missing_metrics:
        raise ValueError(f"Missing expected metric columns for NDG node features: {missing_metrics}")
    return list(METRIC_COLUMNS)


def validate_graph(
    dataset_name: str,
    x: np.ndarray,
    y: np.ndarray,
    edge_index: np.ndarray,
    edge_type: np.ndarray,
    edges: list[dict[str, Any]],
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    num_nodes = x.shape[0]
    if y.shape != (num_nodes,):
        issues.append({"dataset_name": dataset_name, "issue": f"Expected y shape {(num_nodes,)}, received {y.shape}"})
    if edge_index.shape != (2, len(edges)):
        issues.append({"dataset_name": dataset_name, "issue": f"Expected edge_index shape {(2, len(edges))}, received {edge_index.shape}"})
    if edge_type.shape != (len(edges),):
        issues.append({"dataset_name": dataset_name, "issue": f"Expected edge_type shape {(len(edges),)}, received {edge_type.shape}"})
    if not np.isfinite(x).all():
        issues.append({"dataset_name": dataset_name, "issue": "Node feature matrix contains non-finite values"})
    if set(np.unique(y)) - {0, 1}:
        issues.append({"dataset_name": dataset_name, "issue": "Classification labels are not binary"})
    if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= num_nodes):
        issues.append({"dataset_name": dataset_name, "issue": "edge_index contains out-of-bounds node ids"})
    if edge_type.size and (edge_type.min() < 0 or edge_type.max() >= len(EDGE_TYPES)):
        issues.append({"dataset_name": dataset_name, "issue": "edge_type contains unsupported ids"})
    if any(edge["source"] == edge["target"] for edge in edges):
        issues.append({"dataset_name": dataset_name, "issue": "File-level self edge found"})
    typed_keys = {(edge["source"], edge["target"], edge["edge_type"]) for edge in edges}
    if len(typed_keys) != len(edges):
        issues.append({"dataset_name": dataset_name, "issue": "Duplicate typed edge found"})
    return issues


def build_project_ndg(
    dataset_name: str,
    dataset_rows: pd.DataFrame,
    mapped_rows: pd.DataFrame,
    source_root: Path | None,
    features: list[str],
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    mapped_rows = mapped_rows.sort_values("name").reset_index(drop=True)
    mapped_names = mapped_rows["name"].astype(str).tolist()
    node_name_to_id = {name: idx for idx, name in enumerate(mapped_names)}
    node_names = set(node_name_to_id)

    simple_to_types: dict[str, set[str]] = defaultdict(set)
    for type_name in node_names:
        simple_to_types[type_name.split(".")[-1]].add(type_name)

    x = mapped_rows[features].to_numpy(dtype=np.float32)
    y = (mapped_rows["bug"].to_numpy(dtype=np.float32) > 0).astype(np.int64)
    nodes = [
        {
            "id": node_id,
            "dataset_name": dataset_name,
            "name": str(row["name"]),
            "source_path": str(row["source_path"]),
            "bug": int(row["bug"]),
            "label": int(float(row["bug"]) > 0),
            "features": {feature: float(row[feature]) for feature in features},
        }
        for node_id, (_, row) in enumerate(mapped_rows.iterrows())
    ]

    collector = DependencyCollector(node_names=node_names)
    parse_fallbacks: list[dict[str, str]] = []
    parse_failures: list[dict[str, str]] = []
    strict_parses = 0

    for _, row in mapped_rows.iterrows():
        name = str(row["name"])
        source_path = Path(str(row["source_path"]))
        try:
            tree, parser_mode, strict_error = parse_java_source(source_path)
            if parser_mode == "javalang":
                strict_parses += 1
            else:
                parse_fallbacks.append(
                    {
                        "dataset_name": dataset_name,
                        "name": name,
                        "source_path": str(source_path),
                        "strict_error": str(strict_error),
                    }
                )
            resolver = build_type_resolver(tree, node_names, simple_to_types)
            extract_file_dependencies(name, tree, resolver, collector)
        except Exception as error:  # noqa: BLE001
            parse_failures.append(
                {
                    "dataset_name": dataset_name,
                    "name": name,
                    "source_path": str(source_path),
                    "error": str(error),
                }
            )

    edges = []
    for source_name, target_name, edge_type_name in sorted(collector.relations):
        edges.append(
            {
                "source": node_name_to_id[source_name],
                "target": node_name_to_id[target_name],
                "source_name": source_name,
                "target_name": target_name,
                "edge_type": edge_type_name,
                "edge_type_id": EDGE_TYPE_TO_ID[edge_type_name],
            }
        )

    if edges:
        edge_index = np.array([[edge["source"] for edge in edges], [edge["target"] for edge in edges]], dtype=np.int64)
        edge_type = np.array([edge["edge_type_id"] for edge in edges], dtype=np.int64)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_type = np.zeros((0,), dtype=np.int64)

    validation_issues = validate_graph(dataset_name, x, y, edge_index, edge_type, edges)
    graph = {
        "graph_id": dataset_name,
        "dataset_name": dataset_name,
        "graph_type": "file_level_network_dependency_graph",
        "prediction_granularity": "node",
        "source_root": str(source_root) if source_root else None,
        "directed": True,
        "multi_relational": True,
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "node_feature_dim": int(x.shape[1]),
        "edge_type_vocab_size": len(EDGE_TYPE_TO_ID),
        "nodes": nodes,
        "edges": edges,
    }
    summary = {
        "dataset": dataset_name,
        "dataset_rows": int(len(dataset_rows)),
        "num_nodes": int(len(nodes)),
        "unmapped_rows": int(len(dataset_rows) - len(nodes)),
        "num_edges": int(len(edges)),
        "distinct_file_pairs": int(len({(edge["source"], edge["target"]) for edge in edges})),
        "node_feature_dim": int(x.shape[1]),
        "edge_type_vocab_size": len(EDGE_TYPE_TO_ID),
        "defective_nodes": int(y.sum()),
        "non_defective_nodes": int((y == 0).sum()),
        "strict_parses": int(strict_parses),
        "fallback_parses": int(len(parse_fallbacks)),
        "parse_failures": int(len(parse_failures)),
        "validation_issues": int(len(validation_issues)),
    }
    tensors = {"x": x, "y": y, "edge_index": edge_index, "edge_type": edge_type}
    return graph, tensors, summary, parse_fallbacks, parse_failures, validation_issues


def write_project_outputs(
    graph: dict[str, Any],
    tensors: dict[str, np.ndarray],
    graphs_dir: Path,
    tensors_dir: Path,
) -> dict[str, Any]:
    dataset_name = str(graph["dataset_name"])
    safe_name = sanitize_filename(dataset_name)
    graph_path = graphs_dir / f"{safe_name}.json"
    x_path = tensors_dir / f"{safe_name}_x.npy"
    y_path = tensors_dir / f"{safe_name}_y.npy"
    edge_index_path = tensors_dir / f"{safe_name}_edge_index.npy"
    edge_type_path = tensors_dir / f"{safe_name}_edge_type.npy"

    graph_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    np.save(x_path, tensors["x"])
    np.save(y_path, tensors["y"])
    np.save(edge_index_path, tensors["edge_index"])
    np.save(edge_type_path, tensors["edge_type"])

    return {
        "graph_id": dataset_name,
        "dataset_name": dataset_name,
        "num_nodes": int(graph["num_nodes"]),
        "num_edges": int(graph["num_edges"]),
        "node_feature_dim": int(graph["node_feature_dim"]),
        "edge_type_vocab_size": int(graph["edge_type_vocab_size"]),
        "graph_json": str(graph_path),
        "x_npy": str(x_path),
        "y_npy": str(y_path),
        "edge_index_npy": str(edge_index_path),
        "edge_type_npy": str(edge_type_path),
    }



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract project-level NDGs for PROMISE Java SDP datasets.")
    parser.add_argument("--input-csv", type=Path, default=Path("outputs/promise/promise_preprocessed_log1p.csv"))
    parser.add_argument("--preprocess-summary", type=Path, default=Path("outputs/promise/promise_preprocess_summary.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/promise/ndg"))
    parser.add_argument("--dataset-name", help="Optional dataset filter, e.g. ant-1.6")
    parser.add_argument("--no-clean", action="store_true", help="Do not clear existing graph/tensor files before extraction.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    input_csv = (repo_root / args.input_csv).resolve() if not args.input_csv.is_absolute() else args.input_csv.resolve()
    summary_path = (repo_root / args.preprocess_summary).resolve() if not args.preprocess_summary.is_absolute() else args.preprocess_summary.resolve()
    output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()

    if not input_csv.exists():
        raise FileNotFoundError(f"Missing NDG input CSV: {input_csv}. Run scripts/preprocess_promise.py first.")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing preprocessing summary: {summary_path}. Run scripts/preprocess_promise.py first.")

    graphs_dir, tensors_dir = prepare_output_dirs(output_dir, clean=not args.no_clean)
    input_df, mapped = load_mapped_rows(input_csv, args.dataset_name)
    features = feature_columns(input_df)
    source_roots = load_source_roots(summary_path)

    index_rows: list[dict[str, Any]] = []
    dataset_summaries: list[dict[str, Any]] = []
    parse_fallbacks: list[dict[str, str]] = []
    parse_failures: list[dict[str, str]] = []
    validation_issues: list[dict[str, str]] = []

    for dataset_name, dataset_rows in input_df.groupby("dataset_name", sort=True):
        dataset_name = str(dataset_name)
        project_mapped = mapped[mapped["dataset_name"].astype(str) == dataset_name].copy()
        if project_mapped.empty:
            continue
        graph, tensors, project_summary, fallbacks, failures, issues = build_project_ndg(
            dataset_name=dataset_name,
            dataset_rows=dataset_rows,
            mapped_rows=project_mapped,
            source_root=source_roots.get(dataset_name),
            features=features,
        )
        index_rows.append(write_project_outputs(graph, tensors, graphs_dir, tensors_dir))
        dataset_summaries.append(project_summary)
        parse_fallbacks.extend(fallbacks)
        parse_failures.extend(failures)
        validation_issues.extend(issues)

    graph_index = pd.DataFrame(index_rows)
    if not graph_index.empty:
        graph_index = graph_index.sort_values(by=["dataset_name"]).reset_index(drop=True)
    graph_index_path = output_dir / "graph_index.csv"
    graph_index.to_csv(graph_index_path, index=False)

    summary = {
        "input_csv": str(input_csv),
        "preprocess_summary": str(summary_path),
        "output_dir": str(output_dir),
        "dataset_filter": args.dataset_name,
        "input_rows": int(len(input_df)),
        "mapped_rows": int(len(mapped)),
        "graphs_generated": int(len(index_rows)),
        "total_nodes": int(sum(row["num_nodes"] for row in index_rows)),
        "total_edges": int(sum(row["num_edges"] for row in index_rows)),
        "node_feature_dim": len(features),
        "edge_type_vocab_size": len(EDGE_TYPE_TO_ID),
        "defective_nodes": int(sum(row["defective_nodes"] for row in dataset_summaries)),
        "non_defective_nodes": int(sum(row["non_defective_nodes"] for row in dataset_summaries)),
        "fallback_parses": int(len(parse_fallbacks)),
        "parse_failures": int(len(parse_failures)),
        "validation_issues": int(len(validation_issues)),
        "feature_names": features,
        "edge_types": EDGE_TYPES,
        "datasets": dataset_summaries,
    }

    (output_dir / "feature_names.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
    (output_dir / "edge_type_vocab.json").write_text(json.dumps(EDGE_TYPE_TO_ID, indent=2), encoding="utf-8")
    (output_dir / "parse_fallbacks.json").write_text(json.dumps(parse_fallbacks, indent=2), encoding="utf-8")
    (output_dir / "parse_failures.json").write_text(json.dumps(parse_failures, indent=2), encoding="utf-8")
    (output_dir / "validation_issues.json").write_text(json.dumps(validation_issues, indent=2), encoding="utf-8")
    (output_dir / "ndg_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        "PROMISE NDG extraction finished. "
        f"graphs_generated={len(index_rows)} nodes={summary['total_nodes']} edges={summary['total_edges']} "
        f"fallbacks={summary['fallback_parses']} failures={summary['parse_failures']} "
        f"validation_issues={summary['validation_issues']}"
    )
    print(f"index={graph_index_path}")
    print(f"summary={output_dir / 'ndg_summary.json'}")


if __name__ == "__main__":
    main()
