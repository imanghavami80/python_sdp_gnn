#!/usr/bin/env python3
"""Extract filtered AST graphs for multi-project PROMISE Java SDP datasets.

Default input:
    outputs/promise/promise_preprocessed_log1p.csv

Default output:
    outputs/promise/ast/

The extractor creates one graph per mapped Java class. It keeps the same tensor
contract as the earlier Log4j-only AST stage:
- x.npy: structural node features [depth, out_degree, has_identifier]
- node_type_id.npy: exact AST node type ids for a trainable embedding layer
- edge_index.npy: directed AST parent -> child edges
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import javalang
import numpy as np
import pandas as pd
from javalang.ast import Node

IMPORTANT_NODE_TYPES = {
    "CompilationUnit",
    "PackageDeclaration",
    "Import",
    "ClassDeclaration",
    "InterfaceDeclaration",
    "EnumDeclaration",
    "MethodDeclaration",
    "ConstructorDeclaration",
    "FieldDeclaration",
    "VariableDeclarator",
    "FormalParameter",
    "BlockStatement",
    "IfStatement",
    "ForStatement",
    "WhileStatement",
    "DoStatement",
    "SwitchStatement",
    "SwitchStatementCase",
    "TryStatement",
    "CatchClause",
    "SynchronizedStatement",
    "ReturnStatement",
    "ThrowStatement",
    "BreakStatement",
    "ContinueStatement",
    "StatementExpression",
    "LocalVariableDeclaration",
    "MethodInvocation",
    "SuperMethodInvocation",
    "ClassCreator",
    "Assignment",
    "BinaryOperation",
    "TernaryExpression",
    "Cast",
    "MemberReference",
    "This",
    "Literal",
    "ReferenceType",
    "BasicType",
    "TypeArgument",
    "Annotation",
}

NODE_TYPE_EMBEDDING_DIM = 32
NODE_TYPE_TO_ID = {node_type: i for i, node_type in enumerate(sorted(IMPORTANT_NODE_TYPES))}
STRUCTURAL_FEATURE_DIM = 3
MODEL_NODE_FEATURE_DIM = NODE_TYPE_EMBEDDING_DIM + STRUCTURAL_FEATURE_DIM


@dataclass(frozen=True)
class KeptNode:
    node_id: int
    node_type: str
    depth: int
    sibling_index: int
    parent_id: int | None
    text_hint: str


def iter_children(node: Node) -> Iterable[Node]:
    for child in node.children:
        if isinstance(child, Node):
            yield child
        elif isinstance(child, (list, tuple)):
            for item in child:
                if isinstance(item, Node):
                    yield item


def infer_text_hint(node: Node) -> str:
    for attr in ("name", "member", "value", "qualifier"):
        if hasattr(node, attr):
            value = getattr(node, attr)
            if value is not None:
                text = str(value)
                if text:
                    return text[:80]
    return ""


def sanitize_filename(value: str, max_len: int = 180) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    if len(safe) <= max_len:
        return safe
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"{safe[: max_len - 13]}_{digest}"


def graph_id(dataset_name: str, class_name: str) -> str:
    return f"{dataset_name}::{class_name}"


def is_important(node: Node) -> bool:
    return type(node).__name__ in IMPORTANT_NODE_TYPES


def extract_filtered_tree(root: Node) -> tuple[list[KeptNode], list[tuple[int, int]]]:
    kept_nodes: list[KeptNode] = []
    edges: list[tuple[int, int]] = []
    stack: list[tuple[Node, int, int | None, int]] = [(root, 0, None, 0)]

    while stack:
        node, depth, nearest_kept_parent, sibling_index = stack.pop()
        node_type = type(node).__name__
        current_kept_id = nearest_kept_parent

        if is_important(node):
            current_kept_id = len(kept_nodes)
            kept_nodes.append(
                KeptNode(
                    node_id=current_kept_id,
                    node_type=node_type,
                    depth=depth,
                    sibling_index=sibling_index,
                    parent_id=nearest_kept_parent,
                    text_hint=infer_text_hint(node),
                )
            )
            if nearest_kept_parent is not None:
                edges.append((nearest_kept_parent, current_kept_id))

        children = list(iter_children(node))
        for rev_idx, child in enumerate(reversed(children)):
            stack.append((child, depth + 1, current_kept_id, len(children) - rev_idx - 1))

    return kept_nodes, edges


def extract_fallback_structure(code: str) -> tuple[list[KeptNode], list[tuple[int, int]]]:
    """Build a coarse AST-like tree when `javalang` cannot parse a file."""
    nodes: list[KeptNode] = []
    edges: list[tuple[int, int]] = []

    def add(node_type: str, depth: int, parent_id: int | None, hint: str = "") -> int:
        node_id = len(nodes)
        nodes.append(
            KeptNode(
                node_id=node_id,
                node_type=node_type,
                depth=depth,
                sibling_index=0,
                parent_id=parent_id,
                text_hint=hint,
            )
        )
        if parent_id is not None:
            edges.append((parent_id, node_id))
        return node_id

    root = add("CompilationUnit", depth=0, parent_id=None)
    class_stack: list[tuple[int, int]] = [(root, 0)]
    current_method: int | None = None
    brace_depth = 0

    for raw_line in code.splitlines():
        line = raw_line.strip()
        if not line:
            brace_depth += raw_line.count("{") - raw_line.count("}")
            continue

        parent = class_stack[-1][0] if class_stack else root
        depth = len(class_stack)

        class_match = re.search(r"\b(class|interface|enum)\s+([A-Za-z_][A-Za-z0-9_]*)", line)
        if class_match:
            kind = class_match.group(1)
            node_type = {
                "class": "ClassDeclaration",
                "interface": "InterfaceDeclaration",
                "enum": "EnumDeclaration",
            }[kind]
            class_node = add(node_type, depth=depth, parent_id=parent, hint=class_match.group(2))
            class_stack.append((class_node, brace_depth + raw_line.count("{")))
            current_method = None

        method_like = re.search(
            r"\b([A-Za-z_][A-Za-z0-9_<>,\[\]\s]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*\)\s*\{?",
            line,
        )
        control_starters = ("if", "for", "while", "catch", "switch")
        if method_like and not any(line.startswith(starter) for starter in control_starters):
            current_method = add("MethodDeclaration", depth=depth + 1, parent_id=parent, hint=method_like.group(2))

        stmt_parent = current_method if current_method is not None else parent
        stmt_depth = depth + (2 if current_method is not None else 1)

        for keyword, node_type in (
            ("if", "IfStatement"),
            ("for", "ForStatement"),
            ("while", "WhileStatement"),
            ("switch", "SwitchStatement"),
            ("try", "TryStatement"),
            ("catch", "CatchClause"),
            ("return", "ReturnStatement"),
            ("throw", "ThrowStatement"),
            ("break", "BreakStatement"),
            ("continue", "ContinueStatement"),
        ):
            if re.search(rf"\b{keyword}\b", line):
                add(node_type, depth=stmt_depth, parent_id=stmt_parent, hint=keyword)

        if "(" in line and ")" in line and ";" in line:
            call = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", line)
            if call:
                add("MethodInvocation", depth=stmt_depth, parent_id=stmt_parent, hint=call.group(1))
        if "=" in line and not line.startswith("import "):
            add("Assignment", depth=stmt_depth, parent_id=stmt_parent, hint="=")

        brace_depth += raw_line.count("{") - raw_line.count("}")
        while len(class_stack) > 1 and brace_depth < class_stack[-1][1]:
            class_stack.pop()
            current_method = None

    return nodes, edges


def build_feature_matrix(
    nodes: Sequence[KeptNode],
    edges: Sequence[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    n_nodes = len(nodes)
    if n_nodes == 0:
        return np.zeros((0, STRUCTURAL_FEATURE_DIM), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    out_degree = [0] * n_nodes
    for src, _ in edges:
        out_degree[src] += 1

    x = np.zeros((n_nodes, STRUCTURAL_FEATURE_DIM), dtype=np.float32)
    node_type_ids = np.zeros((n_nodes,), dtype=np.int64)
    for node in nodes:
        i = node.node_id
        node_type_ids[i] = NODE_TYPE_TO_ID[node.node_type]
        x[i, 0] = float(node.depth)
        x[i, 1] = float(out_degree[i])
        x[i, 2] = 1.0 if node.text_hint and any(char.isalpha() for char in node.text_hint) else 0.0

    return x, node_type_ids


def to_mermaid(nodes: Sequence[KeptNode], edges: Sequence[tuple[int, int]], limit: int = 45) -> str:
    use_nodes = nodes[:limit]
    keep_ids = {node.node_id for node in use_nodes}
    kept_edges = [(src, dst) for src, dst in edges if src in keep_ids and dst in keep_ids]
    lines = ["graph TD"]
    for node in use_nodes:
        label = f"{node.node_type}#{node.node_id}"
        lines.append(f'  N{node.node_id}["{label}"]')
    for src, dst in kept_edges:
        lines.append(f"  N{src} --> N{dst}")
    return "\n".join(lines)


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


def load_mapped_rows(input_csv: Path, dataset_filter: str | None) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    required = {"dataset_name", "name", "source_path", "match_strategy"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required AST input columns: {missing}")
    if dataset_filter:
        df = df[df["dataset_name"].astype(str) == dataset_filter].copy()
    mapped = df[df["source_path"].notna()].copy()
    mapped = mapped[mapped["match_strategy"].astype(str) != "not_found"].copy()
    mapped = mapped[mapped["match_strategy"].astype(str) != "ambiguous_simple_name"].copy()
    return mapped.reset_index(drop=True)


def extract_one(row: pd.Series, graphs_dir: Path, tensors_dir: Path) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, str] | None]:
    dataset_name = str(row["dataset_name"])
    class_name = str(row["name"])
    source_path = Path(str(row["source_path"]))
    gid = graph_id(dataset_name, class_name)

    try:
        code = source_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:  # noqa: BLE001
        return {}, None, {"dataset_name": dataset_name, "name": class_name, "source_path": str(source_path), "error": str(exc)}

    parser_mode = "javalang"
    fallback_record: dict[str, Any] | None = None
    try:
        tree = javalang.parse.parse(code)
        nodes, edges = extract_filtered_tree(tree)
    except Exception as strict_exc:  # noqa: BLE001
        parser_mode = "fallback"
        nodes, edges = extract_fallback_structure(code)
        fallback_record = {
            "dataset_name": dataset_name,
            "name": class_name,
            "source_path": str(source_path),
            "strict_parser_error": str(strict_exc),
        }

    try:
        x, node_type_ids = build_feature_matrix(nodes, edges)
        edge_index = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
        file_stem = sanitize_filename(gid)
        graph_path = graphs_dir / f"{file_stem}.json"
        x_path = tensors_dir / f"{file_stem}_x.npy"
        node_type_path = tensors_dir / f"{file_stem}_node_type_id.npy"
        edge_index_path = tensors_dir / f"{file_stem}_edge_index.npy"

        graph = {
            "graph_id": gid,
            "dataset_name": dataset_name,
            "name": class_name,
            "source_path": str(source_path),
            "label": int(row["bug"]) if "bug" in row and pd.notna(row["bug"]) else None,
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "feature_dim": int(x.shape[1]),
            "parser_mode": parser_mode,
            "nodes": [
                {
                    "id": node.node_id,
                    "type": node.node_type,
                    "node_type_id": NODE_TYPE_TO_ID[node.node_type],
                    "depth": node.depth,
                    "parent_id": node.parent_id,
                }
                for node in nodes
            ],
            "edges": [[src, dst] for src, dst in edges],
        }
        graph_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
        np.save(x_path, x)
        np.save(node_type_path, node_type_ids)
        np.save(edge_index_path, edge_index)

        index_row = {
            "graph_id": gid,
            "dataset_name": dataset_name,
            "name": class_name,
            "source_path": str(source_path),
            "label": graph["label"],
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "feature_dim": int(x.shape[1]),
            "parser_mode": parser_mode,
            "graph_json": str(graph_path),
            "x_npy": str(x_path),
            "node_type_id_npy": str(node_type_path),
            "edge_index_npy": str(edge_index_path),
        }
        return index_row, fallback_record, None
    except Exception as exc:  # noqa: BLE001
        return {}, fallback_record, {"dataset_name": dataset_name, "name": class_name, "source_path": str(source_path), "error": str(exc)}


def build_report(summary: dict[str, Any], graph_rows: list[dict[str, Any]], top_types: list[tuple[str, int]]) -> str:
    dataset_rows = [
        "| `{dataset}` | {requested} | {graphs} | {fallbacks} | {failures} |".format(**row)
        for row in summary["datasets"]
    ]
    top_type_rows = [f"- `{node_type}`: {count}" for node_type, count in top_types[:15]]

    sample_sections: list[str] = []
    for idx, row in enumerate(graph_rows[:3], start=1):
        graph = json.loads(Path(row["graph_json"]).read_text(encoding="utf-8"))
        nodes = [
            KeptNode(
                node_id=int(node["id"]),
                node_type=str(node["type"]),
                depth=int(node["depth"]),
                sibling_index=0,
                parent_id=node.get("parent_id"),
                text_hint="",
            )
            for node in graph["nodes"]
        ]
        edges = [tuple(edge) for edge in graph["edges"]]
        sample_sections.extend(
            [
                f"### {idx}. `{row['graph_id']}`",
                "```mermaid",
                to_mermaid(nodes, edges, limit=45),
                "```",
                "",
            ]
        )

    lines = [
        "# AST Extraction Report for Multi-Project PROMISE Dataset",
        "",
        "## 1. Objective",
        "This report describes the generalized AST extraction stage for the multi-project PROMISE SDP dataset. "
        "The extractor creates one AST graph for every preprocessed row that has a mapped Java source file.",
        "",
        "## 2. Input Data",
        f"- Input CSV: `{summary['input_csv']}`",
        f"- Output directory: `{summary['output_dir']}`",
        f"- Dataset rows: {summary['input_rows']}",
        f"- Mapped rows requested: {summary['requested_mapped_samples']}",
        f"- Graphs generated: {summary['graphs_generated']}",
        "",
        "## 3. Extraction Pipeline",
        "- Read the combined preprocessed PROMISE CSV.",
        "- Keep rows with valid `source_path` values and non-failed source matching.",
        "- Parse each Java source file using `javalang`.",
        "- Keep a curated set of syntax-relevant AST node types.",
        "- Reconnect each retained node to the nearest retained ancestor.",
        "- Save graph JSON plus GNN-ready tensors.",
        "- If strict parsing fails, create a deterministic fallback AST-like graph and record it in `parse_fallbacks.json`.",
        "",
        "```text",
        "combined PROMISE CSV -> mapped Java files -> filtered AST graphs -> graph/tensor dataset",
        "```",
        "",
        "## 4. Node Features",
        "The stored structural tensor is deliberately compact:",
        "",
        "```text",
        "x(node) = [depth, out_degree, has_identifier]",
        "```",
        "",
        "The exact AST node type is stored separately in `node_type_id.npy`. During GNN training, use a trainable "
        f"`Embedding(vocab_size={summary['node_type_vocab_size']}, embedding_dim={summary['node_type_embedding_dim']})` "
        "and concatenate the learned type embedding with the 3 structural features.",
        "",
        "```text",
        "complete_node_x = concat(node_type_embedding(node_type_id), structural_x)",
        "```",
        "",
        f"- Structural feature dimension: {summary['structural_feature_dim']}",
        f"- Node type embedding dimension: {summary['node_type_embedding_dim']}",
        f"- Model node feature dimension after concatenation: {summary['model_node_feature_dim']}",
        "",
        "## 5. Output Files",
        "- `graph_index.csv`: one row per generated AST graph with graph/tensor paths.",
        "- `graphs/*.json`: readable graph files with metadata, nodes, and edges.",
        "- `tensors/*_x.npy`: structural node features.",
        "- `tensors/*_node_type_id.npy`: exact AST node type ids.",
        "- `tensors/*_edge_index.npy`: AST parent-child connectivity.",
        "- `node_type_vocab.json`: stable exact AST node type vocabulary.",
        "- `ast_summary.json`: global extraction statistics.",
        "- `parse_failures.json`: hard failures.",
        "- `parse_fallbacks.json`: strict parser failures recovered by fallback extraction.",
        "",
        "## 6. Results",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Input rows | {summary['input_rows']} |",
        f"| Mapped rows requested | {summary['requested_mapped_samples']} |",
        f"| Graphs generated | {summary['graphs_generated']} |",
        f"| Parse failures | {summary['parse_failures']} |",
        f"| Fallback graphs | {summary['fallback_graphs']} |",
        f"| Total nodes | {summary['total_nodes']} |",
        f"| Total edges | {summary['total_edges']} |",
        f"| Average nodes per graph | {summary['avg_nodes_per_graph']:.2f} |",
        f"| Average edges per graph | {summary['avg_edges_per_graph']:.2f} |",
        "",
        "## 7. Per-Dataset Results",
        "| Dataset | Requested | Graphs | Fallbacks | Failures |",
        "| --- | ---: | ---: | ---: | ---: |",
        *dataset_rows,
        "",
        "## 8. Top Node Types",
        *top_type_rows,
        "",
        "## 9. Notes",
        "- This AST view captures syntax, not execution order or file-level dependency behavior.",
        "- Keep behavior in CFG/NDG edge types instead of encoding it into AST node features.",
        "- Fallback graphs preserve dataset coverage but are less precise than strict `javalang` graphs.",
        "",
        "## 10. Sample Visualizations",
        *sample_sections,
    ]
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract AST graphs for PROMISE Java SDP datasets.")
    parser.add_argument("--input-csv", type=Path, default=Path("outputs/promise/promise_preprocessed_log1p.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/promise/ast"))
    parser.add_argument("--dataset-name", help="Optional dataset filter, e.g. ant-1.6")
    parser.add_argument("--no-clean", action="store_true", help="Do not clear existing graph/tensor files before extraction.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    input_csv = (repo_root / args.input_csv).resolve() if not args.input_csv.is_absolute() else args.input_csv.resolve()
    output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()

    if not input_csv.exists():
        raise FileNotFoundError(f"Missing AST input CSV: {input_csv}. Run scripts/preprocess_promise.py first.")

    graphs_dir, tensors_dir = prepare_output_dirs(output_dir, clean=not args.no_clean)
    (output_dir / "node_type_vocab.json").write_text(json.dumps(NODE_TYPE_TO_ID, indent=2), encoding="utf-8")

    input_df = pd.read_csv(input_csv)
    mapped = load_mapped_rows(input_csv, args.dataset_name)

    graph_rows: list[dict[str, Any]] = []
    parse_failures: list[dict[str, str]] = []
    parse_fallbacks: list[dict[str, str]] = []
    type_counter: Counter[str] = Counter()

    for _, row in mapped.iterrows():
        index_row, fallback_record, failure_record = extract_one(row, graphs_dir, tensors_dir)
        if fallback_record is not None:
            parse_fallbacks.append(fallback_record)
        if failure_record is not None:
            parse_failures.append(failure_record)
            continue
        graph_rows.append(index_row)
        graph = json.loads(Path(index_row["graph_json"]).read_text(encoding="utf-8"))
        type_counter.update(node["type"] for node in graph["nodes"])

    graph_index = pd.DataFrame(graph_rows)
    if not graph_index.empty:
        graph_index = graph_index.sort_values(by=["dataset_name", "name"]).reset_index(drop=True)
    graph_index_path = output_dir / "graph_index.csv"
    graph_index.to_csv(graph_index_path, index=False)

    dataset_summaries: list[dict[str, Any]] = []
    requested_by_dataset = mapped["dataset_name"].astype(str).value_counts().to_dict()
    graphs_by_dataset = graph_index["dataset_name"].astype(str).value_counts().to_dict() if not graph_index.empty else {}
    fallbacks_by_dataset = Counter(item["dataset_name"] for item in parse_fallbacks)
    failures_by_dataset = Counter(item["dataset_name"] for item in parse_failures)
    for dataset in sorted(requested_by_dataset):
        dataset_summaries.append(
            {
                "dataset": dataset,
                "requested": int(requested_by_dataset.get(dataset, 0)),
                "graphs": int(graphs_by_dataset.get(dataset, 0)),
                "fallbacks": int(fallbacks_by_dataset.get(dataset, 0)),
                "failures": int(failures_by_dataset.get(dataset, 0)),
            }
        )

    total_nodes = int(graph_index["num_nodes"].sum()) if not graph_index.empty else 0
    total_edges = int(graph_index["num_edges"].sum()) if not graph_index.empty else 0
    top_types = type_counter.most_common(20)
    summary = {
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "dataset_filter": args.dataset_name,
        "input_rows": int(len(input_df)),
        "requested_mapped_samples": int(len(mapped)),
        "graphs_generated": int(len(graph_rows)),
        "parse_failures": int(len(parse_failures)),
        "fallback_graphs": int(len(parse_fallbacks)),
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "avg_nodes_per_graph": float(total_nodes / len(graph_rows)) if graph_rows else 0.0,
        "avg_edges_per_graph": float(total_edges / len(graph_rows)) if graph_rows else 0.0,
        "structural_feature_dim": STRUCTURAL_FEATURE_DIM,
        "node_type_embedding_dim": NODE_TYPE_EMBEDDING_DIM,
        "model_node_feature_dim": MODEL_NODE_FEATURE_DIM,
        "node_type_vocab_size": len(NODE_TYPE_TO_ID),
        "important_node_types": sorted(IMPORTANT_NODE_TYPES),
        "top_node_types": top_types,
        "datasets": dataset_summaries,
    }

    (output_dir / "ast_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "parse_failures.json").write_text(json.dumps(parse_failures, indent=2), encoding="utf-8")
    (output_dir / "parse_fallbacks.json").write_text(json.dumps(parse_fallbacks, indent=2), encoding="utf-8")
    (output_dir / "ast_report.md").write_text(build_report(summary, graph_rows, top_types), encoding="utf-8")

    print(
        "PROMISE AST extraction finished. "
        f"datasets={len(dataset_summaries)} graphs_generated={len(graph_rows)} "
        f"fallbacks={len(parse_fallbacks)} failures={len(parse_failures)}"
    )
    print(f"index={graph_index_path}")
    print(f"summary={output_dir / 'ast_summary.json'}")
    print(f"report={output_dir / 'ast_report.md'}")


if __name__ == "__main__":
    main()
