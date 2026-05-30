#!/usr/bin/env python3
"""Extract filtered AST graphs for mapped Log4j classes.

Outputs (default: outputs/log4j/ast):
- graphs/<class_name>.json              : node/edge list with metadata
- tensors/<class_name>_x.npy            : structural node features [num_nodes, 3]
- tensors/<class_name>_node_type_id.npy : exact AST node type ids [num_nodes]
- tensors/<class_name>_edge_index.npy   : edge index [2, num_edges]
- node_type_vocab.json                  : stable exact AST node type -> id mapping
- graph_index.csv                       : graph-level summary and paths
- ast_summary.json                      : global statistics
- parse_fallbacks.json                  : strict parser failures recovered by fallback
- ast_report.md                         : concise implementation report + sample Mermaid views
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import javalang
import numpy as np
import pandas as pd
from javalang.ast import Node

IMPORTANT_NODE_TYPES = {
    # Structure
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
    # Statements / control flow
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
    # Expressions
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
    # Types / annotations
    "ReferenceType",
    "BasicType",
    "TypeArgument",
    "Annotation",
}

NODE_TYPE_EMBEDDING_DIM = 32
NODE_TYPE_TO_ID = {node_type: i for i, node_type in enumerate(sorted(IMPORTANT_NODE_TYPES))}
STRUCTURAL_FEATURE_DIM = 3
MODEL_NODE_FEATURE_DIM = NODE_TYPE_EMBEDDING_DIM + STRUCTURAL_FEATURE_DIM


@dataclass
class KeptNode:
    node_id: int
    node_type: str
    depth: int
    sibling_index: int
    parent_id: Optional[int]
    text_hint: str


def iter_children(node: Node) -> Iterable[Node]:
    for child in node.children:
        if isinstance(child, Node):
            yield child
        elif isinstance(child, (list, tuple)):
            for x in child:
                if isinstance(x, Node):
                    yield x


def infer_text_hint(node: Node) -> str:
    for attr in ("name", "member", "value", "qualifier", "position"):
        if hasattr(node, attr):
            val = getattr(node, attr)
            if val is None:
                continue
            s = str(val)
            if s:
                return s[:60]
    return ""


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def is_important(node: Node) -> bool:
    return type(node).__name__ in IMPORTANT_NODE_TYPES


def extract_filtered_tree(root: Node) -> Tuple[List[KeptNode], List[Tuple[int, int]]]:
    kept_nodes: List[KeptNode] = []
    edges: List[Tuple[int, int]] = []

    stack: List[Tuple[Node, int, Optional[int], int]] = [(root, 0, None, 0)]

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


def build_feature_matrix(
    nodes: Sequence[KeptNode],
    edges: Sequence[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(nodes)
    if n == 0:
        return np.zeros((0, STRUCTURAL_FEATURE_DIM), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    out_deg = [0] * n
    for src, _ in edges:
        out_deg[src] += 1

    x = np.zeros((n, STRUCTURAL_FEATURE_DIM), dtype=np.float32)
    node_type_ids = np.zeros((n,), dtype=np.int64)
    for node in nodes:
        i = node.node_id
        node_type_id = NODE_TYPE_TO_ID[node.node_type]
        node_type_ids[i] = node_type_id

        hint = node.text_hint
        x[i, 0] = node.depth
        x[i, 1] = out_deg[i]
        x[i, 2] = 1.0 if hint and any(c.isalpha() for c in hint) else 0.0

    return x, node_type_ids


def to_mermaid(nodes: Sequence[KeptNode], edges: Sequence[Tuple[int, int]], limit: int = 60) -> str:
    use_nodes = nodes[:limit]
    keep_ids = {n.node_id for n in use_nodes}
    kept_edges = [(u, v) for u, v in edges if u in keep_ids and v in keep_ids]

    lines = ["graph TD"]
    for n in use_nodes:
        label = f"{n.node_type}#{n.node_id}"
        lines.append(f"  N{n.node_id}[\"{label}\"]")
    for u, v in kept_edges:
        lines.append(f"  N{u} --> N{v}")
    return "\n".join(lines)


def extract_fallback_structure(code: str) -> Tuple[List[KeptNode], List[Tuple[int, int]]]:
    """Build a coarse AST-like tree when strict Java parsing fails.

    This keeps dataset completeness for downstream GNN training while preserving
    a declaration/control/expression hierarchy.
    """
    nodes: List[KeptNode] = []
    edges: List[Tuple[int, int]] = []

    def add(node_type: str, depth: int, parent_id: Optional[int], hint: str = "") -> int:
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
    class_stack: List[Tuple[int, int]] = [(root, 0)]  # (node_id, brace_depth)
    current_method: Optional[int] = None
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
            if kind == "class":
                node_type = "ClassDeclaration"
            elif kind == "interface":
                node_type = "InterfaceDeclaration"
            else:
                node_type = "EnumDeclaration"
            class_node = add(node_type, depth=depth, parent_id=parent, hint=class_match.group(2))
            class_stack.append((class_node, brace_depth + raw_line.count("{")))
            current_method = None

        method_like = re.search(r"\b([A-Za-z_][A-Za-z0-9_<>,\[\]\s]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*\)\s*\{?", line)
        if method_like and not any(x in line for x in (" if ", " for ", " while ", " catch ", " switch ")) and not line.startswith("if") and not line.startswith("for") and not line.startswith("while") and not line.startswith("switch"):
            current_method = add("MethodDeclaration", depth=depth + 1, parent_id=parent, hint=method_like.group(2))

        stmt_parent = current_method if current_method is not None else parent
        stmt_depth = depth + (2 if current_method is not None else 1)

        for kw, node_type in (
            ("if", "IfStatement"),
            ("for", "ForStatement"),
            ("while", "WhileStatement"),
            ("switch", "SwitchStatement"),
            ("try", "TryStatement"),
            ("catch", "CatchClause"),
            ("return", "ReturnStatement"),
            ("throw", "ThrowStatement"),
        ):
            if re.search(rf"\b{kw}\b", line):
                add(node_type, depth=stmt_depth, parent_id=stmt_parent, hint=kw)

        if "(" in line and ")" in line and ";" in line:
            call = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", line)
            if call:
                add("MethodInvocation", depth=stmt_depth, parent_id=stmt_parent, hint=call.group(1))
        if "=" in line and not line.startswith("import "):
            add("Assignment", depth=stmt_depth, parent_id=stmt_parent, hint="=")

        delta = raw_line.count("{") - raw_line.count("}")
        brace_depth += delta
        while len(class_stack) > 1 and brace_depth < class_stack[-1][1]:
            class_stack.pop()
            current_method = None

    return nodes, edges


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    mapping_csv = repo_root / "outputs/log4j/log4j_name_to_source_mapping.csv"
    out_dir = repo_root / "outputs/log4j/ast"
    graphs_dir = out_dir / "graphs"
    tensors_dir = out_dir / "tensors"

    out_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    tensors_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = out_dir / "node_type_vocab.json"
    vocab_path.write_text(json.dumps(NODE_TYPE_TO_ID, indent=2))
    legacy_embeddings_path = out_dir / "node_type_embeddings.npy"
    if legacy_embeddings_path.exists():
        legacy_embeddings_path.unlink()

    mapping_df = pd.read_csv(mapping_csv)
    mapped = mapping_df[mapping_df["source_path"].notna()].copy()

    graph_rows: List[Dict[str, Any]] = []
    total_nodes = 0
    total_edges = 0
    parse_failures: List[Dict[str, str]] = []
    parse_fallbacks: List[Dict[str, str]] = []
    type_counter: Dict[str, int] = {}

    for _, row in mapped.iterrows():
        name = row["name"]
        source_path = Path(row["source_path"])

        try:
            code = source_path.read_text(encoding="utf-8", errors="ignore")
            parser_mode = "javalang"
            try:
                tree = javalang.parse.parse(code)
                nodes, edges = extract_filtered_tree(tree)
            except Exception as strict_exc:
                parser_mode = "fallback"
                nodes, edges = extract_fallback_structure(code)
                parse_fallbacks.append(
                    {
                        "name": name,
                        "source_path": str(source_path),
                        "strict_parser_error": str(strict_exc),
                    }
                )
            x, node_type_ids = build_feature_matrix(nodes, edges)
            edge_index = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)

            fname = sanitize_filename(name)
            graph_path = graphs_dir / f"{fname}.json"
            x_path = tensors_dir / f"{fname}_x.npy"
            type_ids_path = tensors_dir / f"{fname}_node_type_id.npy"
            e_path = tensors_dir / f"{fname}_edge_index.npy"

            graph_obj = {
                "name": name,
                "source_path": str(source_path),
                "num_nodes": len(nodes),
                "num_edges": len(edges),
                "feature_dim": int(x.shape[1]),
                "parser_mode": parser_mode,
                "nodes": [
                    {
                        "id": n.node_id,
                        "type": n.node_type,
                        "node_type_id": NODE_TYPE_TO_ID[n.node_type],
                        "depth": n.depth,
                        "parent_id": n.parent_id,
                    }
                    for n in nodes
                ],
                "edges": [[u, v] for u, v in edges],
            }
            graph_path.write_text(json.dumps(graph_obj, indent=2))
            np.save(x_path, x)
            np.save(type_ids_path, node_type_ids)
            np.save(e_path, edge_index)

            for n in nodes:
                type_counter[n.node_type] = type_counter.get(n.node_type, 0) + 1

            total_nodes += len(nodes)
            total_edges += len(edges)
            graph_rows.append(
                {
                    "name": name,
                    "source_path": str(source_path),
                    "num_nodes": len(nodes),
                    "num_edges": len(edges),
                    "feature_dim": int(x.shape[1]),
                    "parser_mode": parser_mode,
                    "graph_json": str(graph_path),
                    "x_npy": str(x_path),
                    "node_type_id_npy": str(type_ids_path),
                    "edge_index_npy": str(e_path),
                }
            )
        except Exception as exc:  # noqa: BLE001
            parse_failures.append({"name": name, "source_path": str(source_path), "error": str(exc)})

    graph_index = pd.DataFrame(graph_rows).sort_values(by="name")
    graph_index_path = out_dir / "graph_index.csv"
    graph_index.to_csv(graph_index_path, index=False)

    top_types = sorted(type_counter.items(), key=lambda kv: kv[1], reverse=True)[:20]
    summary = {
        "requested_mapped_samples": int(len(mapped)),
        "graphs_generated": int(len(graph_rows)),
        "parse_failures": int(len(parse_failures)),
        "total_nodes": int(total_nodes),
        "total_edges": int(total_edges),
        "avg_nodes_per_graph": float(total_nodes / len(graph_rows)) if graph_rows else 0.0,
        "avg_edges_per_graph": float(total_edges / len(graph_rows)) if graph_rows else 0.0,
        "structural_feature_dim": int(graph_rows[0]["feature_dim"]) if graph_rows else 0,
        "node_type_embedding_dim": NODE_TYPE_EMBEDDING_DIM,
        "model_node_feature_dim": MODEL_NODE_FEATURE_DIM,
        "node_type_vocab_size": len(NODE_TYPE_TO_ID),
        "fallback_graphs": int(sum(1 for r in graph_rows if r.get("parser_mode") == "fallback")),
        "strict_parse_fallbacks": int(len(parse_fallbacks)),
        "important_node_types": sorted(IMPORTANT_NODE_TYPES),
        "top_node_types": top_types,
    }
    summary_path = out_dir / "ast_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    failures_path = out_dir / "parse_failures.json"
    failures_path.write_text(json.dumps(parse_failures, indent=2))
    fallbacks_path = out_dir / "parse_fallbacks.json"
    fallbacks_path.write_text(json.dumps(parse_fallbacks, indent=2))

    report_lines: List[str] = []
    report_lines.append("# AST Extraction Report for Log4j 1.0")
    report_lines.append("")
    report_lines.append("## 1. Objective")
    report_lines.append(
        "This report describes the AST graph extraction step for the Log4j 1.0 "
        "PROMISE dataset used in the multi-view software defect prediction pipeline. "
        "The goal is to create one AST graph for each dataset row that has a matched "
        "Java source file, then store the result in a format that can be used directly "
        "by a Graph Neural Network."
    )
    report_lines.append("")
    report_lines.append("## 2. Input Data")
    report_lines.append("- Dataset: `projects/log4j/log4j-1.0.csv`")
    report_lines.append("- Preprocessed dataset: `outputs/log4j/log4j_preprocessed_standard.csv`")
    report_lines.append("- Class-to-source mapping: `outputs/log4j/log4j_name_to_source_mapping.csv`")
    report_lines.append(f"- Mapped source files used for AST extraction: {summary['requested_mapped_samples']}")
    report_lines.append("")
    report_lines.append("## 3. Extraction Pipeline")
    report_lines.append("- Each mapped Java file is parsed with `javalang`.")
    report_lines.append("- Only important AST node types are kept to reduce noise and graph size.")
    report_lines.append("- Removed intermediate nodes are bypassed by reconnecting each kept node to the nearest kept ancestor.")
    report_lines.append("- Parent-child AST relations are saved as directed edges from parent node to child node.")
    report_lines.append("- If strict parsing fails, a deterministic fallback parser is used and the graph is marked with `parser_mode=fallback`.")
    report_lines.append("")
    report_lines.append("## 4. Selected AST Node Types")
    report_lines.append(
        "The extractor keeps nodes that are useful for defect prediction: declarations, "
        "control-flow statements, expressions, method calls, variable references, literals, "
        "and type information. This keeps the graph focused on program structure and behavior "
        "instead of low-value parser details."
    )
    report_lines.append("")
    report_lines.append("| Group | Examples | Why it matters |")
    report_lines.append("| --- | --- | --- |")
    report_lines.append("| Declaration | `ClassDeclaration`, `MethodDeclaration`, `FieldDeclaration` | Captures class and API structure. |")
    report_lines.append("| Control | `IfStatement`, `ForStatement`, `TryStatement`, `ReturnStatement` | Captures branching, loops, exceptions, and exits. |")
    report_lines.append("| Expression | `MethodInvocation`, `Assignment`, `BinaryOperation`, `MemberReference` | Captures behavior inside methods. |")
    report_lines.append("| Type | `ReferenceType`, `BasicType`, `Annotation` | Preserves type-level context. |")
    report_lines.append("| Literal | `Literal` | Preserves constants and string/numeric usage. |")
    report_lines.append("")
    report_lines.append("## 5. What the Tensor Files Mean")
    report_lines.append(
        "A tensor is a numeric array used by machine learning frameworks. In this project, "
        "each AST graph is stored in tensor files so a GNN can process it efficiently."
    )
    report_lines.append("")
    report_lines.append("| File | Shape | Meaning |")
    report_lines.append("| --- | --- | --- |")
    report_lines.append("| `*_x.npy` | `[num_nodes, 3]` | Structural syntax features: depth, out-degree, and has-identifier. |")
    report_lines.append("| `*_node_type_id.npy` | `[num_nodes]` | Exact AST node type ids consumed by a trainable embedding layer. |")
    report_lines.append("| `*_edge_index.npy` | `[2, num_edges]` | Graph connectivity. Row 0 stores source node ids; row 1 stores target node ids. |")
    report_lines.append("")
    report_lines.append(
        "The extractor stores syntax-focused inputs. During model training, transform each "
        "`node_type_id` with `Embedding(vocab_size, 32)` and concatenate the result with "
        "the three structural features. The GNN then receives 35 values per node."
    )
    report_lines.append("")
    report_lines.append("| Model input index range | Feature | Description |")
    report_lines.append("| --- | --- | --- |")
    report_lines.append("| `0-31` | Learned node type embedding | Trainable representation of the exact retained AST node type. |")
    report_lines.append("| `32` | Depth | How deep the node is in the AST. |")
    report_lines.append("| `33` | Out-degree | How many kept child nodes it has. |")
    report_lines.append("| `34` | Has identifier | Whether the node carries identifier-like text. |")
    report_lines.append("")
    report_lines.append(
        "Exact node types are not collapsed into broad groups. Each type has a stable id "
        "in `node_type_vocab.json`, for example `MethodDeclaration`, `IfStatement`, "
        "`Assignment`, and `MethodInvocation`. The embedding table belongs in the training "
        "model and is learned from the defect prediction objective."
    )
    report_lines.append("")
    report_lines.append("Example for one graph:")
    report_lines.append("")
    report_lines.append("```python")
    report_lines.append("import numpy as np")
    report_lines.append("")
    report_lines.append("x = np.load('outputs/log4j/ast/tensors/org.apache.log4j.AppenderSkeleton_x.npy')")
    report_lines.append("node_type_id = np.load('outputs/log4j/ast/tensors/org.apache.log4j.AppenderSkeleton_node_type_id.npy')")
    report_lines.append("edge_index = np.load('outputs/log4j/ast/tensors/org.apache.log4j.AppenderSkeleton_edge_index.npy')")
    report_lines.append("")
    report_lines.append("print(x.shape)            # (num_nodes, 3)")
    report_lines.append("print(node_type_id.shape) # (num_nodes,)")
    report_lines.append("print(edge_index.shape) # (2, num_edges)")
    report_lines.append("```")
    report_lines.append("")
    report_lines.append("## 6. How to Use the AST Tensors in a GNN")
    report_lines.append(
        "The complete node representation is assembled inside the model, not during AST "
        "extraction. This is necessary because the 32 embedding values must be learned from "
        "the defect prediction task. They are model parameters, while depth, out-degree, and "
        "has-identifier are fixed facts extracted from the source code."
    )
    report_lines.append("")
    report_lines.append("For a node with `node_type_id = 24` and structural features `[2, 1, 1]`:")
    report_lines.append("")
    report_lines.append("```text")
    report_lines.append("node_type_id = 24")
    report_lines.append("structural_x = [depth=2, out_degree=1, has_identifier=1]")
    report_lines.append("type_embedding = embedding_layer(24)  # learned vector with 32 values")
    report_lines.append("complete_node_x = concat(type_embedding, structural_x)  # 35 values")
    report_lines.append("```")
    report_lines.append("")
    report_lines.append("At model level, the processing steps are:")
    report_lines.append("")
    report_lines.append("1. Load `node_type_id`, structural `x`, and `edge_index` for each AST graph.")
    report_lines.append("2. Convert every `node_type_id` to a learned 32-dimensional embedding.")
    report_lines.append("3. Concatenate each embedding with its three structural values.")
    report_lines.append("4. Pass the resulting 35-dimensional node vectors and AST edges into GNN layers.")
    report_lines.append("5. Pool all node representations into one graph representation for the Java class.")
    report_lines.append("6. Feed the graph representation into a classifier to predict whether the class is defective.")
    report_lines.append("")
    report_lines.append("Minimal PyTorch-style model preparation:")
    report_lines.append("")
    report_lines.append("```python")
    report_lines.append("import torch")
    report_lines.append("from torch import nn")
    report_lines.append("")
    report_lines.append("class AstNodeEncoder(nn.Module):")
    report_lines.append("    def __init__(self, node_type_vocab_size: int, embedding_dim: int = 32):")
    report_lines.append("        super().__init__()")
    report_lines.append("        self.node_type_embedding = nn.Embedding(node_type_vocab_size, embedding_dim)")
    report_lines.append("")
    report_lines.append("    def forward(self, node_type_id: torch.Tensor, structural_x: torch.Tensor):")
    report_lines.append("        type_x = self.node_type_embedding(node_type_id)")
    report_lines.append("        return torch.cat([type_x, structural_x], dim=1)")
    report_lines.append("")
    report_lines.append("# node_type_id: [num_nodes]")
    report_lines.append("# structural_x: [num_nodes, 3]")
    report_lines.append("# complete_x:   [num_nodes, 35]")
    report_lines.append("complete_x = AstNodeEncoder(node_type_vocab_size=41)(node_type_id, structural_x)")
    report_lines.append("```")
    report_lines.append("")
    report_lines.append(
        "The returned `complete_x` is the node feature matrix passed to the first GNN layer "
        "together with `edge_index`. The future CFG and NDG views should follow the same "
        "principle: keep node attributes compact and represent program behavior primarily "
        "through graph edges."
    )
    report_lines.append("")
    report_lines.append("## 7. Output Files")
    report_lines.append("- `outputs/log4j/ast/graph_index.csv`: index of all generated graphs and tensor paths.")
    report_lines.append("- `outputs/log4j/ast/graphs/*.json`: readable graph files with node metadata and edge lists.")
    report_lines.append("- `outputs/log4j/ast/tensors/*_x.npy`: structural syntax feature tensors.")
    report_lines.append("- `outputs/log4j/ast/tensors/*_node_type_id.npy`: exact node type id tensors for trainable embeddings.")
    report_lines.append("- `outputs/log4j/ast/tensors/*_edge_index.npy`: edge index tensors.")
    report_lines.append("- `outputs/log4j/ast/node_type_vocab.json`: stable mapping from exact AST node type to id.")
    report_lines.append("- `outputs/log4j/ast/ast_summary.json`: global extraction statistics.")
    report_lines.append("- `outputs/log4j/ast/parse_failures.json`: hard failures where both strict and fallback parsing failed.")
    report_lines.append("- `outputs/log4j/ast/parse_fallbacks.json`: files where strict parsing failed but fallback extraction succeeded.")
    report_lines.append("")
    report_lines.append("## 8. Results")
    report_lines.append("")
    report_lines.append("| Metric | Value |")
    report_lines.append("| --- | ---: |")
    report_lines.append(f"| Mapped samples requested | {summary['requested_mapped_samples']} |")
    report_lines.append(f"| Graphs generated | {summary['graphs_generated']} |")
    report_lines.append(f"| Parse failures | {summary['parse_failures']} |")
    report_lines.append(f"| Fallback graphs | {summary['fallback_graphs']} |")
    report_lines.append(f"| Strict parser failures recovered by fallback | {summary['strict_parse_fallbacks']} |")
    report_lines.append(f"| Total nodes | {summary['total_nodes']} |")
    report_lines.append(f"| Total edges | {summary['total_edges']} |")
    report_lines.append(f"| Average nodes per graph | {summary['avg_nodes_per_graph']:.2f} |")
    report_lines.append(f"| Average edges per graph | {summary['avg_edges_per_graph']:.2f} |")
    report_lines.append(f"| Extracted structural feature dimension | {summary['structural_feature_dim']} |")
    report_lines.append(f"| Learned node type embedding dimension | {summary['node_type_embedding_dim']} |")
    report_lines.append(f"| Model node feature dimension after concatenation | {summary['model_node_feature_dim']} |")
    report_lines.append(f"| Exact AST node type vocabulary size | {summary['node_type_vocab_size']} |")
    report_lines.append("")
    report_lines.append("## 9. Top Node Types")
    for t, c in top_types[:10]:
        report_lines.append(f"- {t}: {c}")

    report_lines.append("")
    report_lines.append("## 10. Notes and Limitations")
    report_lines.append(
        "The AST view captures syntactic structure but does not directly encode runtime "
        "execution order or data dependencies. CFG and NDG extraction should be added as "
        "separate graph views and joined later by the same class name or graph index row."
    )
    report_lines.append(
        "The node feature design intentionally stays syntax-focused. Future graph views "
        "should represent program behavior through typed edges, such as control-flow and "
        "dependency relations, instead of adding behavior-specific values to AST nodes."
    )
    report_lines.append("")
    report_lines.append(
        "The fallback graphs are useful for keeping the dataset complete, but they are less "
        "precise than the `javalang` graphs. For experiments, keep `parser_mode` as metadata "
        "so these four samples can be included, excluded, or analyzed separately."
    )
    report_lines.append("")
    report_lines.append("## 11. Sample Visualizations")
    report_lines.append(
        "The following Mermaid diagrams show compact views of the first 45 kept AST nodes "
        "for three example classes."
    )
    for i, gr in enumerate(graph_rows[:3], start=1):
        graph_json = json.loads(Path(gr["graph_json"]).read_text())
        nodes = [
            KeptNode(
                node_id=int(n["id"]),
                node_type=str(n["type"]),
                depth=int(n["depth"]),
                sibling_index=0,
                parent_id=n.get("parent_id"),
                text_hint="",
            )
            for n in graph_json["nodes"]
        ]
        edges = [tuple(e) for e in graph_json["edges"]]
        report_lines.append(f"### {i}. {gr['name']}")
        report_lines.append("```mermaid")
        report_lines.append(to_mermaid(nodes, edges, limit=45))
        report_lines.append("```")

    report_path = out_dir / "ast_report.md"
    report_path.write_text("\n".join(report_lines) + "\n")

    print(f"AST extraction finished. graphs_generated={len(graph_rows)} failures={len(parse_failures)}")
    print(f"index={graph_index_path}")
    print(f"summary={summary_path}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
