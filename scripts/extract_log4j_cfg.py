#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CFG_NODE_TYPES = [
    "ENTRY",
    "EXIT",
    "STATEMENT",
    "CONDITION",
    "RETURN",
    "THROW",
    "LOOP",
    "SWITCH",
    "CATCH",
    "FINALLY",
]
NODE_TYPE_TO_ID = {name: i for i, name in enumerate(CFG_NODE_TYPES)}

CFG_EDGE_TYPES = ["CFG_NEXT", "CFG_TRUE", "CFG_FALSE", "CFG_RETURN", "CFG_EXCEPTION"]
EDGE_TYPE_TO_ID = {name: i for i, name in enumerate(CFG_EDGE_TYPES)}


def run(cmd: list[str], cwd: Path, log_path: Path | None = None) -> None:
    if log_path is None:
        subprocess.run(cmd, cwd=cwd, check=True)
        return

    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
    log_path.write_text((result.stdout or "") + (result.stderr or ""), encoding="utf-8")
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)


def compile_sources(repo_root: Path, mapped: pd.DataFrame, classes_dir: Path, compile_log: Path) -> list[dict[str, str]]:
    classes_dir.mkdir(parents=True, exist_ok=True)
    all_sources = sorted(
        str(p)
        for p in (repo_root / "projects/log4j/logging-log4j1-v_1_0/src/java").rglob("*.java")
    )
    if not all_sources:
        return [{"file_id": "__global__", "source_path": "", "error": "no_java_sources_found"}]
    argfile = repo_root / "build/cfg_soot/javac_sources.txt"
    argfile.write_text("\n".join(all_sources) + "\n", encoding="utf-8")
    ecj_jar = repo_root / "tools/ecj/ecj-4.6.1.jar"
    cmd = [
        "java",
        "-jar",
        str(ecj_jar),
        "-g",
        "-1.3",
        "-proceedOnError",
        "-d",
        str(classes_dir),
        f"@{argfile}",
    ]
    try:
        run(cmd, cwd=repo_root, log_path=compile_log)
        return []
    except subprocess.CalledProcessError as exc:
        # Continue: partial class files may still exist and be analyzable by Soot.
        return [
            {
                "file_id": "__global__",
                "source_path": "",
                "error": f"ecj_partial_failure:{exc.returncode}; see outputs/log4j/cfg/ecj_compile.log",
            }
        ]


def parse_tsv(tsv_path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    by_class: dict[str, Any] = defaultdict(lambda: {"methods": [], "nodes": [], "edges": [], "method_nodes": defaultdict(list)})
    failures: list[dict[str, str]] = []

    for raw in tsv_path.read_text(encoding="utf-8").splitlines():
        parts = raw.split("\t")
        if not parts:
            continue
        rec = parts[0]
        if rec == "FAIL":
            failures.append({"file_id": parts[1], "error": parts[2] if len(parts) > 2 else "unknown"})
            continue
        if rec == "MFAIL":
            failures.append({"file_id": parts[1], "error": f"{parts[2]} :: {parts[3] if len(parts) > 3 else 'method_fail'}"})
            continue

        class_name = parts[1]
        g = by_class[class_name]
        if rec == "METHOD":
            g["methods"].append(
                {
                    "method_id": parts[2],
                    "kind": parts[3],
                    "entry_local": int(parts[4]),
                    "exit_local": int(parts[5]),
                }
            )
        elif rec == "NODE":
            method_id = parts[2]
            node = {
                "local_id": int(parts[3]),
                "method_id": method_id,
                "node_type": parts[4],
                "line_start": int(parts[5]) if parts[5] else None,
                "line_end": int(parts[5]) if parts[5] else None,
                "is_synthetic": parts[6] == "1",
                "snippet": parts[7] if len(parts) > 7 else "",
            }
            g["method_nodes"][method_id].append(node)
        elif rec == "EDGE":
            g["edges"].append(
                {
                    "method_id": parts[2],
                    "source_local": int(parts[3]),
                    "target_local": int(parts[4]),
                    "edge_type": parts[5],
                }
            )

    graphs: dict[str, Any] = {}
    for class_name, g in by_class.items():
        nodes: list[dict[str, Any]] = []
        node_id_map: dict[tuple[str, int], int] = {}
        next_id = 0
        for method in g["methods"]:
            method_id = method["method_id"]
            for n in g["method_nodes"].get(method_id, []):
                node_id_map[(method_id, n["local_id"])] = next_id
                nodes.append(
                    {
                        "id": next_id,
                        "file_id": class_name,
                        "method_id": method_id,
                        "node_type": n["node_type"] if n["node_type"] in NODE_TYPE_TO_ID else "STATEMENT",
                        "node_type_id": NODE_TYPE_TO_ID.get(n["node_type"], NODE_TYPE_TO_ID["STATEMENT"]),
                        "line_start": n["line_start"],
                        "line_end": n["line_end"],
                        "snippet": n["snippet"][:200],
                        "is_synthetic": n["is_synthetic"],
                    }
                )
                next_id += 1

        edges: list[dict[str, Any]] = []
        for e in g["edges"]:
            k1 = (e["method_id"], e["source_local"])
            k2 = (e["method_id"], e["target_local"])
            if k1 not in node_id_map or k2 not in node_id_map:
                continue
            et = e["edge_type"] if e["edge_type"] in EDGE_TYPE_TO_ID else "CFG_NEXT"
            edges.append(
                {
                    "source": node_id_map[k1],
                    "target": node_id_map[k2],
                    "edge_type": et,
                    "edge_type_id": EDGE_TYPE_TO_ID[et],
                }
            )

        methods = []
        for m in g["methods"]:
            entry = node_id_map.get((m["method_id"], m["entry_local"]))
            exit_node = node_id_map.get((m["method_id"], m["exit_local"]))
            if entry is None or exit_node is None:
                continue
            methods.append(
                {
                    "method_id": m["method_id"],
                    "kind": m["kind"],
                    "entry_node": entry,
                    "exit_node": exit_node,
                }
            )

        x = np.zeros((len(nodes), 3), dtype=np.float32)
        node_type_id = np.zeros((len(nodes),), dtype=np.int64)
        for n in nodes:
            i = n["id"]
            x[i, 0] = float(n["line_start"] or 0)
            x[i, 1] = 1.0 if n["snippet"] else 0.0
            x[i, 2] = 1.0 if n["is_synthetic"] else 0.0
            node_type_id[i] = n["node_type_id"]

        if edges:
            edge_index = np.array([[e["source"] for e in edges], [e["target"] for e in edges]], dtype=np.int64)
            edge_type = np.array([e["edge_type_id"] for e in edges], dtype=np.int64)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_type = np.zeros((0,), dtype=np.int64)

        graphs[class_name] = {
            "graph": {
                "file_id": class_name,
                "source_path": "",
                "num_nodes": len(nodes),
                "num_edges": len(edges),
                "node_feature_dim": 3,
                "nodes": nodes,
                "edges": edges,
                "methods": methods,
            },
            "tensors": {"x": x, "node_type_id": node_type_id, "edge_index": edge_index, "edge_type": edge_type},
        }

    return graphs, failures


def make_placeholder_graph(file_id: str) -> dict[str, Any]:
    nodes = [
        {
            "id": 0,
            "file_id": file_id,
            "method_id": "__placeholder__.cfg#0[p0]",
            "node_type": "ENTRY",
            "node_type_id": NODE_TYPE_TO_ID["ENTRY"],
            "line_start": None,
            "line_end": None,
            "snippet": "__ENTRY__",
            "is_synthetic": True,
        },
        {
            "id": 1,
            "file_id": file_id,
            "method_id": "__placeholder__.cfg#0[p0]",
            "node_type": "EXIT",
            "node_type_id": NODE_TYPE_TO_ID["EXIT"],
            "line_start": None,
            "line_end": None,
            "snippet": "__EXIT__",
            "is_synthetic": True,
        },
    ]
    edges = [{"source": 0, "target": 1, "edge_type": "CFG_NEXT", "edge_type_id": EDGE_TYPE_TO_ID["CFG_NEXT"]}]
    x = np.array([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32)
    node_type_id = np.array([NODE_TYPE_TO_ID["ENTRY"], NODE_TYPE_TO_ID["EXIT"]], dtype=np.int64)
    edge_index = np.array([[0], [1]], dtype=np.int64)
    edge_type = np.array([EDGE_TYPE_TO_ID["CFG_NEXT"]], dtype=np.int64)
    return {
        "graph": {
            "file_id": file_id,
            "source_path": "",
            "num_nodes": 2,
            "num_edges": 1,
            "node_feature_dim": 3,
            "nodes": nodes,
            "edges": edges,
            "methods": [{"method_id": "__placeholder__.cfg#0[p0]", "kind": "placeholder", "entry_node": 0, "exit_node": 1}],
        },
        "tensors": {"x": x, "node_type_id": node_type_id, "edge_index": edge_index, "edge_type": edge_type},
    }


def cfg_method_mermaid(graph: dict[str, Any], method_id: str, limit: int = 30) -> str:
    nodes = [n for n in graph["nodes"] if n["method_id"] == method_id][:limit]
    node_ids = {n["id"] for n in nodes}
    edges = [e for e in graph["edges"] if e["source"] in node_ids and e["target"] in node_ids]
    lines = ["graph TD"]
    for node in nodes:
        snippet = node["snippet"].replace('"', "'")
        label = f"{node['node_type']}#{node['id']}"
        if snippet and not node["is_synthetic"]:
            label += f" | {snippet[:55]}"
        lines.append(f'  N{node["id"]}["{label}"]')
    for edge in edges:
        lines.append(f'  N{edge["source"]} -->|{edge["edge_type"]}| N{edge["target"]}')
    return "\n".join(lines)


def build_report(
    summary: dict[str, Any],
    graphs: list[dict[str, Any]],
    parse_failures: list[dict[str, str]],
) -> str:
    node_counts = Counter(n["node_type"] for graph in graphs for n in graph["nodes"])
    edge_counts = Counter(e["edge_type"] for graph in graphs for e in graph["edges"])
    node_type_rows = [f"| `{name}` | {idx} |" for name, idx in NODE_TYPE_TO_ID.items()]
    edge_type_rows = [f"| `{name}` | {idx} |" for name, idx in EDGE_TYPE_TO_ID.items()]
    top_node_rows = [f"- `{name}`: {count}" for name, count in node_counts.most_common()]
    top_edge_rows = [f"- `{name}`: {count}" for name, count in edge_counts.most_common()]
    failure_rows = [
        f"| `{failure['file_id']}` | `{failure['error']}` |"
        for failure in parse_failures
    ] or ["| None | None |"]

    example_graph = next(
        graph
        for graph in graphs
        if graph["file_id"] == "org.apache.log4j.helpers.BoundedFIFO"
    )
    example_method_id = "BoundedFIFO.get#1[p0]"
    mermaid = cfg_method_mermaid(example_graph, example_method_id)

    lines = [
        "# CFG Extraction Report for Log4j 1.0",
        "",
        "## 1. Objective",
        "This report describes the CFG graph extraction step for the Log4j 1.0 PROMISE dataset used in the "
        "multi-view software defect prediction pipeline. The goal is to create method-level control-flow "
        "graphs from compiled Java source and aggregate them into one graph per mapped Java class so they "
        "remain aligned with the file-level PROMISE labels.",
        "",
        "## 2. Input Data",
        "- Dataset: `projects/log4j/log4j-1.0.csv`",
        "- Preprocessed dataset: `outputs/log4j/log4j_preprocessed_standard.csv`",
        "- Class-to-source mapping: `outputs/log4j/log4j_name_to_source_mapping.csv`",
        "- Java source root: `projects/log4j/logging-log4j1-v_1_0/src/java`",
        f"- Mapped source files requested for CFG extraction: {summary['requested_mapped_files']}",
        "",
        "## 3. Extraction Pipeline",
        "- The Java sources are compiled with Eclipse ECJ in Java 1.3 compatibility mode because Log4j 1.0 "
        "contains identifiers that became reserved words in newer Java versions.",
        "- Soot loads the generated class files and retrieves each concrete method or constructor body.",
        "- Soot `ExceptionalUnitGraph` builds the control-flow graph for each concrete method.",
        "- Each method graph receives explicit synthetic `ENTRY` and `EXIT` nodes.",
        "- Soot Jimple units become statement-level CFG nodes.",
        "- Normal, conditional, return, and exceptional flow relations become typed CFG edges.",
        "- All method graphs from the same Java class are aggregated into one file-level graph.",
        "- The AST extractor is independent and remains unchanged.",
        "",
        "```text",
        "Java source -> ECJ compile -> Soot Jimple -> ExceptionalUnitGraph -> method CFGs -> file-level CFG",
        "```",
        "",
        "## 4. CFG Representation",
        "The CFG view stores control behavior mainly in typed edges. Node features remain compact so future "
        "AST, CFG, NDG, control-dependency, and call relations can be integrated without duplicating behavior "
        "inside node feature vectors.",
        "",
        "### 4.1 Node Types",
        "| Node type | ID |",
        "| --- | ---: |",
        *node_type_rows,
        "",
        "`LOOP`, `CATCH`, and `FINALLY` remain reserved vocabulary entries for compatibility. The current "
        "Soot Jimple extractor represents their behavior through statement nodes and typed graph edges.",
        "",
        "### 4.2 Edge Types",
        "| Edge type | ID |",
        "| --- | ---: |",
        *edge_type_rows,
        "",
        "| Edge type | Meaning |",
        "| --- | --- |",
        "| `CFG_NEXT` | Normal control-flow successor. |",
        "| `CFG_TRUE` | Target reached when an `if` condition evaluates true. |",
        "| `CFG_FALSE` | Target reached when an `if` condition evaluates false. |",
        "| `CFG_RETURN` | Return statement flow to the method `EXIT`. |",
        "| `CFG_EXCEPTION` | Exceptional successor or explicit throw flow. |",
        "",
        "### 4.3 Stored Node Metadata",
        "- `file_id`, `method_id`, `id`",
        "- `node_type`, `node_type_id`",
        "- `line_start`, `line_end` when Soot exposes source line metadata",
        "- `snippet`: normalized Jimple statement text",
        "- `is_synthetic`: true for generated `ENTRY`, `EXIT`, and placeholder nodes",
        "",
        "## 5. What the Tensor Files Mean",
        "A tensor is a numeric array used by machine learning frameworks. Each file-level CFG is exported as:",
        "",
        "| File | Shape | Meaning |",
        "| --- | --- | --- |",
        "| `*_x.npy` | `[num_nodes, 3]` | Structural features: source line number, has-snippet flag, and is-synthetic flag. |",
        "| `*_node_type_id.npy` | `[num_nodes]` | Exact CFG node type ids consumed by a trainable embedding layer. |",
        "| `*_edge_index.npy` | `[2, num_edges]` | Connectivity. Row 0 stores source ids; row 1 stores target ids. |",
        "| `*_edge_type.npy` | `[num_edges]` | Exact edge type id for each matching column in `edge_index`. |",
        "",
        "During model training, convert each `node_type_id` to a learned embedding and concatenate it with "
        "the three stored structural values. Use `edge_type` for relation-aware message passing.",
        "",
        "```text",
        "complete_node_x = concat(node_type_embedding(node_type_id), structural_x)",
        "edge_index[:, i] and edge_type[i] describe the same directed CFG edge",
        "```",
        "",
        "## 6. How to Use the CFG Tensors in a GNN",
        "At model level, the processing steps are:",
        "",
        "1. Load `node_type_id`, structural `x`, `edge_index`, and `edge_type` for each CFG graph.",
        "2. Convert every CFG node type id to a learned embedding.",
        "3. Concatenate each embedding with its three fixed structural values.",
        "4. Use `edge_type` with relation-aware GNN layers or edge embeddings.",
        "5. Pool node representations into one graph representation for the Java class.",
        "6. Join the pooled CFG representation with the AST and future NDG view representations.",
        "7. Feed the fused representation into a defect classifier.",
        "",
        "Minimal PyTorch-style preparation:",
        "",
        "```python",
        "import torch",
        "from torch import nn",
        "",
        "node_type_embedding = nn.Embedding(num_embeddings=10, embedding_dim=32)",
        "edge_type_embedding = nn.Embedding(num_embeddings=5, embedding_dim=8)",
        "",
        "complete_node_x = torch.cat([node_type_embedding(node_type_id), structural_x], dim=1)",
        "edge_x = edge_type_embedding(edge_type_id)",
        "```",
        "",
        "## 7. Output Files",
        "- `outputs/log4j/cfg/cfg_index.csv`: index of generated file-level graphs and tensor paths.",
        "- `outputs/log4j/cfg/graphs/*.json`: readable file-level graphs with method, node, and edge metadata.",
        "- `outputs/log4j/cfg/tensors/*_x.npy`: structural node features.",
        "- `outputs/log4j/cfg/tensors/*_node_type_id.npy`: exact CFG node type ids.",
        "- `outputs/log4j/cfg/tensors/*_edge_index.npy`: directed edge connectivity.",
        "- `outputs/log4j/cfg/tensors/*_edge_type.npy`: edge relation type ids.",
        "- `outputs/log4j/cfg/node_type_vocab.json`: stable CFG node type vocabulary.",
        "- `outputs/log4j/cfg/edge_type_vocab.json`: stable CFG edge type vocabulary.",
        "- `outputs/log4j/cfg/cfg_summary.json`: global extraction statistics.",
        "- `outputs/log4j/cfg/parse_failures.json`: compile and Soot extraction issues.",
        "- `outputs/log4j/cfg/validation_issues.json`: CFG validation issue log.",
        "- `outputs/log4j/cfg/ecj_compile.log`: ECJ compiler diagnostics captured during partial compilation.",
        "- `outputs/log4j/cfg/soot_extractor.log`: Soot runner diagnostics captured during CFG extraction.",
        "",
        "## 8. Results",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Mapped samples requested | {summary['requested_mapped_files']} |",
        f"| File-level graphs generated | {summary['graphs_generated']} |",
        f"| Real Soot graphs | {summary['soot_graphs']} |",
        f"| Placeholder graphs | {summary['placeholder_graphs']} |",
        f"| Logged compile/extraction issues | {summary['parse_failures']} |",
        f"| Total method CFGs | {summary['total_methods']} |",
        f"| Total nodes | {summary['total_nodes']} |",
        f"| Total edges | {summary['total_edges']} |",
        f"| Average nodes per graph | {summary['avg_nodes_per_file']:.2f} |",
        f"| Average edges per graph | {summary['avg_edges_per_file']:.2f} |",
        f"| Extracted structural feature dimension | {summary['node_feature_dim']} |",
        f"| CFG node type vocabulary size | {summary['node_type_vocab_size']} |",
        f"| CFG edge type vocabulary size | {summary['edge_type_vocab_size']} |",
        "",
        "## 9. Node and Edge Distribution",
        "### 9.1 Node Types",
        *top_node_rows,
        "",
        "### 9.2 Edge Types",
        *top_edge_rows,
        "",
        "## 10. Notes and Limitations",
        "- The extractor uses Soot CFGs from compiled class files. It does not construct CFGs from `javalang` AST nodes.",
        "- Soot operates on Jimple statements, so snippets are normalized intermediate-representation statements, "
        "not exact source substrings.",
        "- Two mapped interface-like classes currently receive placeholder `ENTRY -> EXIT` graphs because no concrete "
        "method body is available. Keep this distinction visible during experiments.",
        "- Log4j 1.0 is legacy Java source. ECJ emits compile diagnostics under the modern Java runtime while still "
        "producing analyzable class files with `-proceedOnError`. These diagnostics are saved in "
        "`outputs/log4j/cfg/ecj_compile.log` instead of being printed as terminal output.",
        "- One Soot method body retrieval issue is logged for `AppenderSkeleton`; other recoverable methods in that "
        "class remain included.",
        "- `CFG_BREAK` and `CFG_CONTINUE` are intentionally not exported as separate edge types by this backend. "
        "Soot resolves them to normal control-flow successors after compilation.",
        "",
        "### 10.1 Logged Issues",
        "| File | Issue |",
        "| --- | --- |",
        *failure_rows,
        "",
        "## 11. Sample Visualization",
        f"The following diagram shows the Soot CFG for `{example_method_id}`. It is limited to a compact method-level "
        "example so the control-flow relations remain readable.",
        "",
        "```mermaid",
        mermaid,
        "```",
        "",
        "## 12. Future Integration",
        "The file-level graph boundary matches the AST pipeline and the PROMISE labels. Future multi-view graph "
        "integration can unify relations such as:",
        "",
        "```text",
        "AST_CHILD, CFG_NEXT, CFG_TRUE, CFG_FALSE, CFG_RETURN, CFG_EXCEPTION, DATA_DEP, CONTROL_DEP, CALL",
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "outputs/log4j/cfg"
    graphs_dir = out_dir / "graphs"
    tensors_dir = out_dir / "tensors"
    build_dir = repo_root / "build/cfg_soot"
    classes_dir = build_dir / "classes"
    java_out = build_dir / "soot_cfg.tsv"
    ecj_compile_log = out_dir / "ecj_compile.log"
    soot_extractor_log = out_dir / "soot_extractor.log"
    java_src = repo_root / "scripts/soot_cfg_extractor.java"
    java_cls_dir = build_dir / "java"
    java_cls_dir.mkdir(parents=True, exist_ok=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    tensors_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "node_type_vocab.json").write_text(json.dumps(NODE_TYPE_TO_ID, indent=2))
    (out_dir / "edge_type_vocab.json").write_text(json.dumps(EDGE_TYPE_TO_ID, indent=2))

    mapping_df = pd.read_csv(repo_root / "outputs/log4j/log4j_name_to_source_mapping.csv")
    mapped = mapping_df[mapping_df["source_path"].notna()].copy()

    javac_failures = compile_sources(repo_root, mapped, classes_dir, ecj_compile_log)

    class_list = build_dir / "class_list.txt"
    class_list.write_text("\n".join(sorted(mapped["name"].astype(str).tolist())) + "\n", encoding="utf-8")

    soot_jar = repo_root / "tools/soot/soot-4.5.0-jar-with-dependencies.jar"
    run(["javac", "-cp", str(soot_jar), "-d", str(java_cls_dir), str(java_src)], cwd=repo_root)
    run(
        [
            "java",
            "-cp",
            f"{java_cls_dir}:{soot_jar}",
            "soot_cfg_extractor",
            str(classes_dir),
            str(class_list),
            str(repo_root / "projects/log4j/logging-log4j1-v_1_0/src/java"),
            str(java_out),
        ],
        cwd=repo_root,
        log_path=soot_extractor_log,
    )

    parsed, soot_failures = parse_tsv(java_out)
    mapped_ids = set(mapped["name"].astype(str).tolist())
    placeholder_ids = sorted(mapped_ids - set(parsed.keys()))
    for missing_id in placeholder_ids:
        parsed[missing_id] = make_placeholder_graph(missing_id)

    parse_failures = list(javac_failures)
    for f in soot_failures:
        parse_failures.append({"file_id": f["file_id"], "source_path": "", "error": f["error"]})

    index_rows = []
    written_graphs = []
    total_nodes = 0
    total_edges = 0

    for file_id, result in sorted(parsed.items()):
        graph = result["graph"]
        tensors = result["tensors"]
        source_path = str(mapped.loc[mapped["name"] == file_id, "source_path"].iloc[0])
        graph["source_path"] = source_path
        extraction_mode = "placeholder" if file_id in placeholder_ids else "soot"
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in file_id)
        graph_path = graphs_dir / f"{safe_name}.json"
        x_path = tensors_dir / f"{safe_name}_x.npy"
        node_type_path = tensors_dir / f"{safe_name}_node_type_id.npy"
        edge_index_path = tensors_dir / f"{safe_name}_edge_index.npy"
        edge_type_path = tensors_dir / f"{safe_name}_edge_type.npy"

        graph_path.write_text(json.dumps(graph, indent=2))
        np.save(x_path, tensors["x"])
        np.save(node_type_path, tensors["node_type_id"])
        np.save(edge_index_path, tensors["edge_index"])
        np.save(edge_type_path, tensors["edge_type"])

        total_nodes += graph["num_nodes"]
        total_edges += graph["num_edges"]
        written_graphs.append(graph)
        index_rows.append(
            {
                "file_id": file_id,
                "source_path": source_path,
                "extraction_mode": extraction_mode,
                "num_nodes": graph["num_nodes"],
                "num_edges": graph["num_edges"],
                "num_methods": len(graph["methods"]),
                "node_feature_dim": 3,
                "graph_json": str(graph_path),
                "x_npy": str(x_path),
                "node_type_id_npy": str(node_type_path),
                "edge_index_npy": str(edge_index_path),
                "edge_type_npy": str(edge_type_path),
            }
        )

    cfg_index = pd.DataFrame(index_rows)
    if not cfg_index.empty:
        cfg_index = cfg_index.sort_values(by="file_id")
    cfg_index.to_csv(out_dir / "cfg_index.csv", index=False)

    (out_dir / "parse_failures.json").write_text(json.dumps(parse_failures, indent=2))
    (out_dir / "validation_issues.json").write_text("[]\n")

    summary = {
        "requested_mapped_files": int(len(mapped)),
        "graphs_generated": int(len(index_rows)),
        "soot_graphs": int(len(index_rows) - len(placeholder_ids)),
        "placeholder_graphs": int(len(placeholder_ids)),
        "parse_failures": int(len(parse_failures)),
        "validation_issues": 0,
        "total_nodes": int(total_nodes),
        "total_edges": int(total_edges),
        "total_methods": int(sum(row["num_methods"] for row in index_rows)),
        "avg_nodes_per_file": float(total_nodes / len(index_rows)) if index_rows else 0.0,
        "avg_edges_per_file": float(total_edges / len(index_rows)) if index_rows else 0.0,
        "node_feature_dim": 3,
        "node_type_vocab_size": len(NODE_TYPE_TO_ID),
        "edge_type_vocab_size": len(EDGE_TYPE_TO_ID),
        "backend": "soot",
        "ecj_compile_log": "outputs/log4j/cfg/ecj_compile.log",
        "soot_extractor_log": "outputs/log4j/cfg/soot_extractor.log",
    }
    (out_dir / "cfg_summary.json").write_text(json.dumps(summary, indent=2))

    (out_dir / "cfg_report.md").write_text(build_report(summary, written_graphs, parse_failures))

    print(f"CFG extraction finished (backend=soot). graphs_generated={len(index_rows)} parse_failures={len(parse_failures)}")


if __name__ == "__main__":
    main()
