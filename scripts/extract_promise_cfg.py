#!/usr/bin/env python3
"""Extract one Soot-based file-level CFG per mapped PROMISE Java file."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from collections import defaultdict
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
NODE_TYPE_TO_ID = {name: idx for idx, name in enumerate(CFG_NODE_TYPES)}

CFG_EDGE_TYPES = [
    "CFG_NEXT",
    "CFG_TRUE",
    "CFG_FALSE",
    "CFG_RETURN",
    "CFG_EXCEPTION",
    "CFG_BACK",
    "CFG_SWITCH_CASE",
    "CFG_SWITCH_DEFAULT",
]
EDGE_TYPE_TO_ID = {name: idx for idx, name in enumerate(CFG_EDGE_TYPES)}

CFG_STMT_KINDS = [
    "NO_STMT",
    "ASSIGN",
    "INVOKE",
    "IDENTITY",
    "IF",
    "GOTO",
    "SWITCH",
    "RETURN_VALUE",
    "RETURN_VOID",
    "THROW",
    "MONITOR",
    "NOP",
    "OTHER",
]
STMT_KIND_TO_ID = {name: idx for idx, name in enumerate(CFG_STMT_KINDS)}

CFG_INVOKE_KINDS = [
    "NO_INVOKE",
    "STATIC_INVOKE",
    "VIRTUAL_INVOKE",
    "INTERFACE_INVOKE",
    "SPECIAL_INVOKE",
    "DYNAMIC_INVOKE",
    "UNKNOWN_INVOKE",
]
INVOKE_KIND_TO_ID = {name: idx for idx, name in enumerate(CFG_INVOKE_KINDS)}

BASE_FEATURE_NAMES = ["line_position", "has_source_line", "is_synthetic"]
INSTRUCTION_FLAG_NAMES = [
    "has_method_call",
    "has_field_read",
    "has_field_write",
    "has_array_read",
    "has_array_write",
    "has_new_object",
    "has_new_array",
    "has_cast",
    "has_arithmetic_op",
    "has_comparison_op",
    "has_null_constant",
    "has_string_constant",
    "has_numeric_constant",
]
CFG_ROLE_FEATURE_NAMES = [
    "in_degree_log",
    "out_degree_log",
    "is_branch_node",
    "is_join_node",
    "is_terminal_node",
    "node_position_in_method",
    "method_size_normalized",
    "is_loop_header",
    "is_in_loop",
]
FEATURE_NAMES = [*BASE_FEATURE_NAMES, *INSTRUCTION_FLAG_NAMES, *CFG_ROLE_FEATURE_NAMES]

STRUCTURAL_FEATURE_DIM = len(FEATURE_NAMES)
NODE_TYPE_EMBEDDING_DIM = 32
STMT_KIND_EMBEDDING_DIM = 16
INVOKE_KIND_EMBEDDING_DIM = 8
MODEL_NODE_FEATURE_DIM = STRUCTURAL_FEATURE_DIM + NODE_TYPE_EMBEDDING_DIM + STMT_KIND_EMBEDDING_DIM + INVOKE_KIND_EMBEDDING_DIM


def run(cmd: list[str], cwd: Path, log_path: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    if log_path is None:
        return subprocess.run(cmd, cwd=cwd, text=True, check=check)
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text((result.stdout or "") + (result.stderr or ""), encoding="utf-8")
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    return result


def sanitize_filename(value: str, max_len: int = 180) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    if len(safe) <= max_len:
        return safe
    import hashlib

    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"{safe[: max_len - 13]}_{digest}"


def graph_id(dataset_name: str, class_name: str) -> str:
    return f"{dataset_name}::{class_name}"


def choose_ecj_level(source_root: Path) -> str:
    """Choose a practical ECJ source level for legacy and newer PROMISE projects."""
    checked = 0
    has_java5_syntax = False
    for java_path in source_root.rglob("*.java"):
        if checked >= 300:
            break
        checked += 1
        text = java_path.read_text(encoding="utf-8", errors="ignore")
        if "@" in text and re.search(r"@\s*[A-Za-z_][A-Za-z0-9_.]*", text):
            has_java5_syntax = True
            break
        if re.search(r"\b(?:List|Map|Set|Iterator|Collection|Comparable|Class)\s*<", text):
            has_java5_syntax = True
            break
    return "-1.5" if has_java5_syntax else "-1.3"


def project_classpath(source_root: Path, classes_dir: Path) -> str:
    jars = [str(path) for path in sorted(source_root.rglob("*.jar"))]
    entries = [str(classes_dir), str(source_root), *jars]
    return os.pathsep.join(entries)


def empty_instruction_flags() -> dict[str, int]:
    return {name: 0 for name in INSTRUCTION_FLAG_NAMES}


def parse_flag(value: str) -> int:
    return 1 if str(value).strip() == "1" else 0


def compile_sources(
    repo_root: Path,
    dataset_name: str,
    source_root: Path,
    classes_dir: Path,
    argfile: Path,
    compile_log: Path,
    source_level: str,
) -> list[dict[str, str]]:
    classes_dir.mkdir(parents=True, exist_ok=True)
    all_sources = sorted(str(path) for path in source_root.rglob("*.java"))
    if not all_sources:
        return [{"dataset_name": dataset_name, "file_id": "__global__", "source_path": "", "error": "no_java_sources_found"}]

    argfile.parent.mkdir(parents=True, exist_ok=True)
    argfile.write_text("\n".join(all_sources) + "\n", encoding="utf-8")
    ecj_jar = repo_root / "tools/ecj/ecj-4.6.1.jar"
    cmd = [
        "java",
        "-jar",
        str(ecj_jar),
        "-g",
        source_level,
        "-proceedOnError",
        "-nowarn",
        "-d",
        str(classes_dir),
        "-sourcepath",
        str(source_root),
        "-classpath",
        project_classpath(source_root, classes_dir),
        f"@{argfile}",
    ]
    result = run(cmd, cwd=repo_root, log_path=compile_log, check=False)
    if result.returncode == 0:
        return []
    return [
        {
            "dataset_name": dataset_name,
            "file_id": "__global__",
            "source_path": str(source_root),
            "error": f"ecj_partial_failure:{result.returncode}; source_level={source_level}; see {compile_log}",
        }
    ]


def parse_tsv(tsv_path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    by_class: dict[str, Any] = defaultdict(lambda: {"methods": [], "nodes": [], "edges": [], "method_nodes": defaultdict(list)})
    failures: list[dict[str, str]] = []

    if not tsv_path.exists():
        return {}, [{"file_id": "__global__", "error": f"missing_soot_tsv:{tsv_path}"}]

    for raw in tsv_path.read_text(encoding="utf-8", errors="ignore").splitlines():
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
        graph = by_class[class_name]
        if rec == "METHOD":
            graph["methods"].append(
                {
                    "method_id": parts[2],
                    "kind": parts[3],
                    "entry_local": int(parts[4]),
                    "exit_local": int(parts[5]),
                }
            )
        elif rec == "NODE":
            method_id = parts[2]
            stmt_kind = parts[7] if len(parts) > 7 and parts[7] in STMT_KIND_TO_ID else "OTHER"
            invoke_kind = parts[8] if len(parts) > 8 and parts[8] in INVOKE_KIND_TO_ID else "NO_INVOKE"
            instruction_flags = empty_instruction_flags()
            if len(parts) >= 9 + len(INSTRUCTION_FLAG_NAMES):
                for flag_name, flag_value in zip(INSTRUCTION_FLAG_NAMES, parts[9 : 9 + len(INSTRUCTION_FLAG_NAMES)], strict=True):
                    instruction_flags[flag_name] = parse_flag(flag_value)
                snippet_idx = 9 + len(INSTRUCTION_FLAG_NAMES)
            else:
                snippet_idx = 7
            node = {
                "local_id": int(parts[3]),
                "method_id": method_id,
                "node_type": parts[4],
                "line_start": int(parts[5]) if parts[5] else None,
                "line_end": int(parts[5]) if parts[5] else None,
                "is_synthetic": parts[6] == "1",
                "stmt_kind": stmt_kind,
                "stmt_kind_id": STMT_KIND_TO_ID[stmt_kind],
                "invoke_kind": invoke_kind,
                "invoke_kind_id": INVOKE_KIND_TO_ID[invoke_kind],
                "instruction_flags": instruction_flags,
                "snippet": parts[snippet_idx] if len(parts) > snippet_idx else "",
            }
            graph["method_nodes"][method_id].append(node)
        elif rec == "EDGE":
            graph["edges"].append(
                {
                    "method_id": parts[2],
                    "source_local": int(parts[3]),
                    "target_local": int(parts[4]),
                    "edge_type": parts[5],
                }
            )

    parsed: dict[str, Any] = {}
    for class_name, graph in by_class.items():
        nodes: list[dict[str, Any]] = []
        node_id_map: dict[tuple[str, int], int] = {}
        next_id = 0
        for method in graph["methods"]:
            method_id = method["method_id"]
            for node in graph["method_nodes"].get(method_id, []):
                node_id_map[(method_id, node["local_id"])] = next_id
                node_type = node["node_type"] if node["node_type"] in NODE_TYPE_TO_ID else "STATEMENT"
                nodes.append(
                    {
                        "id": next_id,
                        "local_id": node["local_id"],
                        "file_id": class_name,
                        "method_id": method_id,
                        "node_type": node_type,
                        "node_type_id": NODE_TYPE_TO_ID[node_type],
                        "stmt_kind": node["stmt_kind"],
                        "stmt_kind_id": node["stmt_kind_id"],
                        "invoke_kind": node["invoke_kind"],
                        "invoke_kind_id": node["invoke_kind_id"],
                        "instruction_flags": node["instruction_flags"],
                        "line_start": node["line_start"],
                        "line_end": node["line_end"],
                        "snippet": node["snippet"][:200],
                        "is_synthetic": node["is_synthetic"],
                    }
                )
                next_id += 1

        edges: list[dict[str, Any]] = []
        for edge in graph["edges"]:
            src_key = (edge["method_id"], edge["source_local"])
            dst_key = (edge["method_id"], edge["target_local"])
            if src_key not in node_id_map or dst_key not in node_id_map:
                continue
            edge_type = edge["edge_type"] if edge["edge_type"] in EDGE_TYPE_TO_ID else "CFG_NEXT"
            edges.append(
                {
                    "source": node_id_map[src_key],
                    "target": node_id_map[dst_key],
                    "edge_type": edge_type,
                    "edge_type_id": EDGE_TYPE_TO_ID[edge_type],
                }
            )

        annotate_cfg_roles(nodes, edges)
        methods: list[dict[str, Any]] = []
        for method in graph["methods"]:
            entry_node = node_id_map.get((method["method_id"], method["entry_local"]))
            exit_node = node_id_map.get((method["method_id"], method["exit_local"]))
            if entry_node is None or exit_node is None:
                continue
            methods.append(
                {
                    "method_id": method["method_id"],
                    "kind": method["kind"],
                    "entry_node": entry_node,
                    "exit_node": exit_node,
                }
            )

        tensors = build_tensors(nodes, edges)
        parsed[class_name] = {
            "graph": {
                "file_id": class_name,
                "source_path": "",
                "num_nodes": len(nodes),
                "num_edges": len(edges),
                "node_feature_dim": STRUCTURAL_FEATURE_DIM,
                "nodes": nodes,
                "edges": edges,
                "methods": methods,
            },
            "tensors": tensors,
        }

    return parsed, failures


def annotate_cfg_roles(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> None:
    n_nodes = len(nodes)
    in_degree = [0] * n_nodes
    out_degree = [0] * n_nodes
    for edge in edges:
        source = int(edge["source"])
        target = int(edge["target"])
        if 0 <= source < n_nodes:
            out_degree[source] += 1
        if 0 <= target < n_nodes:
            in_degree[target] += 1

    nodes_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        nodes_by_method[str(node["method_id"])].append(node)
    for method_nodes in nodes_by_method.values():
        method_nodes.sort(key=lambda item: int(item["id"]))
    max_method_size = max((len(method_nodes) for method_nodes in nodes_by_method.values()), default=1)

    is_loop_header = [0] * n_nodes
    is_in_loop = [0] * n_nodes
    for edge in edges:
        if edge["edge_type"] != "CFG_BACK":
            continue
        source = int(edge["source"])
        target = int(edge["target"])
        if not (0 <= source < n_nodes and 0 <= target < n_nodes):
            continue
        if nodes[source]["method_id"] != nodes[target]["method_id"]:
            continue
        is_loop_header[target] = 1
        start, end = sorted((source, target))
        for node in nodes_by_method[str(nodes[source]["method_id"])]:
            node_id = int(node["id"])
            if start <= node_id <= end:
                is_in_loop[node_id] = 1

    for method_nodes in nodes_by_method.values():
        method_size = len(method_nodes)
        method_denominator = max(method_size - 1, 1)
        method_size_normalized = float(method_size / max_method_size) if max_method_size else 0.0
        for position, node in enumerate(method_nodes):
            node_id = int(node["id"])
            node["cfg_role_features"] = {
                "in_degree": int(in_degree[node_id]),
                "out_degree": int(out_degree[node_id]),
                "in_degree_log": float(np.log1p(in_degree[node_id])),
                "out_degree_log": float(np.log1p(out_degree[node_id])),
                "is_branch_node": 1 if out_degree[node_id] > 1 else 0,
                "is_join_node": 1 if in_degree[node_id] > 1 else 0,
                "is_terminal_node": 1 if node["node_type"] in {"EXIT", "RETURN", "THROW"} or out_degree[node_id] == 0 else 0,
                "node_position_in_method": float(position / method_denominator),
                "method_size_normalized": method_size_normalized,
                "is_loop_header": int(is_loop_header[node_id]),
                "is_in_loop": int(is_in_loop[node_id]),
            }


def build_tensors(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    x = np.zeros((len(nodes), STRUCTURAL_FEATURE_DIM), dtype=np.float32)
    node_type_id = np.zeros((len(nodes),), dtype=np.int64)
    stmt_kind_id = np.zeros((len(nodes),), dtype=np.int64)
    invoke_kind_id = np.zeros((len(nodes),), dtype=np.int64)
    source_lines = [int(node["line_start"]) for node in nodes if node.get("line_start") is not None]
    min_line = min(source_lines) if source_lines else 0
    max_line = max(source_lines) if source_lines else min_line
    line_span = max(max_line - min_line, 1)

    for node in nodes:
        i = node["id"]
        if node.get("line_start") is not None:
            x[i, 0] = float((int(node["line_start"]) - min_line) / line_span)
            x[i, 1] = 1.0
        x[i, 2] = 1.0 if node["is_synthetic"] else 0.0
        offset = len(BASE_FEATURE_NAMES)
        for j, flag_name in enumerate(INSTRUCTION_FLAG_NAMES):
            x[i, offset + j] = float(node.get("instruction_flags", {}).get(flag_name, 0))
        offset += len(INSTRUCTION_FLAG_NAMES)
        role_features = node.get("cfg_role_features", {})
        for j, feature_name in enumerate(CFG_ROLE_FEATURE_NAMES):
            x[i, offset + j] = float(role_features.get(feature_name, 0.0))
        node_type_id[i] = node["node_type_id"]
        stmt_kind_id[i] = node.get("stmt_kind_id", STMT_KIND_TO_ID["OTHER"])
        invoke_kind_id[i] = node.get("invoke_kind_id", INVOKE_KIND_TO_ID["NO_INVOKE"])

    if edges:
        edge_index = np.array([[edge["source"] for edge in edges], [edge["target"] for edge in edges]], dtype=np.int64)
        edge_type = np.array([edge["edge_type_id"] for edge in edges], dtype=np.int64)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_type = np.zeros((0,), dtype=np.int64)

    return {
        "x": x,
        "node_type_id": node_type_id,
        "stmt_kind_id": stmt_kind_id,
        "invoke_kind_id": invoke_kind_id,
        "edge_index": edge_index,
        "edge_type": edge_type,
    }


def make_placeholder_graph(dataset_name: str, class_name: str, source_path: str, label: int | None) -> dict[str, Any]:
    gid = graph_id(dataset_name, class_name)
    method_id = "__placeholder__.cfg#0[p0]"
    nodes = [
        {
            "id": 0,
            "file_id": class_name,
            "method_id": method_id,
            "node_type": "ENTRY",
            "node_type_id": NODE_TYPE_TO_ID["ENTRY"],
            "stmt_kind": "NO_STMT",
            "stmt_kind_id": STMT_KIND_TO_ID["NO_STMT"],
            "invoke_kind": "NO_INVOKE",
            "invoke_kind_id": INVOKE_KIND_TO_ID["NO_INVOKE"],
            "instruction_flags": empty_instruction_flags(),
            "line_start": None,
            "line_end": None,
            "snippet": "__ENTRY__",
            "is_synthetic": True,
        },
        {
            "id": 1,
            "file_id": class_name,
            "method_id": method_id,
            "node_type": "EXIT",
            "node_type_id": NODE_TYPE_TO_ID["EXIT"],
            "stmt_kind": "NO_STMT",
            "stmt_kind_id": STMT_KIND_TO_ID["NO_STMT"],
            "invoke_kind": "NO_INVOKE",
            "invoke_kind_id": INVOKE_KIND_TO_ID["NO_INVOKE"],
            "instruction_flags": empty_instruction_flags(),
            "line_start": None,
            "line_end": None,
            "snippet": "__EXIT__",
            "is_synthetic": True,
        },
    ]
    edges = [{"source": 0, "target": 1, "edge_type": "CFG_NEXT", "edge_type_id": EDGE_TYPE_TO_ID["CFG_NEXT"]}]
    annotate_cfg_roles(nodes, edges)
    graph = {
        "graph_id": gid,
        "dataset_name": dataset_name,
        "file_id": class_name,
        "source_path": source_path,
        "label": label,
        "num_nodes": 2,
        "num_edges": 1,
        "node_feature_dim": STRUCTURAL_FEATURE_DIM,
        "nodes": nodes,
        "edges": edges,
        "methods": [{"method_id": method_id, "kind": "placeholder", "entry_node": 0, "exit_node": 1}],
    }
    return {"graph": graph, "tensors": build_tensors(nodes, edges)}


def prepare_dirs(output_dir: Path, build_dir: Path, clean: bool) -> tuple[Path, Path, Path]:
    graphs_dir = output_dir / "graphs"
    tensors_dir = output_dir / "tensors"
    logs_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    if clean:
        shutil.rmtree(graphs_dir, ignore_errors=True)
        shutil.rmtree(tensors_dir, ignore_errors=True)
        shutil.rmtree(logs_dir, ignore_errors=True)
        shutil.rmtree(build_dir, ignore_errors=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    tensors_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    return graphs_dir, tensors_dir, logs_dir


def load_source_roots(summary_path: Path) -> dict[str, Path]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    roots: dict[str, Path] = {}
    for project in summary.get("project_summaries", []):
        roots[str(project["dataset_name"])] = Path(project["source_root"])
    return roots


def load_mapped_rows(input_csv: Path, dataset_filter: str | None) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    required = {"dataset_name", "name", "source_path", "match_strategy", "bug"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required CFG input columns: {missing}")
    if dataset_filter:
        df = df[df["dataset_name"].astype(str) == dataset_filter].copy()
    mapped = df[df["source_path"].notna()].copy()
    mapped = mapped[mapped["match_strategy"].astype(str) != "not_found"].copy()
    mapped = mapped[mapped["match_strategy"].astype(str) != "ambiguous_simple_name"].copy()
    return mapped.reset_index(drop=True)


def compile_soot_helper(repo_root: Path, build_dir: Path) -> Path:
    java_src = repo_root / "scripts/soot_cfg_extractor.java"
    java_classes_dir = build_dir / "java"
    java_classes_dir.mkdir(parents=True, exist_ok=True)
    soot_jar = repo_root / "tools/soot/soot-4.5.0-jar-with-dependencies.jar"
    run(["javac", "-cp", str(soot_jar), "-d", str(java_classes_dir), str(java_src)], cwd=repo_root)
    return java_classes_dir


def extract_dataset_cfgs(
    repo_root: Path,
    dataset_name: str,
    mapped: pd.DataFrame,
    source_root: Path,
    build_dir: Path,
    logs_dir: Path,
    java_classes_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, str]], dict[str, Any]]:
    dataset_build_dir = build_dir / sanitize_filename(dataset_name)
    classes_dir = dataset_build_dir / "classes"
    argfile = dataset_build_dir / "javac_sources.txt"
    class_list = dataset_build_dir / "class_list.txt"
    soot_tsv = dataset_build_dir / "soot_cfg.tsv"
    source_level = choose_ecj_level(source_root)
    ecj_log = logs_dir / f"{sanitize_filename(dataset_name)}_ecj_compile.log"
    soot_log = logs_dir / f"{sanitize_filename(dataset_name)}_soot_extractor.log"

    compile_issues = compile_sources(repo_root, dataset_name, source_root, classes_dir, argfile, ecj_log, source_level)
    class_names = sorted(mapped["name"].astype(str).tolist())
    class_list.write_text("\n".join(class_names) + "\n", encoding="utf-8")

    soot_jar = repo_root / "tools/soot/soot-4.5.0-jar-with-dependencies.jar"
    result = run(
        [
            "java",
            "-cp",
            os.pathsep.join([str(java_classes_dir), str(soot_jar)]),
            "soot_cfg_extractor",
            str(classes_dir),
            str(class_list),
            str(source_root),
            str(soot_tsv),
        ],
        cwd=repo_root,
        log_path=soot_log,
        check=False,
    )
    soot_run_issues: list[dict[str, str]] = []
    if result.returncode != 0:
        soot_run_issues.append(
            {
                "dataset_name": dataset_name,
                "file_id": "__global__",
                "source_path": str(source_root),
                "error": f"soot_runner_failure:{result.returncode}; see {soot_log}",
            }
        )

    parsed, soot_failures = parse_tsv(soot_tsv)
    all_issues = [*compile_issues, *soot_run_issues]
    for failure in soot_failures:
        all_issues.append(
            {
                "dataset_name": dataset_name,
                "file_id": failure["file_id"],
                "source_path": "",
                "error": failure["error"],
            }
        )

    dataset_meta = {
        "dataset": dataset_name,
        "source_root": str(source_root),
        "requested": int(len(mapped)),
        "source_level": source_level,
        "ecj_compile_log": str(ecj_log),
        "soot_extractor_log": str(soot_log),
        "soot_returncode": int(result.returncode),
    }
    return parsed, all_issues, dataset_meta


def write_graph_outputs(
    result: dict[str, Any],
    dataset_name: str,
    class_name: str,
    source_path: str,
    label: int | None,
    extraction_mode: str,
    graphs_dir: Path,
    tensors_dir: Path,
) -> dict[str, Any]:
    gid = graph_id(dataset_name, class_name)
    graph = result["graph"]
    tensors = result["tensors"]
    graph.update(
        {
            "graph_id": gid,
            "dataset_name": dataset_name,
            "file_id": class_name,
            "source_path": source_path,
            "label": label,
        }
    )
    for node in graph["nodes"]:
        node["file_id"] = class_name

    safe_name = sanitize_filename(gid)
    graph_path = graphs_dir / f"{safe_name}.json"
    x_path = tensors_dir / f"{safe_name}_x.npy"
    node_type_path = tensors_dir / f"{safe_name}_node_type_id.npy"
    edge_index_path = tensors_dir / f"{safe_name}_edge_index.npy"
    edge_type_path = tensors_dir / f"{safe_name}_edge_type.npy"
    stmt_kind_path = tensors_dir / f"{safe_name}_stmt_kind_id.npy"
    invoke_kind_path = tensors_dir / f"{safe_name}_invoke_kind_id.npy"

    graph_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    np.save(x_path, tensors["x"])
    np.save(node_type_path, tensors["node_type_id"])
    np.save(stmt_kind_path, tensors["stmt_kind_id"])
    np.save(invoke_kind_path, tensors["invoke_kind_id"])
    np.save(edge_index_path, tensors["edge_index"])
    np.save(edge_type_path, tensors["edge_type"])

    return {
        "graph_id": gid,
        "dataset_name": dataset_name,
        "name": class_name,
        "source_path": source_path,
        "label": label,
        "extraction_mode": extraction_mode,
        "num_nodes": int(graph["num_nodes"]),
        "num_edges": int(graph["num_edges"]),
        "num_methods": int(len(graph["methods"])),
        "node_feature_dim": int(tensors["x"].shape[1]),
        "graph_json": str(graph_path),
        "x_npy": str(x_path),
        "node_type_id_npy": str(node_type_path),
        "stmt_kind_id_npy": str(stmt_kind_path),
        "invoke_kind_id_npy": str(invoke_kind_path),
        "edge_index_npy": str(edge_index_path),
        "edge_type_npy": str(edge_type_path),
    }


def validate_outputs(index_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for row in index_rows:
        x = np.load(row["x_npy"])
        node_type_id = np.load(row["node_type_id_npy"])
        stmt_kind_id = np.load(row["stmt_kind_id_npy"])
        invoke_kind_id = np.load(row["invoke_kind_id_npy"])
        edge_index = np.load(row["edge_index_npy"])
        edge_type = np.load(row["edge_type_npy"])
        if x.shape != (row["num_nodes"], STRUCTURAL_FEATURE_DIM):
            issues.append({"graph_id": row["graph_id"], "issue": f"x_shape:{x.shape}"})
        if node_type_id.shape != (row["num_nodes"],):
            issues.append({"graph_id": row["graph_id"], "issue": f"node_type_shape:{node_type_id.shape}"})
        if stmt_kind_id.shape != (row["num_nodes"],):
            issues.append({"graph_id": row["graph_id"], "issue": f"stmt_kind_shape:{stmt_kind_id.shape}"})
        if invoke_kind_id.shape != (row["num_nodes"],):
            issues.append({"graph_id": row["graph_id"], "issue": f"invoke_kind_shape:{invoke_kind_id.shape}"})
        if edge_index.shape != (2, row["num_edges"]):
            issues.append({"graph_id": row["graph_id"], "issue": f"edge_index_shape:{edge_index.shape}"})
        if edge_type.shape != (row["num_edges"],):
            issues.append({"graph_id": row["graph_id"], "issue": f"edge_type_shape:{edge_type.shape}"})
        if not np.isfinite(x).all():
            issues.append({"graph_id": row["graph_id"], "issue": "non_finite_x"})
    return issues



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Soot-based CFG graphs for PROMISE Java SDP datasets.")
    parser.add_argument("--input-csv", type=Path, default=Path("outputs/promise/promise_preprocessed_log1p.csv"))
    parser.add_argument("--preprocess-summary", type=Path, default=Path("outputs/promise/promise_preprocess_summary.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/promise/cfg"))
    parser.add_argument("--build-dir", type=Path, default=Path("build/promise_cfg_soot"))
    parser.add_argument("--dataset-name", help="Optional dataset filter, e.g. ant-1.6")
    parser.add_argument("--no-clean", action="store_true", help="Do not clear existing graph/tensor/log/build files first.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    input_csv = (repo_root / args.input_csv).resolve() if not args.input_csv.is_absolute() else args.input_csv.resolve()
    summary_path = (repo_root / args.preprocess_summary).resolve() if not args.preprocess_summary.is_absolute() else args.preprocess_summary.resolve()
    output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    build_dir = (repo_root / args.build_dir).resolve() if not args.build_dir.is_absolute() else args.build_dir.resolve()

    if not input_csv.exists():
        raise FileNotFoundError(f"Missing CFG input CSV: {input_csv}. Run scripts/preprocess_promise.py first.")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing preprocessing summary: {summary_path}. Run scripts/preprocess_promise.py first.")

    graphs_dir, tensors_dir, logs_dir = prepare_dirs(output_dir, build_dir, clean=not args.no_clean)
    (output_dir / "node_type_vocab.json").write_text(json.dumps(NODE_TYPE_TO_ID, indent=2), encoding="utf-8")
    (output_dir / "stmt_kind_vocab.json").write_text(json.dumps(STMT_KIND_TO_ID, indent=2), encoding="utf-8")
    (output_dir / "invoke_kind_vocab.json").write_text(json.dumps(INVOKE_KIND_TO_ID, indent=2), encoding="utf-8")
    (output_dir / "edge_type_vocab.json").write_text(json.dumps(EDGE_TYPE_TO_ID, indent=2), encoding="utf-8")
    (output_dir / "feature_names.json").write_text(json.dumps(FEATURE_NAMES, indent=2), encoding="utf-8")

    input_df = pd.read_csv(input_csv)
    mapped = load_mapped_rows(input_csv, args.dataset_name)
    source_roots = load_source_roots(summary_path)
    java_classes_dir = compile_soot_helper(repo_root, build_dir)

    index_rows: list[dict[str, Any]] = []
    parse_failures: list[dict[str, str]] = []
    dataset_summaries: list[dict[str, Any]] = []

    for dataset_name, dataset_rows in mapped.groupby("dataset_name", sort=True):
        dataset_name = str(dataset_name)
        source_root = source_roots.get(dataset_name)
        if source_root is None:
            raise ValueError(f"Missing source root for dataset {dataset_name}")

        parsed, issues, dataset_meta = extract_dataset_cfgs(
            repo_root=repo_root,
            dataset_name=dataset_name,
            mapped=dataset_rows,
            source_root=source_root,
            build_dir=build_dir,
            logs_dir=logs_dir,
            java_classes_dir=java_classes_dir,
        )
        parse_failures.extend(issues)

        parsed_ids = set(parsed)
        soot_count = 0
        placeholder_count = 0
        for _, row in dataset_rows.sort_values("name").iterrows():
            class_name = str(row["name"])
            source_path = str(row["source_path"])
            label = int(row["bug"]) if pd.notna(row["bug"]) else None
            if class_name in parsed:
                result = parsed[class_name]
                extraction_mode = "soot"
                soot_count += 1
            else:
                result = make_placeholder_graph(dataset_name, class_name, source_path, label)
                extraction_mode = "placeholder"
                placeholder_count += 1
            index_rows.append(
                write_graph_outputs(
                    result=result,
                    dataset_name=dataset_name,
                    class_name=class_name,
                    source_path=source_path,
                    label=label,
                    extraction_mode=extraction_mode,
                    graphs_dir=graphs_dir,
                    tensors_dir=tensors_dir,
                )
            )

        unresolved_classes = sorted(parsed_ids - set(dataset_rows["name"].astype(str)))
        if unresolved_classes:
            parse_failures.append(
                {
                    "dataset_name": dataset_name,
                    "file_id": "__global__",
                    "source_path": str(source_root),
                    "error": f"soot_returned_unrequested_classes:{len(unresolved_classes)}",
                }
            )

        dataset_summaries.append(
            {
                **dataset_meta,
                "graphs": int(len(dataset_rows)),
                "soot_graphs": int(soot_count),
                "placeholder_graphs": int(placeholder_count),
                "issues": int(sum(1 for issue in issues if issue.get("dataset_name") == dataset_name)),
            }
        )

    cfg_index = pd.DataFrame(index_rows)
    if not cfg_index.empty:
        cfg_index = cfg_index.sort_values(by=["dataset_name", "name"]).reset_index(drop=True)
    cfg_index_path = output_dir / "graph_index.csv"
    cfg_index.to_csv(cfg_index_path, index=False)

    validation_issues = validate_outputs(index_rows)
    total_nodes = int(sum(row["num_nodes"] for row in index_rows))
    total_edges = int(sum(row["num_edges"] for row in index_rows))
    total_methods = int(sum(row["num_methods"] for row in index_rows))
    summary = {
        "input_csv": str(input_csv),
        "preprocess_summary": str(summary_path),
        "output_dir": str(output_dir),
        "dataset_filter": args.dataset_name,
        "input_rows": int(len(input_df)),
        "requested_mapped_files": int(len(mapped)),
        "graphs_generated": int(len(index_rows)),
        "soot_graphs": int(sum(1 for row in index_rows if row["extraction_mode"] == "soot")),
        "placeholder_graphs": int(sum(1 for row in index_rows if row["extraction_mode"] == "placeholder")),
        "parse_failures": int(len(parse_failures)),
        "validation_issues": int(len(validation_issues)),
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "total_methods": total_methods,
        "avg_nodes_per_file": float(total_nodes / len(index_rows)) if index_rows else 0.0,
        "avg_edges_per_file": float(total_edges / len(index_rows)) if index_rows else 0.0,
        "structural_feature_dim": STRUCTURAL_FEATURE_DIM,
        "feature_names": FEATURE_NAMES,
        "node_type_embedding_dim": NODE_TYPE_EMBEDDING_DIM,
        "stmt_kind_embedding_dim": STMT_KIND_EMBEDDING_DIM,
        "invoke_kind_embedding_dim": INVOKE_KIND_EMBEDDING_DIM,
        "model_node_feature_dim": MODEL_NODE_FEATURE_DIM,
        "node_feature_dim": STRUCTURAL_FEATURE_DIM,
        "node_type_vocab_size": len(NODE_TYPE_TO_ID),
        "stmt_kind_vocab_size": len(STMT_KIND_TO_ID),
        "invoke_kind_vocab_size": len(INVOKE_KIND_TO_ID),
        "edge_type_vocab_size": len(EDGE_TYPE_TO_ID),
        "backend": "soot",
        "datasets": dataset_summaries,
    }

    (output_dir / "cfg_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "parse_failures.json").write_text(json.dumps(parse_failures, indent=2), encoding="utf-8")
    (output_dir / "validation_issues.json").write_text(json.dumps(validation_issues, indent=2), encoding="utf-8")
    print(
        "PROMISE CFG extraction finished. "
        f"datasets={len(dataset_summaries)} graphs_generated={len(index_rows)} "
        f"soot_graphs={summary['soot_graphs']} placeholders={summary['placeholder_graphs']} "
        f"issues={summary['parse_failures']} validation_issues={summary['validation_issues']}"
    )
    print(f"index={cfg_index_path}")
    print(f"summary={output_dir / 'cfg_summary.json'}")


if __name__ == "__main__":
    main()
