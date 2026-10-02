#!/usr/bin/env python3
"""Shared input, encoder and reporting helpers for NDG experiments."""

from __future__ import annotations

import argparse
import json
import random
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
for path in (SCRIPTS_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import torch

    import cfg_training as cfg_pipeline
    from extract_promise_cfg import EDGE_TYPE_TO_ID as CFG_EDGE_TYPE_TO_ID
    import ast_training as ast_pipeline
    from thesis_project.training.ndg import ProjectGraph, add_inverse_relations, binary_metrics
    from thesis_project.models import ASTEncoderConfig, ASTGINEncoder, ASTGraphClassifier
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing GNN dependencies. Install them with:\n"
        "  .venv/bin/pip install -r requirements.txt\n"
        f"Original error: {exc}"
    ) from exc


@dataclass(frozen=True)
class BaseNDGProject:
    dataset_name: str
    names: list[str]
    source_paths: list[str]
    metrics_x: np.ndarray
    y: np.ndarray
    edge_index: np.ndarray
    edge_type: np.ndarray
    ndg_structural_x: np.ndarray


def validate_cfg_inputs(index_path: Path, edge_vocab: dict[str, int]) -> None:
    index = pd.read_csv(index_path)
    if (index.empty or "construction" not in index
            or not index["construction"].eq("exceptional_control_flow").all()
            or edge_vocab != CFG_EDGE_TYPE_TO_ID):
        raise ValueError("Only canonical CFG inputs are supported. Run scripts/extract_promise_cfg.py.")


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is unavailable")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ast_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        ast_feature_dim=args.ast_feature_dim,
        epochs=args.upstream_epochs,
        batch_size=args.ast_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        min_delta=args.min_delta,
        hidden_dim=args.hidden_dim,
        output_dim=args.embedding_dim,
        num_layers=args.ast_layers,
        node_type_embedding_dim=32,
        dropout=args.dropout,
        seed=args.seed,
        num_workers=args.num_workers,
        no_normalize_structural_features=False,
    )


def cfg_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        epochs=args.upstream_epochs,
        batch_size=args.cfg_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        min_delta=args.min_delta,
        hidden_dim=args.hidden_dim,
        output_dim=args.embedding_dim,
        num_layers=args.cfg_layers,
        node_type_embedding_dim=32,
        stmt_kind_embedding_dim=16,
        invoke_kind_embedding_dim=8,
        edge_type_embedding_dim=16,
        heads=args.heads,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        num_workers=args.num_workers,
        no_normalize_structural_features=False,
    )


def ndg_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        epochs=args.ndg_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
    )


def build_ast_model(args: SimpleNamespace, num_node_types: int, device: torch.device) -> tuple[ASTGraphClassifier, ASTEncoderConfig]:
    config = ASTEncoderConfig(
        num_node_types=num_node_types,
        structural_feature_dim=args.ast_feature_dim,
        node_type_embedding_dim=args.node_type_embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_batch_norm=True,
        bidirectional_edges=True,
    )
    return ASTGraphClassifier(ASTGINEncoder(config), dropout=args.dropout).to(device), config


def encode_ast(
    model: ASTGraphClassifier,
    graph_index: pd.DataFrame,
    indices: np.ndarray,
    args: SimpleNamespace,
    device: torch.device,
) -> dict[tuple[str, str], np.ndarray]:
    dataset = ast_pipeline.ASTGraphDataset(graph_index, indices, normalize_structural_features=True)
    matrix = ast_pipeline.extract_embeddings(model, dataset, args.batch_size, args.num_workers, device, args.output_dim)
    return {
        (str(graph_index.iloc[index]["dataset_name"]), str(graph_index.iloc[index]["name"])): matrix[index]
        for index in indices
    }


def encode_cfg(
    model: cfg_pipeline.CFGGraphClassifier,
    graph_index: pd.DataFrame,
    indices: np.ndarray,
    args: SimpleNamespace,
    device: torch.device,
) -> dict[tuple[str, str], np.ndarray]:
    dataset = cfg_pipeline.CFGGraphDataset(graph_index, indices, normalize_structural_features=True)
    loader = cfg_pipeline.make_loader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    rows, embeddings = cfg_pipeline.extract_test_embeddings(model, loader, device, args.output_dim)
    return {
        (str(graph_index.iloc[index]["dataset_name"]), str(graph_index.iloc[index]["name"])): vector
        for index, vector in zip(rows, embeddings, strict=True)
    }


def load_ndg_structural_matrices(
    index_path: Path,
    feature_names_path: Path,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Load deterministic label-free NDG features keyed by project."""
    index = pd.read_csv(index_path)
    required = {"dataset_name", "num_nodes", "feature_dim", "structural_x_npy"}
    missing = sorted(required - set(index.columns))
    if missing:
        raise ValueError(f"NDG structural index is missing columns: {missing}")
    feature_names = [str(value) for value in load_json(feature_names_path)]
    matrices: dict[str, np.ndarray] = {}
    for _, row in index.iterrows():
        dataset_name = str(row["dataset_name"])
        matrix = np.load(str(row["structural_x_npy"])).astype(np.float32)
        expected_shape = (int(row["num_nodes"]), len(feature_names))
        if matrix.shape != expected_shape:
            raise ValueError(
                f"NDG structural matrix for {dataset_name} has shape {matrix.shape}; "
                f"expected {expected_shape}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError(f"NDG structural matrix for {dataset_name} is not finite")
        matrices[dataset_name] = matrix
    return matrices, feature_names


def validate_structural_provenance(ndg_index_path: Path, structural_index_path: Path) -> None:
    """Do not attach old topology descriptors to the corrected dependency graph."""
    ndgs = pd.read_csv(ndg_index_path)
    features = pd.read_csv(structural_index_path)
    if features.dataset_name.duplicated().any():
        raise ValueError("Duplicate structural project rows")
    if "source_graph_sha256" not in features:
        raise ValueError("Re-extract structural features for the selected NDG")
    features = features.set_index("dataset_name")
    for _, row in ndgs.iterrows():
        digest = hashlib.sha256(Path(row.graph_json).read_bytes()).hexdigest()
        if row.dataset_name not in features.index or features.loc[row.dataset_name, "source_graph_sha256"] != digest:
            raise ValueError(f"Stale or mismatched NDG structural features: {row.dataset_name}")


def load_base_ndgs(
    index_path: Path,
    num_forward_relations: int,
    structural_matrices: dict[str, np.ndarray] | None = None,
) -> dict[str, BaseNDGProject]:
    index = pd.read_csv(index_path)
    required = {"dataset_name", "graph_json", "x_npy", "y_npy", "edge_index_npy", "edge_type_npy"}
    missing = sorted(required - set(index.columns))
    if missing:
        raise ValueError(f"NDG index is missing columns: {missing}")
    if index.empty or "resolution" not in index or not index.resolution.eq("lexical_scope").all():
        raise ValueError("Scope-aware NDG inputs required. Run scripts/extract_promise_ndg.py.")
    projects: dict[str, BaseNDGProject] = {}
    for _, row in index.iterrows():
        dataset_name = str(row["dataset_name"])
        graph = load_json(Path(str(row["graph_json"])))
        nodes = sorted(graph["nodes"], key=lambda node: int(node["id"]))
        edge_index, edge_type = add_inverse_relations(
            np.load(row["edge_index_npy"]).astype(np.int64),
            np.load(row["edge_type_npy"]).astype(np.int64),
            num_forward_relations,
        )
        node_count = len(nodes)
        if structural_matrices is None:
            structural_x = np.empty((node_count, 0), dtype=np.float32)
        else:
            if dataset_name not in structural_matrices:
                raise ValueError(f"Missing NDG structural features for {dataset_name}")
            structural_x = structural_matrices[dataset_name]
            if structural_x.shape[0] != node_count:
                raise ValueError(
                    f"NDG and structural node counts differ for {dataset_name}: "
                    f"{node_count} != {structural_x.shape[0]}"
                )
        projects[dataset_name] = BaseNDGProject(
            dataset_name=dataset_name,
            names=[str(node["name"]) for node in nodes],
            source_paths=[str(node["source_path"]) for node in nodes],
            metrics_x=np.load(row["x_npy"]).astype(np.float32),
            y=np.load(row["y_npy"]).astype(np.float32),
            edge_index=edge_index,
            edge_type=edge_type,
            ndg_structural_x=structural_x,
        )
    return projects


def assemble_ndgs(
    base_projects: dict[str, BaseNDGProject],
    ast_lookup: dict[tuple[str, str], np.ndarray],
    cfg_lookup: dict[tuple[str, str], np.ndarray],
    embedding_dim: int,
    selected_projects: list[str],
) -> dict[str, ProjectGraph]:
    graphs: dict[str, ProjectGraph] = {}
    for project_name in selected_projects:
        base = base_projects[project_name]
        ast_x = np.zeros((len(base.names), embedding_dim), dtype=np.float32)
        cfg_x = np.zeros((len(base.names), embedding_dim), dtype=np.float32)
        mask = np.zeros((len(base.names), 3), dtype=np.bool_)
        mask[:, 0] = True
        for node_id, name in enumerate(base.names):
            key = (project_name, name)
            if key in ast_lookup:
                ast_x[node_id] = ast_lookup[key]
                mask[node_id, 1] = True
            if key in cfg_lookup:
                cfg_x[node_id] = cfg_lookup[key]
                mask[node_id, 2] = True
        graphs[project_name] = ProjectGraph(
            dataset_name=project_name,
            names=base.names,
            source_paths=base.source_paths,
            metrics_x=torch.from_numpy(base.metrics_x),
            ast_x=torch.from_numpy(ast_x),
            cfg_x=torch.from_numpy(cfg_x),
            view_mask=torch.from_numpy(mask),
            y=torch.from_numpy(base.y),
            edge_index=torch.from_numpy(base.edge_index),
            edge_type=torch.from_numpy(base.edge_type),
            ndg_structural_x=torch.from_numpy(base.ndg_structural_x),
        )
    return graphs


def safe_pooled_metrics(
    predictions: pd.DataFrame,
    probability_column: str,
    prediction_column: str,
) -> dict[str, float | None]:
    return binary_metrics(
        predictions["label"].to_numpy(dtype=np.int64),
        predictions[probability_column].to_numpy(dtype=np.float32),
        predictions=predictions[prediction_column].to_numpy(dtype=np.int64),
    )


def aggregate_fold_metrics(fold_metrics: pd.DataFrame, prefix: str) -> dict[str, dict[str, float | int | None]]:
    """Summarize project-level performance without weighting large projects more."""
    summary: dict[str, dict[str, float | int | None]] = {}
    for metric in (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "mcc",
        "g_mean",
        "roc_auc",
        "pr_auc",
        "brier_score",
    ):
        values = pd.to_numeric(fold_metrics[f"{prefix}_{metric}"], errors="coerce").dropna()
        summary[metric] = {
            "mean": float(values.mean()) if not values.empty else None,
            "std": float(values.std(ddof=1)) if len(values) > 1 else None,
            "median": float(values.median()) if not values.empty else None,
            "valid_projects": int(len(values)),
        }
    return summary


def prepare_output_dir(output_dir: Path) -> None:
    """Never silently replace another scenario's results."""
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Output directory is not empty: {output_dir}. Choose a new directory.")
    output_dir.mkdir(parents=True, exist_ok=True)

