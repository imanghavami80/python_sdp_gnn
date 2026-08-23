#!/usr/bin/env python3
"""Run strict nested LOPO evaluation for the final multi-view NDG model."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass, replace
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
    from torch import Tensor, nn

    import evaluate_cfg_lopo as cfg_pipeline
    import generate_ast_embeddings as ast_pipeline
    from thesis_project.training import (
        ClusterFeatureConfig,
        ProjectGraph,
        add_inverse_relations,
        attach_cluster_features,
        binary_metrics,
        combine_graphs,
        evaluate,
        evaluate_attention,
        fit_cluster_features,
        make_model,
        retrain,
        standardize_metrics,
        train_with_validation,
    )
    from thesis_project.models import ASTEncoderConfig, ASTGINEncoder, ASTGraphClassifier, NDGEncoderConfig
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict nested AST + CFG + NDG LOPO node classification.")
    parser.add_argument("--ndg-index", type=Path, default=Path("outputs/promise/ndg/graph_index.csv"))
    parser.add_argument("--ndg-edge-vocab", type=Path, default=Path("outputs/promise/ndg/edge_type_vocab.json"))
    parser.add_argument("--ndg-feature-names", type=Path, default=Path("outputs/promise/ndg/feature_names.json"))
    parser.add_argument("--ast-index", type=Path, default=Path("outputs/promise/ast/graph_index.csv"))
    parser.add_argument("--ast-node-vocab", type=Path, default=Path("outputs/promise/ast/node_type_vocab.json"))
    parser.add_argument("--cfg-index", type=Path, default=Path("outputs/promise/cfg/graph_index.csv"))
    parser.add_argument("--cfg-node-vocab", type=Path, default=Path("outputs/promise/cfg/node_type_vocab.json"))
    parser.add_argument("--cfg-stmt-vocab", type=Path, default=Path("outputs/promise/cfg/stmt_kind_vocab.json"))
    parser.add_argument("--cfg-invoke-vocab", type=Path, default=Path("outputs/promise/cfg/invoke_kind_vocab.json"))
    parser.add_argument("--cfg-edge-vocab", type=Path, default=Path("outputs/promise/cfg/edge_type_vocab.json"))
    parser.add_argument("--cfg-feature-names", type=Path, default=Path("outputs/promise/cfg/feature_names.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/promise/final_ndg_nested_lopo"))
    parser.add_argument("--upstream-epochs", type=int, default=50)
    parser.add_argument("--ndg-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Reserved for upstream loader compatibility.")
    parser.add_argument("--ast-batch-size", type=int, default=32)
    parser.add_argument("--cfg-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--ast-layers", type=int, default=3)
    parser.add_argument("--cfg-layers", type=int, default=3)
    parser.add_argument("--ndg-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--attention-dropout", type=float, default=0.15)
    parser.add_argument(
        "--fusion-stage",
        choices=["early", "late"],
        default="early",
        help="Fuse AST/CFG before NDG propagation (current implementation) or after independent NDG encoding (proposal).",
    )
    parser.add_argument(
        "--cluster-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a gated, training-only cluster geometry and defect-risk branch.",
    )
    parser.add_argument(
        "--cluster-method",
        choices=["kmeans", "gmm"],
        default="kmeans",
        help="Clustering backend. K-means++ remains the default for reproducibility.",
    )
    parser.add_argument(
        "--cluster-count",
        type=int,
        default=None,
        help="Use a fixed component count. Otherwise use silhouette for k-means or BIC for GMM.",
    )
    parser.add_argument("--cluster-min", type=int, default=2, help="Smallest K considered during automatic selection.")
    parser.add_argument("--cluster-max", type=int, default=10, help="Largest K considered during automatic selection.")
    parser.add_argument(
        "--cluster-silhouette-sample-size",
        type=int,
        default=2000,
        help="Maximum training-node sample used to score each candidate K.",
    )
    parser.add_argument("--cluster-n-init", type=int, default=20, help="k-means++ restarts per candidate K.")
    parser.add_argument("--gmm-n-init", type=int, default=5, help="GMM EM restarts per candidate count.")
    parser.add_argument(
        "--gmm-covariance-type",
        choices=["full", "tied", "diag", "spherical"],
        default="full",
        help="GMM covariance structure; full captures correlations among the 20 metrics.",
    )
    parser.add_argument(
        "--gmm-reg-covar",
        type=float,
        default=1e-4,
        help="Positive covariance regularization added during GMM fitting.",
    )
    parser.add_argument(
        "--cluster-risk-smoothing",
        type=float,
        default=20.0,
        help="Pseudo-count strength for training-only cluster defect rates.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--include-ast-fallbacks",
        action="store_true",
        help="Use coarse fallback ASTs. By default they are masked as an unavailable AST view.",
    )
    parser.add_argument("--test-project", action="append", default=[], help="Run only selected outer test projects.")
    return parser.parse_args()


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


def project_indices(index: pd.DataFrame, projects: list[str]) -> np.ndarray:
    return index.index[index["dataset_name"].astype(str).isin(projects)].to_numpy(dtype=np.int64)


def assert_outer_boundary(index: pd.DataFrame, train_indices: np.ndarray, test_project: str, stage: str) -> None:
    used_projects = set(index.iloc[train_indices]["dataset_name"].astype(str))
    if test_project in used_projects:
        raise AssertionError(f"Leakage guard failed: {stage} training includes outer test project {test_project}")


def choose_validation_project(
    base_projects: dict[str, BaseNDGProject],
    outer_train_projects: list[str],
    min_class_nodes: int = 5,
) -> str:
    """Choose a representative inner project with usable binary validation."""
    all_labels = np.concatenate([base_projects[name].y for name in outer_train_projects]).astype(np.int64)
    target_rate = float(all_labels.mean())
    candidates: list[tuple[float, str]] = []
    for name in outer_train_projects:
        labels = base_projects[name].y.astype(np.int64)
        counts = np.bincount(labels, minlength=2)
        if int(counts.min()) < min_class_nodes:
            continue
        candidates.append((abs(float(labels.mean()) - target_rate), name))
    if not candidates:
        raise ValueError(
            f"No inner validation project has at least {min_class_nodes} clean and defective mapped nodes"
        )
    return min(candidates)[1]


def ast_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
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


def cluster_args(
    args: argparse.Namespace,
    random_state: int,
    fixed_clusters: int | None = None,
) -> ClusterFeatureConfig:
    return ClusterFeatureConfig(
        method=args.cluster_method,
        min_clusters=args.cluster_min,
        max_clusters=args.cluster_max,
        fixed_clusters=args.cluster_count if fixed_clusters is None else fixed_clusters,
        silhouette_sample_size=args.cluster_silhouette_sample_size,
        n_init=args.cluster_n_init,
        gmm_n_init=args.gmm_n_init,
        gmm_covariance_type=args.gmm_covariance_type,
        gmm_reg_covar=args.gmm_reg_covar,
        risk_smoothing=args.cluster_risk_smoothing,
        random_state=random_state,
    )


def build_ast_model(args: SimpleNamespace, num_node_types: int, device: torch.device) -> tuple[ASTGraphClassifier, ASTEncoderConfig]:
    config = ASTEncoderConfig(
        num_node_types=num_node_types,
        structural_feature_dim=3,
        node_type_embedding_dim=args.node_type_embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_batch_norm=True,
        bidirectional_edges=True,
    )
    return ASTGraphClassifier(ASTGINEncoder(config), dropout=args.dropout).to(device), config


def train_ast_fixed(
    model: ASTGraphClassifier,
    graph_index: pd.DataFrame,
    train_indices: np.ndarray,
    epochs: int,
    args: SimpleNamespace,
    device: torch.device,
) -> None:
    labels = graph_index["label"].astype(int).to_numpy()
    train_labels = labels[train_indices]
    positives = int(train_labels.sum())
    negatives = int(len(train_labels) - positives)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives if positives else 1.0], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dataset = ast_pipeline.ASTGraphDataset(graph_index, train_indices, normalize_structural_features=True)
    loader = ast_pipeline.make_loader(dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    for _ in range(max(epochs, 1)):
        ast_pipeline.run_epoch(model, loader, loss_fn, device, optimizer=optimizer)


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


def train_ast_fold(
    graph_index: pd.DataFrame,
    vocab: dict[str, int],
    fit_projects: list[str],
    validation_project: str,
    outer_train_projects: list[str],
    test_project: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], np.ndarray], ASTGraphClassifier, ASTEncoderConfig, dict[str, Any]]:
    stage_args = ast_args(args)
    labels = graph_index["label"].astype(int).to_numpy()
    fit_indices = project_indices(graph_index, fit_projects)
    validation_indices = project_indices(graph_index, [validation_project])
    outer_train_indices = project_indices(graph_index, outer_train_projects)
    test_indices = project_indices(graph_index, [test_project])
    assert_outer_boundary(graph_index, fit_indices, test_project, "AST selection")
    assert_outer_boundary(graph_index, outer_train_indices, test_project, "AST final")

    stage_started = time.perf_counter()
    print(f"fold={test_project} stage=ast_selection status=started graphs={len(fit_indices)}", flush=True)
    selection_model, config = build_ast_model(stage_args, len(vocab), device)
    train_loader = ast_pipeline.make_loader(
        ast_pipeline.ASTGraphDataset(graph_index, fit_indices, normalize_structural_features=True),
        stage_args.batch_size,
        shuffle=True,
        num_workers=stage_args.num_workers,
    )
    validation_loader = ast_pipeline.make_loader(
        ast_pipeline.ASTGraphDataset(graph_index, validation_indices, normalize_structural_features=True),
        stage_args.batch_size,
        shuffle=False,
        num_workers=stage_args.num_workers,
    )
    best_state, history, best_info = ast_pipeline.train_model(
        selection_model, train_loader, validation_loader, labels, fit_indices, stage_args, device
    )
    print(
        f"fold={test_project} stage=ast_selection status=finished "
        f"seconds={time.perf_counter() - stage_started:.1f}",
        flush=True,
    )
    selection_model.load_state_dict(best_state)
    stage_started = time.perf_counter()
    print(f"fold={test_project} stage=ast_selection_encoding status=started", flush=True)
    selection_lookup = encode_ast(
        selection_model,
        graph_index,
        np.concatenate([fit_indices, validation_indices]),
        stage_args,
        device,
    )
    print(
        f"fold={test_project} stage=ast_selection_encoding status=finished "
        f"seconds={time.perf_counter() - stage_started:.1f}",
        flush=True,
    )

    set_seed(args.seed)
    stage_started = time.perf_counter()
    print(
        f"fold={test_project} stage=ast_final status=started epochs={int(best_info['best_epoch'])}",
        flush=True,
    )
    final_model, config = build_ast_model(stage_args, len(vocab), device)
    train_ast_fixed(final_model, graph_index, outer_train_indices, int(best_info["best_epoch"]), stage_args, device)
    print(f"fold={test_project} stage=ast_final_encoding status=started", flush=True)
    final_lookup = encode_ast(
        final_model,
        graph_index,
        np.concatenate([outer_train_indices, test_indices]),
        stage_args,
        device,
    )
    print(
        f"fold={test_project} stage=ast_final status=finished "
        f"seconds={time.perf_counter() - stage_started:.1f}",
        flush=True,
    )
    metadata = {"best_info": best_info, "history": history, "config": asdict(config)}
    return selection_lookup, final_lookup, final_model, config, metadata


def train_cfg_fixed(
    model: cfg_pipeline.CFGGraphClassifier,
    graph_index: pd.DataFrame,
    train_indices: np.ndarray,
    epochs: int,
    args: SimpleNamespace,
    device: torch.device,
) -> None:
    labels = graph_index["label"].astype(int).to_numpy()
    loss_fn = cfg_pipeline.make_loss_fn(labels, train_indices, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dataset = cfg_pipeline.CFGGraphDataset(graph_index, train_indices, normalize_structural_features=True)
    loader = cfg_pipeline.make_loader(dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    for _ in range(max(epochs, 1)):
        cfg_pipeline.run_epoch(model, loader, loss_fn, device, optimizer=optimizer)


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


def train_cfg_fold(
    graph_index: pd.DataFrame,
    vocabs: tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, int]],
    feature_names: list[str],
    fit_projects: list[str],
    validation_project: str,
    outer_train_projects: list[str],
    test_project: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], np.ndarray], cfg_pipeline.CFGGraphClassifier, Any, dict[str, Any]]:
    node_vocab, stmt_vocab, invoke_vocab, edge_vocab = vocabs
    stage_args = cfg_args(args)
    labels = graph_index["label"].astype(int).to_numpy()
    fit_indices = project_indices(graph_index, fit_projects)
    validation_indices = project_indices(graph_index, [validation_project])
    outer_train_indices = project_indices(graph_index, outer_train_projects)
    test_indices = project_indices(graph_index, [test_project])
    assert_outer_boundary(graph_index, fit_indices, test_project, "CFG selection")
    assert_outer_boundary(graph_index, outer_train_indices, test_project, "CFG final")

    stage_started = time.perf_counter()
    print(f"fold={test_project} stage=cfg_selection status=started graphs={len(fit_indices)}", flush=True)
    selection_model, config = cfg_pipeline.build_model(
        stage_args, node_vocab, stmt_vocab, invoke_vocab, edge_vocab, feature_names, device
    )
    train_loader = cfg_pipeline.make_loader(
        cfg_pipeline.CFGGraphDataset(graph_index, fit_indices, normalize_structural_features=True),
        stage_args.batch_size,
        shuffle=True,
        num_workers=stage_args.num_workers,
    )
    validation_loader = cfg_pipeline.make_loader(
        cfg_pipeline.CFGGraphDataset(graph_index, validation_indices, normalize_structural_features=True),
        stage_args.batch_size,
        shuffle=False,
        num_workers=stage_args.num_workers,
    )
    best_state, history, best_info = cfg_pipeline.train_model(
        selection_model, train_loader, validation_loader, labels, fit_indices, stage_args, device
    )
    print(
        f"fold={test_project} stage=cfg_selection status=finished "
        f"seconds={time.perf_counter() - stage_started:.1f}",
        flush=True,
    )
    selection_model.load_state_dict(best_state)
    stage_started = time.perf_counter()
    print(f"fold={test_project} stage=cfg_selection_encoding status=started", flush=True)
    selection_lookup = encode_cfg(
        selection_model,
        graph_index,
        np.concatenate([fit_indices, validation_indices]),
        stage_args,
        device,
    )
    print(
        f"fold={test_project} stage=cfg_selection_encoding status=finished "
        f"seconds={time.perf_counter() - stage_started:.1f}",
        flush=True,
    )

    set_seed(args.seed)
    stage_started = time.perf_counter()
    print(
        f"fold={test_project} stage=cfg_final status=started epochs={int(best_info['best_epoch'])}",
        flush=True,
    )
    final_model, config = cfg_pipeline.build_model(
        stage_args, node_vocab, stmt_vocab, invoke_vocab, edge_vocab, feature_names, device
    )
    train_cfg_fixed(final_model, graph_index, outer_train_indices, int(best_info["best_epoch"]), stage_args, device)
    final_lookup = encode_cfg(
        final_model,
        graph_index,
        np.concatenate([outer_train_indices, test_indices]),
        stage_args,
        device,
    )
    print(
        f"fold={test_project} stage=cfg_final status=finished "
        f"seconds={time.perf_counter() - stage_started:.1f}",
        flush=True,
    )
    metadata = {"best_info": best_info, "history": history, "config": asdict(config)}
    return selection_lookup, final_lookup, final_model, config, metadata


def load_base_ndgs(index_path: Path, num_forward_relations: int) -> dict[str, BaseNDGProject]:
    index = pd.read_csv(index_path)
    required = {"dataset_name", "graph_json", "x_npy", "y_npy", "edge_index_npy", "edge_type_npy"}
    missing = sorted(required - set(index.columns))
    if missing:
        raise ValueError(f"NDG index is missing columns: {missing}")
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
        projects[dataset_name] = BaseNDGProject(
            dataset_name=dataset_name,
            names=[str(node["name"]) for node in nodes],
            source_paths=[str(node["source_path"]) for node in nodes],
            metrics_x=np.load(row["x_npy"]).astype(np.float32),
            y=np.load(row["y_npy"]).astype(np.float32),
            edge_index=edge_index,
            edge_type=edge_type,
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
            loss_weight=torch.ones(len(base.names), dtype=torch.float32),
            y=torch.from_numpy(base.y),
            edge_index=torch.from_numpy(base.edge_index),
            edge_type=torch.from_numpy(base.edge_type),
        )
    return graphs


def save_upstream_checkpoint(
    path: Path,
    model: nn.Module,
    config: Any,
    test_project: str,
    train_projects: list[str],
    validation_project: str,
    best_info: dict[str, Any],
) -> None:
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "encoder_config": asdict(config),
            "outer_test_project": test_project,
            "outer_train_projects": train_projects,
            "inner_validation_project": validation_project,
            "selected_epochs": int(best_info["best_epoch"]),
            "protocol": "strict_nested_LOPO",
        },
        path,
    )


def run_outer_fold(
    fold_index: int,
    test_project: str,
    all_projects: list[str],
    base_ndgs: dict[str, BaseNDGProject],
    ast_index: pd.DataFrame,
    ast_vocab: dict[str, int],
    cfg_index: pd.DataFrame,
    cfg_vocabs: tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, int]],
    cfg_feature_names: list[str],
    ndg_config: NDGEncoderConfig,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    outer_train_projects = [project for project in all_projects if project != test_project]
    validation_project = choose_validation_project(base_ndgs, outer_train_projects)
    fit_projects = [project for project in outer_train_projects if project != validation_project]
    if test_project in fit_projects or test_project == validation_project:
        raise AssertionError("Outer test project crossed the nested training boundary")
    fold_dir = output_dir / "folds" / test_project
    fold_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed + fold_index)

    ast_selection, ast_final, ast_model, ast_config, ast_meta = train_ast_fold(
        ast_index,
        ast_vocab,
        fit_projects,
        validation_project,
        outer_train_projects,
        test_project,
        args,
        device,
    )
    set_seed(args.seed + 1000 + fold_index)
    cfg_selection, cfg_final, cfg_model, cfg_config, cfg_meta = train_cfg_fold(
        cfg_index,
        cfg_vocabs,
        cfg_feature_names,
        fit_projects,
        validation_project,
        outer_train_projects,
        test_project,
        args,
        device,
    )

    selection_graphs = assemble_ndgs(
        base_ndgs,
        ast_selection,
        cfg_selection,
        args.embedding_dim,
        outer_train_projects,
    )
    fit_graph = combine_graphs([selection_graphs[project] for project in fit_projects])
    validation_graph = selection_graphs[validation_project]
    fit_graph, validation_graph = standardize_metrics(fit_graph, validation_graph)
    cluster_metadata: dict[str, Any] | None = None
    selected_cluster_count: int | None = None
    if args.cluster_features:
        fit_groups = np.concatenate(
            [np.full(selection_graphs[project].num_nodes, project) for project in fit_projects]
        )
        selection_clusterer = fit_cluster_features(
            fit_graph.metrics_x,
            fit_graph.y,
            fit_groups,
            cluster_args(args, args.seed + 4000 + fold_index),
        )
        selected_cluster_count = selection_clusterer.num_clusters
        fit_graph = attach_cluster_features(
            selection_clusterer,
            fit_graph,
            training_groups=fit_groups,
        )
        validation_graph = attach_cluster_features(selection_clusterer, validation_graph)
        cluster_metadata = {"selection": selection_clusterer.metadata()}
        print(
            f"fold={test_project} stage=cluster_selection status=finished "
            f"method={args.cluster_method} clusters={selected_cluster_count} "
            f"features={len(selection_clusterer.feature_names)}",
            flush=True,
        )
        if args.cluster_count is None and selected_cluster_count in {
            args.cluster_min,
            args.cluster_max,
        }:
            print(
                f"fold={test_project} stage=cluster_selection status=boundary "
                f"selected={selected_cluster_count} range={args.cluster_min}..{args.cluster_max}",
                flush=True,
            )
    cluster_dim = 0 if fit_graph.cluster_x is None else int(fit_graph.cluster_x.size(1))
    fold_ndg_config = replace(ndg_config, cluster_dim=cluster_dim)
    set_seed(args.seed + 2000 + fold_index)
    stage_started = time.perf_counter()
    print(f"fold={test_project} stage=ndg_selection status=started", flush=True)
    selection_ndg = make_model(fold_ndg_config, device)
    best_ndg_epoch, decision_threshold, ndg_history = train_with_validation(
        selection_ndg,
        fit_graph.to(device),
        validation_graph.to(device),
        ndg_args(args),
    )
    print(
        f"fold={test_project} stage=ndg_selection status=finished "
        f"seconds={time.perf_counter() - stage_started:.1f} best_epoch={best_ndg_epoch} "
        f"threshold={decision_threshold:.4f}",
        flush=True,
    )

    final_graphs = assemble_ndgs(
        base_ndgs,
        ast_final,
        cfg_final,
        args.embedding_dim,
        [*outer_train_projects, test_project],
    )
    full_train = combine_graphs([final_graphs[project] for project in outer_train_projects])
    test_graph = final_graphs[test_project]
    full_train, test_graph = standardize_metrics(full_train, test_graph)
    test_cluster_assignments = np.full(test_graph.num_nodes, -1, dtype=np.int64)
    test_cluster_confidence = np.full(test_graph.num_nodes, np.nan, dtype=np.float32)
    test_cluster_outlier_distance = np.full(test_graph.num_nodes, np.nan, dtype=np.float32)
    test_cluster_defect_risk = np.full(test_graph.num_nodes, np.nan, dtype=np.float32)
    if args.cluster_features:
        if selected_cluster_count is None or cluster_metadata is None:
            raise AssertionError("Cluster selection metadata is missing")
        full_train_groups = np.concatenate(
            [np.full(final_graphs[project].num_nodes, project) for project in outer_train_projects]
        )
        final_clusterer = fit_cluster_features(
            full_train.metrics_x,
            full_train.y,
            full_train_groups,
            cluster_args(args, args.seed + 5000 + fold_index, fixed_clusters=selected_cluster_count),
        )
        test_cluster_assignments = final_clusterer.assignments(test_graph.metrics_x)
        test_cluster_features = final_clusterer.transform(test_graph.metrics_x)
        membership_start = final_clusterer.num_clusters
        membership_end = 2 * final_clusterer.num_clusters
        test_cluster_confidence = test_cluster_features[:, membership_start:membership_end].max(axis=1)
        test_cluster_outlier_distance = test_cluster_features[:, -2]
        test_cluster_defect_risk = test_cluster_features[:, -1]
        full_train = attach_cluster_features(
            final_clusterer,
            full_train,
            training_groups=full_train_groups,
        )
        test_graph = attach_cluster_features(final_clusterer, test_graph)
        cluster_metadata["final"] = final_clusterer.metadata()
        (fold_dir / "cluster_features.json").write_text(
            json.dumps(cluster_metadata, indent=2), encoding="utf-8"
        )
    final_cluster_dim = 0 if full_train.cluster_x is None else int(full_train.cluster_x.size(1))
    if int(full_train.metrics_x.size(1)) != fold_ndg_config.metrics_dim:
        raise AssertionError("Clustering unexpectedly changed the original metric dimension")
    if final_cluster_dim != fold_ndg_config.cluster_dim:
        raise AssertionError("Selection and final cluster feature dimensions differ")
    set_seed(args.seed + 3000 + fold_index)
    stage_started = time.perf_counter()
    print(f"fold={test_project} stage=ndg_final status=started epochs={best_ndg_epoch}", flush=True)
    final_ndg = make_model(fold_ndg_config, device)
    retrain(final_ndg, full_train.to(device), best_ndg_epoch, ndg_args(args))
    device_test_graph = test_graph.to(device)
    embeddings, probabilities, labels = evaluate(final_ndg, device_test_graph)
    attention_diagnostics = evaluate_attention(final_ndg, device_test_graph)
    test_cluster_gate = (
        attention_diagnostics["cluster_gate"].reshape(-1).astype(np.float32)
        if args.cluster_features
        else np.full(test_graph.num_nodes, np.nan, dtype=np.float32)
    )
    print(
        f"fold={test_project} stage=ndg_final status=finished "
        f"seconds={time.perf_counter() - stage_started:.1f}",
        flush=True,
    )
    model_predictions = (probabilities >= decision_threshold).astype(np.int64)
    metrics = binary_metrics(labels, probabilities, decision_threshold, model_predictions)
    majority_label = int(full_train.y.float().mean().item() >= 0.5)
    baseline_probabilities = np.full(len(labels), float(majority_label), dtype=np.float32)
    baseline_predictions = np.full(len(labels), majority_label, dtype=np.int64)
    baseline_metrics = binary_metrics(
        labels,
        baseline_probabilities,
        predictions=baseline_predictions,
    )

    split = {
        "outer_test_project": test_project,
        "outer_train_projects": outer_train_projects,
        "inner_validation_project": validation_project,
        "inner_fit_projects": fit_projects,
        "test_project_used_by_ast_training": False,
        "test_project_used_by_cfg_training": False,
        "test_project_used_by_ndg_training": False,
        "test_project_used_by_clustering": False,
        "test_project_used_by_cluster_defect_risk": False,
        "test_project_used_for_threshold_selection": False,
    }
    (fold_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    pd.DataFrame(ast_meta["history"]).to_csv(fold_dir / "ast_selection_history.csv", index=False)
    pd.DataFrame(cfg_meta["history"]).to_csv(fold_dir / "cfg_selection_history.csv", index=False)
    pd.DataFrame(ndg_history).to_csv(fold_dir / "ndg_selection_history.csv", index=False)
    save_upstream_checkpoint(
        fold_dir / "ast_encoder.pt",
        ast_model,
        ast_config,
        test_project,
        outer_train_projects,
        validation_project,
        ast_meta["best_info"],
    )
    save_upstream_checkpoint(
        fold_dir / "cfg_encoder.pt",
        cfg_model,
        cfg_config,
        test_project,
        outer_train_projects,
        validation_project,
        cfg_meta["best_info"],
    )
    torch.save(
        {
            "model_state_dict": final_ndg.cpu().state_dict(),
            "encoder_config": asdict(fold_ndg_config),
            "outer_test_project": test_project,
            "outer_train_projects": outer_train_projects,
            "inner_validation_project": validation_project,
            "selected_epochs": best_ndg_epoch,
            "decision_threshold": decision_threshold,
            "protocol": "strict_nested_LOPO",
            "cluster_features": cluster_metadata,
        },
        fold_dir / "ndg_encoder.pt",
    )
    np.save(fold_dir / "test_node_embeddings.npy", embeddings)
    predictions = pd.DataFrame(
        {
            "dataset_name": test_project,
            "name": test_graph.names,
            "source_path": test_graph.source_paths,
            "label": labels,
            "probability": probabilities,
            "decision_threshold": decision_threshold,
            "prediction": model_predictions,
            "baseline_probability": baseline_probabilities,
            "baseline_prediction": baseline_predictions,
            "has_ast": test_graph.view_mask[:, 1].cpu().numpy().astype(np.int64),
            "has_cfg": test_graph.view_mask[:, 2].cpu().numpy().astype(np.int64),
            "cluster_id": test_cluster_assignments,
            "cluster_confidence": test_cluster_confidence,
            "cluster_outlier_distance": test_cluster_outlier_distance,
            "cluster_defect_risk": test_cluster_defect_risk,
            "cluster_gate": test_cluster_gate,
        }
    )
    predictions.to_csv(fold_dir / "test_node_predictions.csv", index=False)
    fold_row = {
        "test_project": test_project,
        "validation_project": validation_project,
        "train_nodes": full_train.num_nodes,
        "test_nodes": test_graph.num_nodes,
        "ast_selected_epochs": int(ast_meta["best_info"]["best_epoch"]),
        "cfg_selected_epochs": int(cfg_meta["best_info"]["best_epoch"]),
        "ndg_selected_epochs": int(best_ndg_epoch),
        "decision_threshold": decision_threshold,
        "cluster_count": selected_cluster_count,
        "cluster_method": args.cluster_method if args.cluster_features else None,
        "cluster_feature_dim": 0 if selected_cluster_count is None else 2 * selected_cluster_count + 2,
        "cluster_gate_mean": float(np.nanmean(test_cluster_gate)) if args.cluster_features else None,
        "cluster_gate_std": float(np.nanstd(test_cluster_gate)) if args.cluster_features else None,
        **{f"model_{key}": value for key, value in metrics.items()},
        **{f"baseline_{key}": value for key, value in baseline_metrics.items()},
    }
    return fold_row, predictions, embeddings


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
    """Clean a specific experiment directory while refusing broad targets."""
    forbidden = {Path(output_dir.anchor), Path.home().resolve(), REPO_ROOT.resolve()}
    if output_dir.resolve() in forbidden:
        raise ValueError(f"Refusing to clean unsafe output directory: {output_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    if min(args.upstream_epochs, args.ndg_epochs, args.patience, args.ast_batch_size, args.cfg_batch_size) <= 0:
        raise ValueError("Epoch, patience, and batch-size arguments must be positive")
    if args.cluster_features:
        cluster_args(args, args.seed)
    elif args.cluster_count is not None:
        raise ValueError("--cluster-count requires --cluster-features")
    output_dir = resolve_path(args.output_dir)
    prepare_output_dir(output_dir)
    device = choose_device(args.device)

    ndg_edge_vocab = {str(key): int(value) for key, value in load_json(resolve_path(args.ndg_edge_vocab)).items()}
    ndg_feature_names = [str(value) for value in load_json(resolve_path(args.ndg_feature_names))]
    base_ndgs = load_base_ndgs(resolve_path(args.ndg_index), len(ndg_edge_vocab))
    all_projects = sorted(base_ndgs)
    selected_test_projects = args.test_project or all_projects
    unknown = sorted(set(selected_test_projects) - set(all_projects))
    if unknown:
        raise ValueError(f"Unknown test projects: {unknown}")

    ast_index_all, ast_vocab = ast_pipeline.load_inputs(resolve_path(args.ast_index), resolve_path(args.ast_node_vocab))
    ast_fallback_count = int((ast_index_all["parser_mode"].astype(str) == "fallback").sum())
    ast_index = ast_index_all.copy()
    if not args.include_ast_fallbacks:
        ast_index = ast_index[ast_index["parser_mode"].astype(str) != "fallback"].reset_index(drop=True)
    cfg_index, node_vocab, stmt_vocab, invoke_vocab, cfg_edge_vocab, cfg_feature_names, _ = cfg_pipeline.load_inputs(
        resolve_path(args.cfg_index),
        resolve_path(args.cfg_node_vocab),
        resolve_path(args.cfg_stmt_vocab),
        resolve_path(args.cfg_invoke_vocab),
        resolve_path(args.cfg_edge_vocab),
        resolve_path(args.cfg_feature_names),
    )
    if set(all_projects) != set(ast_index_all["dataset_name"].astype(str).unique()):
        raise ValueError("AST and NDG project sets differ")
    missing_ast_projects = sorted(set(all_projects) - set(ast_index["dataset_name"].astype(str).unique()))
    if missing_ast_projects:
        raise ValueError(f"No usable AST graphs remain for projects: {missing_ast_projects}")
    if not set(all_projects).issuperset(set(cfg_index["dataset_name"].astype(str).unique())):
        raise ValueError("CFG index contains a project absent from NDG")

    ndg_config = NDGEncoderConfig(
        metrics_dim=len(ndg_feature_names),
        ast_dim=args.embedding_dim,
        cfg_dim=args.embedding_dim,
        num_edge_types=2 * len(ndg_edge_vocab),
        hidden_dim=args.hidden_dim,
        output_dim=args.embedding_dim,
        edge_type_embedding_dim=16,
        num_layers=args.ndg_layers,
        heads=args.heads,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        fusion_stage=args.fusion_stage,
    )
    print(
        f"Strict nested LOPO started: outer_folds={len(selected_test_projects)} projects={len(all_projects)} "
        f"nodes={sum(len(project.names) for project in base_ndgs.values())} device={device}",
        flush=True,
    )
    fold_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    embedding_arrays: list[np.ndarray] = []
    embedding_index_frames: list[pd.DataFrame] = []
    for test_project in selected_test_projects:
        fold_index = all_projects.index(test_project)
        fold_row, predictions, embeddings = run_outer_fold(
            fold_index,
            test_project,
            all_projects,
            base_ndgs,
            ast_index,
            ast_vocab,
            cfg_index,
            (node_vocab, stmt_vocab, invoke_vocab, cfg_edge_vocab),
            cfg_feature_names,
            ndg_config,
            args,
            device,
            output_dir,
        )
        fold_rows.append(fold_row)
        prediction_frames.append(predictions)
        embedding_arrays.append(embeddings)
        embedding_index_frames.append(
            predictions[["dataset_name", "name", "source_path", "label", "has_ast", "has_cfg"]].copy()
        )
        print(
            f"outer_fold={test_project} nodes={fold_row['test_nodes']} f1={fold_row['model_f1']:.4f} "
            f"roc_auc={fold_row['model_roc_auc']}",
            flush=True,
        )

    fold_metrics = pd.DataFrame(fold_rows)
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    all_embeddings = np.concatenate(embedding_arrays, axis=0).astype(np.float32)
    embedding_index = pd.concat(embedding_index_frames, ignore_index=True)
    embedding_index.insert(0, "embedding_row", np.arange(len(embedding_index), dtype=np.int64))
    embeddings_path = output_dir / "ndg_node_embeddings.npy"
    embedding_index["embedding_npy"] = str(embeddings_path)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    all_predictions.to_csv(output_dir / "all_test_node_predictions.csv", index=False)
    np.save(embeddings_path, all_embeddings)
    embedding_index.to_csv(output_dir / "ndg_node_embedding_index.csv", index=False)
    summary = {
        "protocol": "strict_nested_LOPO",
        "prediction_granularity": "file_node",
        "outer_test_projects": selected_test_projects,
        "all_projects": all_projects,
        "completed_folds": len(selected_test_projects),
        "nodes_evaluated": len(all_predictions),
        "embedding_shape": list(all_embeddings.shape),
        "test_project_excluded_from": [
            "AST training and epoch selection",
            "CFG training and epoch selection",
            "NDG training and epoch selection",
            "metric normalization",
            "cluster-count selection and cluster fitting",
            "cluster defect-risk estimation",
            "decision-threshold selection",
        ],
        "model_pooled_metrics": safe_pooled_metrics(all_predictions, "probability", "prediction"),
        "baseline_pooled_metrics": safe_pooled_metrics(
            all_predictions,
            "baseline_probability",
            "baseline_prediction",
        ),
        "model_macro_project_metrics": aggregate_fold_metrics(fold_metrics, "model"),
        "baseline_macro_project_metrics": aggregate_fold_metrics(fold_metrics, "baseline"),
        "ndg_encoder_config_template": asdict(ndg_config),
        "cluster_features": {
            "enabled": args.cluster_features,
            "algorithm": (
                "k-means++" if args.cluster_method == "kmeans" else "gaussian-mixture"
            ) if args.cluster_features else None,
            "input": "training-fold standardized handcrafted metrics",
            "uses_defect_labels_for_cluster_geometry": False,
            "defect_labels_used_for": (
                "cross-fitted smoothed cluster risk only" if args.cluster_features else None
            ),
            "risk_smoothing": args.cluster_risk_smoothing if args.cluster_features else None,
            "training_project_weighting": (
                "equal total weight per project" if args.cluster_features else None
            ),
            "gmm_project_balancing": (
                "deterministic equal-project resampling"
                if args.cluster_features and args.cluster_method == "gmm"
                else None
            ),
            "fixed_cluster_count": args.cluster_count,
            "candidate_range": [args.cluster_min, args.cluster_max] if args.cluster_features else None,
            "selection_criterion": (
                "maximum silhouette score" if args.cluster_method == "kmeans" else "minimum BIC"
            ) if args.cluster_features else None,
            "gmm_covariance_type": (
                args.gmm_covariance_type
                if args.cluster_features and args.cluster_method == "gmm"
                else None
            ),
            "gmm_reg_covar": (
                args.gmm_reg_covar
                if args.cluster_features and args.cluster_method == "gmm"
                else None
            ),
            "gmm_n_init": (
                args.gmm_n_init
                if args.cluster_features and args.cluster_method == "gmm"
                else None
            ),
            "selected_counts_by_fold": {
                str(row["test_project"]): row["cluster_count"] for row in fold_rows
            },
            "features_per_fold": {
                str(row["test_project"]): row["cluster_feature_dim"] for row in fold_rows
            },
        },
        "cfg_placeholder_policy": "Missing CFG view is masked; the NDG file node is retained.",
        "ast_fallback_policy": "included" if args.include_ast_fallbacks else "masked_as_missing_view",
        "ast_fallback_graphs": ast_fallback_count,
        "metric_transform": (
            "training-fold median imputation and standard scaling; a separate gated cluster branch "
            "uses cluster distances, soft memberships, outlier distance, and cross-fitted defect risk"
            if args.cluster_features
            else "training-fold median imputation followed by training-fold standard scaling"
        ),
        "random_seed": args.seed,
        "classification_threshold": "selected on the inner-validation project independently per outer fold",
        "training_project_weighting": "equal total loss contribution per outer-training project",
    }
    (output_dir / "nested_lopo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"node_embeddings={embeddings_path}", flush=True)
    print(f"predictions={output_dir / 'all_test_node_predictions.csv'}", flush=True)
    print("Strict nested LOPO finished.", flush=True)


if __name__ == "__main__":
    main()
