#!/usr/bin/env python3
"""Reusable file-view training for within-project evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import torch
    from torch import Tensor, nn
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    from thesis_project.models import (
        CFGEdgeAwareGATEncoder,
        CFGEncoderConfig,
        CFGGraphClassifier,
        normalize_cfg_structural_features,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only in missing dependency environments.
    raise SystemExit(
        "Missing GNN dependencies. Install them with:\n"
        "  .venv/bin/pip install -r requirements.txt\n"
        f"Original error: {exc}"
    ) from exc

DEFAULT_FEATURE_NAMES = [
    "line_position",
    "has_source_line",
    "is_synthetic",
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


class CFGGraphDataset:
    """Lazy dataset for CFG tensor files listed in `graph_index.csv`."""

    def __init__(
        self,
        graph_index: pd.DataFrame,
        indices: list[int] | np.ndarray | None = None,
        normalize_structural_features: bool = True,
    ) -> None:
        self.graph_index = graph_index.reset_index(drop=True)
        self.normalize_structural_features = normalize_structural_features
        if indices is None:
            self.indices = list(range(len(self.graph_index)))
        else:
            self.indices = [int(index) for index in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Data:
        row_idx = self.indices[item]
        row = self.graph_index.iloc[row_idx]
        x = torch.from_numpy(np.load(row["x_npy"])).float()
        if self.normalize_structural_features:
            x = normalize_cfg_structural_features(x)
        node_type_id = torch.from_numpy(np.load(row["node_type_id_npy"])).long()
        stmt_kind_id = torch.from_numpy(np.load(row["stmt_kind_id_npy"])).long()
        invoke_kind_id = torch.from_numpy(np.load(row["invoke_kind_id_npy"])).long()
        edge_index = torch.from_numpy(np.load(row["edge_index_npy"])).long()
        edge_type = torch.from_numpy(np.load(row["edge_type_npy"])).long()
        y = torch.tensor([float(row["label"])], dtype=torch.float32)
        return Data(
            x=x,
            node_type_id=node_type_id,
            stmt_kind_id=stmt_kind_id,
            invoke_kind_id=invoke_kind_id,
            edge_index=edge_index,
            edge_type=edge_type,
            y=y,
            row_idx=torch.tensor([row_idx], dtype=torch.long),
        )


def load_vocab(path: Path, label: str) -> dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CFG {label} vocabulary: {path}. Run scripts/extract_promise_cfg.py first.")
    vocab = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): int(value) for key, value in vocab.items()}


def load_feature_names(path: Path) -> list[str]:
    if not path.exists():
        return DEFAULT_FEATURE_NAMES
    feature_names = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(feature_names, list) or not all(isinstance(item, str) for item in feature_names):
        raise ValueError(f"Invalid CFG feature name file: {path}")
    return feature_names


def load_inputs(
    graph_index_path: Path,
    node_vocab_path: Path,
    stmt_vocab_path: Path,
    invoke_vocab_path: Path,
    edge_vocab_path: Path,
    feature_names_path: Path,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, int], dict[str, int], dict[str, int], list[str], dict[str, int]]:
    if not graph_index_path.exists():
        raise FileNotFoundError(f"Missing CFG graph index: {graph_index_path}. Run scripts/extract_promise_cfg.py first.")

    graph_index = pd.read_csv(graph_index_path)
    required = {
        "graph_id",
        "dataset_name",
        "name",
        "source_path",
        "label",
        "extraction_mode",
        "x_npy",
        "node_type_id_npy",
        "stmt_kind_id_npy",
        "invoke_kind_id_npy",
        "edge_index_npy",
        "edge_type_npy",
    }
    missing = sorted(required - set(graph_index.columns))
    if missing:
        raise ValueError(f"CFG graph index is missing required columns: {missing}")

    original_modes = graph_index["extraction_mode"].astype(str).value_counts().sort_index().astype(int).to_dict()
    graph_index = graph_index[graph_index["extraction_mode"].astype(str) != "placeholder"].copy()
    if graph_index.empty:
        raise ValueError("CFG graph index is empty after excluding placeholder graphs")

    labels = set(graph_index["label"].dropna().astype(int).unique())
    if labels - {0, 1}:
        raise ValueError(f"CFG labels must be binary, found: {sorted(labels)}")

    node_vocab = load_vocab(node_vocab_path, "node-type")
    stmt_vocab = load_vocab(stmt_vocab_path, "statement-kind")
    invoke_vocab = load_vocab(invoke_vocab_path, "invocation-kind")
    edge_vocab = load_vocab(edge_vocab_path, "edge-type")
    feature_names = load_feature_names(feature_names_path)
    return graph_index.reset_index(drop=True), node_vocab, stmt_vocab, invoke_vocab, edge_vocab, feature_names, original_modes


def build_model(
    args: argparse.Namespace,
    node_vocab: dict[str, int],
    stmt_vocab: dict[str, int],
    invoke_vocab: dict[str, int],
    edge_vocab: dict[str, int],
    feature_names: list[str],
    device: torch.device,
) -> tuple[CFGGraphClassifier, CFGEncoderConfig]:
    config = CFGEncoderConfig(
        num_node_types=len(node_vocab),
        num_stmt_kinds=len(stmt_vocab),
        num_invoke_kinds=len(invoke_vocab),
        num_edge_types=len(edge_vocab),
        structural_feature_dim=len(feature_names),
        node_type_embedding_dim=args.node_type_embedding_dim,
        stmt_kind_embedding_dim=args.stmt_kind_embedding_dim,
        invoke_kind_embedding_dim=args.invoke_kind_embedding_dim,
        edge_type_embedding_dim=args.edge_type_embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        use_layer_norm=True,
        add_self_loops=False,
    )
    model = CFGGraphClassifier(CFGEdgeAwareGATEncoder(config), dropout=args.dropout).to(device)
    return model, config


def make_loader(dataset: CFGGraphDataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def move_batch(batch: Data, device: torch.device) -> Data:
    return batch.to(device)


def safe_metric_dict(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float | None]:
    y_true = y_true.astype(int)
    y_pred = (y_prob >= threshold).astype(int)
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(set(y_true.tolist())) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
    return metrics


def make_loss_fn(labels: np.ndarray, train_indices: np.ndarray, device: torch.device) -> nn.Module:
    train_labels = labels[train_indices]
    positives = int(train_labels.sum())
    negatives = int(len(train_labels) - positives)
    pos_weight_value = float(negatives / positives) if positives else 1.0
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def run_epoch(
    model: CFGGraphClassifier,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, dict[str, float | None]]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_graphs = 0
    logits_list: list[Tensor] = []
    targets_list: list[Tensor] = []

    for batch in loader:
        batch = move_batch(batch, device)
        targets = batch.y.float().view(-1)
        if is_training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(
            batch.x,
            batch.node_type_id,
            batch.stmt_kind_id,
            batch.invoke_kind_id,
            batch.edge_index,
            batch.edge_type,
            batch.batch,
        )
        loss = loss_fn(logits, targets)
        if is_training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        batch_size = int(targets.numel())
        total_loss += float(loss.item()) * batch_size
        total_graphs += batch_size
        logits_list.append(logits.detach())
        targets_list.append(targets.detach())

    if total_graphs == 0:
        raise ValueError("Cannot run an epoch on an empty loader")
    logits = torch.cat(logits_list)
    targets = torch.cat(targets_list)
    return total_loss / total_graphs, safe_metric_dict(
        targets.detach().cpu().numpy().astype(int),
        torch.sigmoid(logits).detach().cpu().numpy(),
    )


def train_model(
    model: CFGGraphClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    labels: np.ndarray,
    train_indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Tensor], list[dict[str, Any]], dict[str, Any]]:
    train_labels = labels[train_indices]
    positives = int(train_labels.sum())
    negatives = int(len(train_labels) - positives)
    pos_weight_value = float(negatives / positives) if positives else 1.0
    loss_fn = make_loss_fn(labels, train_indices, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, Any]] = []
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_score = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, loss_fn, device, optimizer=optimizer)
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"train_{key}": value for key, value in train_metrics.items()},
        }

        if val_loader is not None:
            with torch.no_grad():
                val_loss, val_metrics = run_epoch(model, val_loader, loss_fn, device)
            row.update({"val_loss": val_loss, **{f"val_{key}": value for key, value in val_metrics.items()}})
            score = val_loss
        else:
            score = train_loss

        history.append(row)
        print(
            f"behavior_epoch={epoch:03d} train_loss={train_loss:.4f} "
            + (f"val_loss={val_loss:.4f} " if val_loader is not None else "")
            + f"train_f1={train_metrics['f1']:.4f}",
            flush=True,
        )
        if score < best_score - args.min_delta:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if val_loader is not None and epochs_without_improvement >= args.patience:
            print(
                f"cfg_early_stopping epoch={epoch} best_epoch={best_epoch} "
                f"best_val_loss={best_score:.4f}",
                flush=True,
            )
            break

    best_info = {
        "best_epoch": best_epoch,
        "best_loss": best_score,
        "pos_weight": pos_weight_value,
        "train_positives": positives,
        "train_negatives": negatives,
    }
    return best_state, history, best_info


def extract_test_embeddings(
    model: CFGGraphClassifier,
    loader: DataLoader,
    device: torch.device,
    output_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    rows: list[np.ndarray] = []
    vectors: list[np.ndarray] = []
    with torch.no_grad():
        total_batches = len(loader)
        for batch_number, batch in enumerate(loader, start=1):
            batch = move_batch(batch, device)
            embeddings = model.encode(
                batch.x,
                batch.node_type_id,
                batch.stmt_kind_id,
                batch.invoke_kind_id,
                batch.edge_index,
                batch.edge_type,
                batch.batch,
            )
            rows.append(batch.row_idx.view(-1).detach().cpu().numpy().astype(int))
            vectors.append(embeddings.detach().cpu().numpy().astype(np.float32))
            if batch_number == 1 or batch_number % 25 == 0 or batch_number == total_batches:
                print(f"cfg_encoding batch={batch_number}/{total_batches}", flush=True)
    if not rows:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, output_dim), dtype=np.float32)
    return np.concatenate(rows), np.concatenate(vectors, axis=0)


