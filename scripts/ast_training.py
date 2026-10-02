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

    from thesis_project.models import ASTEncoderConfig, ASTGINEncoder, ASTGraphClassifier, normalize_ast_structural_features
    from thesis_project.features.ast_schema import FEATURE_NAMES as SYNTAX_FEATURE_NAMES
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only in missing dependency environments.
    raise SystemExit(
        "Missing GNN dependencies. Install them with:\n"
        "  .venv/bin/pip install -r requirements.txt\n"
        f"Original error: {exc}"
    ) from exc

class ASTGraphDataset:
    """Lazy dataset for AST tensor files listed in `graph_index.csv`."""

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
            x = normalize_ast_structural_features(x)
        node_type_id = torch.from_numpy(np.load(row["node_type_id_npy"])).long()
        edge_index = torch.from_numpy(np.load(row["edge_index_npy"])).long()
        y = torch.tensor([float(row["label"])], dtype=torch.float32)
        return Data(
            x=x,
            node_type_id=node_type_id,
            edge_index=edge_index,
            y=y,
            row_idx=torch.tensor([row_idx], dtype=torch.long),
        )


def load_inputs(graph_index_path: Path, vocab_path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    if not graph_index_path.exists():
        raise FileNotFoundError(f"Missing AST graph index: {graph_index_path}. Run scripts/extract_promise_ast.py first.")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing AST node type vocabulary: {vocab_path}. Run scripts/extract_promise_ast.py first.")

    graph_index = pd.read_csv(graph_index_path)
    required = {"graph_id", "dataset_name", "name", "source_path", "label", "feature_dim", "x_npy", "node_type_id_npy", "edge_index_npy"}
    missing = sorted(required - set(graph_index.columns))
    if missing:
        raise ValueError(f"AST graph index is missing required columns: {missing}")
    if graph_index.empty:
        raise ValueError("AST graph index is empty")

    labels = set(graph_index["label"].dropna().astype(int).unique())
    if labels - {0, 1}:
        raise ValueError(f"AST labels must be binary, found: {sorted(labels)}")

    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    from thesis_project.features.ast_schema import FEATURE_DIM, NODE_TYPES
    dims = set(graph_index["feature_dim"].astype(int))
    if dims != {FEATURE_DIM}:
        raise ValueError("Unsupported AST features. Run scripts/extract_promise_ast.py.")
    if vocab != {name: i for i, name in enumerate(NODE_TYPES)}:
        raise ValueError("AST node vocabulary does not match the encoder")
    return graph_index.reset_index(drop=True), {str(key): int(value) for key, value in vocab.items()}


def make_loader(dataset: ASTGraphDataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def move_batch(batch: Data, device: torch.device) -> Data:
    return batch.to(device)


def metric_dict(logits: Tensor, targets: Tensor) -> dict[str, float | None]:
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    y_true = targets.detach().cpu().numpy().astype(int)
    y_pred = (probs >= 0.5).astype(int)

    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(set(y_true.tolist())) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probs))
        metrics["pr_auc"] = float(average_precision_score(y_true, probs))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
    return metrics


def run_epoch(
    model: ASTGraphClassifier,
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
        logits = model(batch.x, batch.node_type_id, batch.edge_index, batch.batch)
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
    all_logits = torch.cat(logits_list)
    all_targets = torch.cat(targets_list)
    return total_loss / total_graphs, metric_dict(all_logits, all_targets)


def train_model(
    model: ASTGraphClassifier,
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
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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
            val_loss = None
            score = train_loss

        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            + (f"val_loss={val_loss:.4f} " if val_loss is not None else "")
            + f"train_f1={train_metrics['f1']:.4f}"
        )

        if score < best_score - args.min_delta:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if val_loader is not None and epochs_without_improvement >= args.patience:
            print(f"early_stopping epoch={epoch} best_epoch={best_epoch} best_val_loss={best_score:.4f}")
            break

    best_info = {
        "best_epoch": best_epoch,
        "best_loss": best_score,
        "pos_weight": pos_weight_value,
        "train_positives": positives,
        "train_negatives": negatives,
    }
    return best_state, history, best_info


def extract_embeddings(
    model: ASTGraphClassifier,
    dataset: ASTGraphDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    output_dim: int,
) -> np.ndarray:
    loader = make_loader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    embeddings = np.zeros((len(dataset.graph_index), output_dim), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        total_batches = len(loader)
        for batch_number, batch in enumerate(loader, start=1):
            batch = move_batch(batch, device)
            graph_embeddings = model.encode(batch.x, batch.node_type_id, batch.edge_index, batch.batch)
            row_indices = batch.row_idx.view(-1).detach().cpu().numpy()
            embeddings[row_indices] = graph_embeddings.detach().cpu().numpy().astype(np.float32)
            if batch_number == 1 or batch_number % 25 == 0 or batch_number == total_batches:
                print(f"ast_encoding batch={batch_number}/{total_batches}", flush=True)
    return embeddings


