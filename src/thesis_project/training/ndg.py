"""Shared data and training operations for node-level NDG models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import Tensor, nn

from thesis_project.models import NDGEncoderConfig, NDGMultiViewRelationalGATEncoder, NDGNodeClassifier


@dataclass
class ProjectGraph:
    """One project-level NDG with file-node views and binary labels."""

    dataset_name: str
    names: list[str]
    source_paths: list[str]
    metrics_x: Tensor
    ast_x: Tensor
    cfg_x: Tensor
    view_mask: Tensor
    y: Tensor
    edge_index: Tensor
    edge_type: Tensor

    @property
    def num_nodes(self) -> int:
        return int(self.y.numel())

    def to(self, device: torch.device) -> ProjectGraph:
        return ProjectGraph(
            dataset_name=self.dataset_name,
            names=self.names,
            source_paths=self.source_paths,
            metrics_x=self.metrics_x.to(device),
            ast_x=self.ast_x.to(device),
            cfg_x=self.cfg_x.to(device),
            view_mask=self.view_mask.to(device),
            y=self.y.to(device),
            edge_index=self.edge_index.to(device),
            edge_type=self.edge_type.to(device),
        )


def add_inverse_relations(
    edge_index: np.ndarray,
    edge_type: np.ndarray,
    num_forward_relations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Add separately typed reverse edges without changing extracted tensors."""
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("NDG edge_index must have shape [2, num_edges]")
    if edge_index.shape[1] != edge_type.shape[0]:
        raise ValueError("NDG edge_index and edge_type sizes do not match")
    if num_forward_relations <= 0:
        raise ValueError("num_forward_relations must be positive")
    reverse_index = edge_index[[1, 0], :]
    reverse_type = edge_type + num_forward_relations
    return (
        np.concatenate([edge_index, reverse_index], axis=1).astype(np.int64),
        np.concatenate([edge_type, reverse_type], axis=0).astype(np.int64),
    )


def combine_graphs(graphs: list[ProjectGraph]) -> ProjectGraph:
    """Combine project graphs as disconnected components for full-batch training."""
    if not graphs:
        raise ValueError("Cannot combine an empty project list")
    node_offset = 0
    edge_indices: list[Tensor] = []
    for graph in graphs:
        edge_indices.append(graph.edge_index + node_offset)
        node_offset += graph.num_nodes
    return ProjectGraph(
        dataset_name="+".join(graph.dataset_name for graph in graphs),
        names=[name for graph in graphs for name in graph.names],
        source_paths=[path for graph in graphs for path in graph.source_paths],
        metrics_x=torch.cat([graph.metrics_x for graph in graphs]),
        ast_x=torch.cat([graph.ast_x for graph in graphs]),
        cfg_x=torch.cat([graph.cfg_x for graph in graphs]),
        view_mask=torch.cat([graph.view_mask for graph in graphs]),
        y=torch.cat([graph.y for graph in graphs]),
        edge_index=torch.cat(edge_indices, dim=1),
        edge_type=torch.cat([graph.edge_type for graph in graphs]),
    )


def standardize_metrics(train_graph: ProjectGraph, *other_graphs: ProjectGraph) -> tuple[ProjectGraph, ...]:
    """Fit median imputation and standard scaling on training nodes only."""
    medians = torch.nanmedian(train_graph.metrics_x, dim=0).values
    if not torch.isfinite(medians).all():
        raise ValueError("At least one metric is missing for every outer-training node")

    def impute(x: Tensor) -> Tensor:
        return torch.where(torch.isnan(x), medians.unsqueeze(0), x)

    imputed_train = impute(train_graph.metrics_x)
    mean = imputed_train.mean(dim=0)
    std = imputed_train.std(dim=0, unbiased=False).clamp_min(1e-6)

    def transform(graph: ProjectGraph) -> ProjectGraph:
        metrics_x = (impute(graph.metrics_x) - mean) / std
        if not torch.isfinite(metrics_x).all():
            raise ValueError(f"Non-finite metrics remain after transforming {graph.dataset_name}")
        return ProjectGraph(
            dataset_name=graph.dataset_name,
            names=graph.names,
            source_paths=graph.source_paths,
            metrics_x=metrics_x,
            ast_x=graph.ast_x,
            cfg_x=graph.cfg_x,
            view_mask=graph.view_mask,
            y=graph.y,
            edge_index=graph.edge_index,
            edge_type=graph.edge_type,
        )

    return tuple(transform(graph) for graph in (train_graph, *other_graphs))


def model_inputs(graph: ProjectGraph) -> dict[str, Tensor]:
    return {
        "metrics_x": graph.metrics_x,
        "ast_x": graph.ast_x,
        "cfg_x": graph.cfg_x,
        "view_mask": graph.view_mask,
        "edge_index": graph.edge_index,
        "edge_type": graph.edge_type,
    }


def binary_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float | None]:
    """Calculate fixed-threshold and threshold-independent binary metrics."""
    predictions = (probabilities >= 0.5).astype(np.int64)
    result: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }
    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, probabilities))
        result["pr_auc"] = float(average_precision_score(y_true, probabilities))
    else:
        result["roc_auc"] = None
        result["pr_auc"] = None
    return result


def make_model(config: NDGEncoderConfig, device: torch.device) -> NDGNodeClassifier:
    return NDGNodeClassifier(NDGMultiViewRelationalGATEncoder(config), dropout=config.dropout).to(device)


def loss_function(labels: Tensor) -> nn.BCEWithLogitsLoss:
    positives = float(labels.sum().item())
    negatives = float(labels.numel() - positives)
    weight = negatives / positives if positives else 1.0
    return nn.BCEWithLogitsLoss(pos_weight=labels.new_tensor(weight))


def train_with_validation(
    model: NDGNodeClassifier,
    train_graph: ProjectGraph,
    validation_graph: ProjectGraph,
    args: Any,
) -> tuple[int, list[dict[str, Any]]]:
    """Select an epoch using one untouched inner-validation project."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = loss_function(train_graph.y)
    best_epoch = 1
    best_loss = float("inf")
    stale_epochs = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_logits = model(**model_inputs(train_graph))
        train_loss = loss_fn(train_logits, train_graph.y)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_logits = model(**model_inputs(validation_graph))
            validation_loss = loss_fn(validation_logits, validation_graph.y)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss.item()),
                "validation_loss": float(validation_loss.item()),
            }
        )
        if validation_loss.item() < best_loss - args.min_delta:
            best_loss = float(validation_loss.item())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            break
    return best_epoch, history


def retrain(model: NDGNodeClassifier, train_graph: ProjectGraph, epochs: int, args: Any) -> None:
    """Retrain a fresh model on all outer-training project graphs."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = loss_function(train_graph.y)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(**model_inputs(train_graph)), train_graph.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()


def evaluate(model: NDGNodeClassifier, graph: ProjectGraph) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one embedding, probability, and label per file node."""
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(**model_inputs(graph))
        logits = model.classifier(embeddings).view(-1)
    probabilities = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
    return embeddings.cpu().numpy().astype(np.float32), probabilities, graph.y.cpu().numpy().astype(np.int64)
