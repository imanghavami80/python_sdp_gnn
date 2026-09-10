"""Shared data and training operations for node-level NDG models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
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
    loss_weight: Tensor
    y: Tensor
    edge_index: Tensor
    edge_type: Tensor
    cluster_x: Tensor | None = None

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
            loss_weight=self.loss_weight.to(device),
            y=self.y.to(device),
            edge_index=self.edge_index.to(device),
            edge_type=self.edge_type.to(device),
            cluster_x=None if self.cluster_x is None else self.cluster_x.to(device),
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
    loss_weights: list[Tensor] = []
    cluster_dims = {0 if graph.cluster_x is None else int(graph.cluster_x.size(1)) for graph in graphs}
    if len(cluster_dims) != 1:
        raise ValueError("All combined projects must have the same cluster feature dimension")
    cluster_dim = cluster_dims.pop()
    for graph in graphs:
        edge_indices.append(graph.edge_index + node_offset)
        # Equalize total loss contribution across differently sized projects.
        loss_weights.append(
            torch.full(
                (graph.num_nodes,),
                1.0 / graph.num_nodes,
                dtype=torch.float32,
                device=graph.y.device,
            )
        )
        node_offset += graph.num_nodes
    return ProjectGraph(
        dataset_name="+".join(graph.dataset_name for graph in graphs),
        names=[name for graph in graphs for name in graph.names],
        source_paths=[path for graph in graphs for path in graph.source_paths],
        metrics_x=torch.cat([graph.metrics_x for graph in graphs]),
        ast_x=torch.cat([graph.ast_x for graph in graphs]),
        cfg_x=torch.cat([graph.cfg_x for graph in graphs]),
        view_mask=torch.cat([graph.view_mask for graph in graphs]),
        loss_weight=torch.cat(loss_weights),
        y=torch.cat([graph.y for graph in graphs]),
        edge_index=torch.cat(edge_indices, dim=1),
        edge_type=torch.cat([graph.edge_type for graph in graphs]),
        cluster_x=(
            torch.cat([graph.cluster_x for graph in graphs if graph.cluster_x is not None])
            if cluster_dim
            else None
        ),
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
            loss_weight=graph.loss_weight,
            y=graph.y,
            edge_index=graph.edge_index,
            edge_type=graph.edge_type,
            cluster_x=graph.cluster_x,
        )

    return tuple(transform(graph) for graph in (train_graph, *other_graphs))


def model_inputs(graph: ProjectGraph) -> dict[str, Tensor]:
    cluster_x = graph.cluster_x
    if cluster_x is None:
        cluster_x = graph.metrics_x.new_empty((graph.num_nodes, 0))
    return {
        "metrics_x": graph.metrics_x,
        "cluster_x": cluster_x,
        "ast_x": graph.ast_x,
        "cfg_x": graph.cfg_x,
        "view_mask": graph.view_mask,
        "edge_index": graph.edge_index,
        "edge_type": graph.edge_type,
    }


def binary_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
    predictions: np.ndarray | None = None,
) -> dict[str, float | None]:
    """Calculate threshold-dependent, ranking, and calibration metrics."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if predictions is None:
        predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    sensitivity = float(tp / (tp + fn)) if tp + fn else 0.0
    result: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, predictions)),
        "g_mean": float(np.sqrt(sensitivity * specificity)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
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


def loss_function(labels: Tensor, sample_weights: Tensor) -> nn.BCEWithLogitsLoss:
    weights = sample_weights.to(dtype=labels.dtype)
    positives = float((weights * labels).sum().item())
    negatives = float((weights * (1.0 - labels)).sum().item())
    weight = negatives / positives if positives else 1.0
    return nn.BCEWithLogitsLoss(pos_weight=labels.new_tensor(weight), reduction="none")


def weighted_loss(loss_fn: nn.BCEWithLogitsLoss, logits: Tensor, graph: ProjectGraph) -> Tensor:
    losses = loss_fn(logits, graph.y)
    weights = graph.loss_weight.to(dtype=losses.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1e-12)


def select_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Select a deterministic F1 threshold using validation labels only."""
    candidates = np.unique(np.concatenate([probabilities.astype(np.float64), np.asarray([0.5])]))
    scores = np.asarray(
        [f1_score(labels, probabilities >= threshold, zero_division=0) for threshold in candidates]
    )
    best_score = scores.max()
    tied = candidates[np.isclose(scores, best_score)]
    return float(tied[np.argmin(np.abs(tied - 0.5))])


def train_with_validation(
    model: NDGNodeClassifier,
    train_graph: ProjectGraph,
    validation_graph: ProjectGraph,
    args: Any,
) -> tuple[int, float, list[dict[str, Any]]]:
    """Select an epoch using one untouched inner-validation project."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = loss_function(train_graph.y, train_graph.loss_weight)
    best_epoch = 1
    best_loss = float("inf")
    best_state: dict[str, Tensor] | None = None
    stale_epochs = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_logits = model(**model_inputs(train_graph))
        train_loss = weighted_loss(loss_fn, train_logits, train_graph)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_logits = model(**model_inputs(validation_graph))
            validation_loss = weighted_loss(loss_fn, validation_logits, validation_graph)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss.item()),
                "validation_loss": float(validation_loss.item()),
            }
        )
        print(
            f"ndg_epoch={epoch:03d} train_loss={train_loss.item():.4f} "
            f"val_loss={validation_loss.item():.4f}",
            flush=True,
        )
        if validation_loss.item() < best_loss - args.min_delta:
            best_loss = float(validation_loss.item())
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("NDG epoch selection did not produce a checkpoint")
    model.load_state_dict(best_state)
    _, validation_probabilities, validation_labels = evaluate(model, validation_graph)
    threshold = select_f1_threshold(validation_labels, validation_probabilities)
    return best_epoch, threshold, history


def retrain(model: NDGNodeClassifier, train_graph: ProjectGraph, epochs: int, args: Any) -> None:
    """Retrain a fresh model on all outer-training project graphs."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = loss_function(train_graph.y, train_graph.loss_weight)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = weighted_loss(loss_fn, model(**model_inputs(train_graph)), train_graph)
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


def evaluate_attention(model: NDGNodeClassifier, graph: ProjectGraph) -> dict[str, np.ndarray]:
    """Return detached encoder diagnostics for a fitted project graph."""
    model.eval()
    with torch.no_grad():
        _, attention = model.encoder(**model_inputs(graph), return_attention=True)
    return {key: value.cpu().numpy() for key, value in attention.items()}
