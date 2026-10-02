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
    y: Tensor
    edge_index: Tensor
    edge_type: Tensor
    ndg_structural_x: Tensor | None = None

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
            ndg_structural_x=(
                None if self.ndg_structural_x is None else self.ndg_structural_x.to(device)
            ),
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


def model_inputs(graph: ProjectGraph) -> dict[str, Tensor]:
    ndg_structural_x = graph.ndg_structural_x
    if ndg_structural_x is None:
        ndg_structural_x = graph.metrics_x.new_empty((graph.num_nodes, 0))
    return {
        "metrics_x": graph.metrics_x,
        "ndg_structural_x": ndg_structural_x,
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


def select_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Select a deterministic F1 threshold using validation labels only."""
    candidates = np.unique(np.concatenate([probabilities.astype(np.float64), np.asarray([0.5])]))
    scores = np.asarray(
        [f1_score(labels, probabilities >= threshold, zero_division=0) for threshold in candidates]
    )
    best_score = scores.max()
    tied = candidates[np.isclose(scores, best_score)]
    return float(tied[np.argmin(np.abs(tied - 0.5))])


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
