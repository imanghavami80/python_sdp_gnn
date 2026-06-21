#!/usr/bin/env python3
"""Run Leave-One-Project-Out CFG encoder evaluation without placeholder CFGs.

The CFG encoder is evaluated as a cross-project SDP model. Each fold trains on
all projects except one held-out project, uses a validation split only inside the
training projects for early stopping, and tests on the held-out project.

Placeholder CFG graphs are always excluded because they only contain a trivial
ENTRY -> EXIT structure and do not carry useful control-flow information.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

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

STRUCTURAL_FEATURE_NAMES = ["line_position", "has_source_line", "is_synthetic"]


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
        edge_index = torch.from_numpy(np.load(row["edge_index_npy"])).long()
        edge_type = torch.from_numpy(np.load(row["edge_type_npy"])).long()
        y = torch.tensor([float(row["label"])], dtype=torch.float32)
        return Data(
            x=x,
            node_type_id=node_type_id,
            edge_index=edge_index,
            edge_type=edge_type,
            y=y,
            row_idx=torch.tensor([row_idx], dtype=torch.long),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LOPO CFG edge-aware GAT evaluation without placeholder CFGs.")
    parser.add_argument("--graph-index", type=Path, default=Path("outputs/promise/cfg/graph_index.csv"))
    parser.add_argument("--node-type-vocab", type=Path, default=Path("outputs/promise/cfg/node_type_vocab.json"))
    parser.add_argument("--edge-type-vocab", type=Path, default=Path("outputs/promise/cfg/edge_type_vocab.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/promise/embeddings/cfg_lopo"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--output-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--node-type-embedding-dim", type=int, default=32)
    parser.add_argument("--edge-type-embedding-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--attention-dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not clear existing CFG LOPO output files before writing new outputs.",
    )
    parser.add_argument(
        "--no-normalize-structural-features",
        action="store_true",
        help="Use raw CFG structural features instead of clamping them to expected ranges.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("--device cuda was requested, but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_output_dir(output_dir: Path, clean: bool) -> None:
    if clean:
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)


def load_vocab(path: Path, label: str) -> dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CFG {label} vocabulary: {path}. Run scripts/extract_promise_cfg.py first.")
    vocab = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): int(value) for key, value in vocab.items()}


def load_inputs(
    graph_index_path: Path,
    node_vocab_path: Path,
    edge_vocab_path: Path,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, int], dict[str, int]]:
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
    edge_vocab = load_vocab(edge_vocab_path, "edge-type")
    return graph_index.reset_index(drop=True), node_vocab, edge_vocab, original_modes


def build_model(args: argparse.Namespace, node_vocab: dict[str, int], edge_vocab: dict[str, int], device: torch.device) -> tuple[CFGGraphClassifier, CFGEncoderConfig]:
    config = CFGEncoderConfig(
        num_node_types=len(node_vocab),
        num_edge_types=len(edge_vocab),
        structural_feature_dim=3,
        node_type_embedding_dim=args.node_type_embedding_dim,
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


def split_train_val(
    graph_index: pd.DataFrame,
    train_project_indices: np.ndarray,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if val_ratio <= 0.0:
        return train_project_indices, np.array([], dtype=np.int64)
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val-ratio must be in [0, 1)")

    labels = graph_index.iloc[train_project_indices]["label"].astype(int).to_numpy()
    stratify = labels if len(np.unique(labels)) == 2 and min(np.bincount(labels)) >= 2 else None
    train_idx, val_idx = train_test_split(
        train_project_indices,
        test_size=val_ratio,
        random_state=seed,
        stratify=stratify,
    )
    return np.array(train_idx, dtype=np.int64), np.array(val_idx, dtype=np.int64)


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


def baseline_metrics(train_labels: np.ndarray, test_labels: np.ndarray) -> tuple[int, dict[str, float | None]]:
    positives = int(train_labels.sum())
    negatives = int(len(train_labels) - positives)
    majority_label = 1 if positives >= negatives else 0
    y_prob = np.full(len(test_labels), float(majority_label), dtype=np.float32)
    return majority_label, safe_metric_dict(test_labels.astype(int), y_prob)


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
        logits = model(batch.x, batch.node_type_id, batch.edge_index, batch.edge_type, batch.batch)
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
        if score < best_score - args.min_delta:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if val_loader is not None and epochs_without_improvement >= args.patience:
            break

    best_info = {
        "best_epoch": best_epoch,
        "best_loss": best_score,
        "pos_weight": pos_weight_value,
        "train_positives": positives,
        "train_negatives": negatives,
    }
    return best_state, history, best_info


def evaluate_model(
    model: CFGGraphClassifier,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float | None], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_graphs = 0
    logits_list: list[Tensor] = []
    targets_list: list[Tensor] = []
    row_idx_list: list[Tensor] = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            targets = batch.y.float().view(-1)
            logits = model(batch.x, batch.node_type_id, batch.edge_index, batch.edge_type, batch.batch)
            loss = loss_fn(logits, targets)
            batch_size = int(targets.numel())
            total_loss += float(loss.item()) * batch_size
            total_graphs += batch_size
            logits_list.append(logits.detach().cpu())
            targets_list.append(targets.detach().cpu())
            row_idx_list.append(batch.row_idx.view(-1).detach().cpu())

    logits = torch.cat(logits_list)
    targets = torch.cat(targets_list).numpy().astype(int)
    row_indices = torch.cat(row_idx_list).numpy().astype(int)
    probs = torch.sigmoid(logits).numpy()
    return total_loss / total_graphs, safe_metric_dict(targets, probs), probs, targets, row_indices


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
        for batch in loader:
            batch = move_batch(batch, device)
            embeddings = model.encode(batch.x, batch.node_type_id, batch.edge_index, batch.edge_type, batch.batch)
            rows.append(batch.row_idx.view(-1).detach().cpu().numpy().astype(int))
            vectors.append(embeddings.detach().cpu().numpy().astype(np.float32))
    if not rows:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, output_dim), dtype=np.float32)
    return np.concatenate(rows), np.concatenate(vectors, axis=0)


def prefixed_metrics(prefix: str, metrics: dict[str, float | None]) -> dict[str, float | None]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def weighted_mean(df: pd.DataFrame, column: str, weight_column: str = "test_graphs") -> float | None:
    valid = df[[column, weight_column]].dropna()
    if valid.empty:
        return None
    weights = valid[weight_column].astype(float).to_numpy()
    values = valid[column].astype(float).to_numpy()
    if float(weights.sum()) == 0.0:
        return None
    return float(np.average(values, weights=weights))


def aggregate_results(
    graph_index: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    all_predictions: pd.DataFrame,
    original_modes: dict[str, int],
) -> dict[str, Any]:
    model_pooled = safe_metric_dict(
        all_predictions["label"].astype(int).to_numpy(),
        all_predictions["model_probability"].astype(float).to_numpy(),
    )
    baseline_pooled = safe_metric_dict(
        all_predictions["label"].astype(int).to_numpy(),
        all_predictions["baseline_probability"].astype(float).to_numpy(),
    )
    aggregate = {
        "protocol": "LOPO",
        "placeholder_policy": "excluded",
        "original_extraction_modes": original_modes,
        "graphs": int(len(graph_index)),
        "soot_graphs": int((graph_index["extraction_mode"].astype(str) == "soot").sum()),
        "placeholder_graphs": int((graph_index["extraction_mode"].astype(str) == "placeholder").sum()),
        "excluded_placeholder_graphs": int(original_modes.get("placeholder", 0)),
        "clean": int((graph_index["label"].astype(int) == 0).sum()),
        "defective": int((graph_index["label"].astype(int) == 1).sum()),
        "defective_ratio": float(graph_index["label"].astype(int).mean()),
        "folds": int(len(fold_metrics)),
        "model_pooled": model_pooled,
        "baseline_pooled": baseline_pooled,
        "model_weighted_by_project": {
            key: weighted_mean(fold_metrics, f"model_{key}") for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
        },
        "baseline_weighted_by_project": {
            key: weighted_mean(fold_metrics, f"baseline_{key}") for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
        },
    }
    aggregate["pooled_delta"] = {
        key: (
            None
            if model_pooled.get(key) is None or baseline_pooled.get(key) is None
            else float(model_pooled[key] - baseline_pooled[key])
        )
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    }
    return aggregate


def run_lopo(
    args: argparse.Namespace,
    device: torch.device,
    graph_index: pd.DataFrame,
    node_vocab: dict[str, int],
    edge_vocab: dict[str, int],
    original_modes: dict[str, int],
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    labels = graph_index["label"].astype(int).to_numpy()
    normalize_structural_features = not args.no_normalize_structural_features
    folds_dir = output_dir / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)

    fold_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    projects = sorted(graph_index["dataset_name"].astype(str).unique())

    for fold_idx, test_project in enumerate(projects):
        set_seed(args.seed + fold_idx)
        test_indices = graph_index.index[graph_index["dataset_name"].astype(str) == test_project].to_numpy(dtype=np.int64)
        train_project_indices = graph_index.index[graph_index["dataset_name"].astype(str) != test_project].to_numpy(dtype=np.int64)
        train_indices, val_indices = split_train_val(graph_index, train_project_indices, args.val_ratio, args.seed + fold_idx)

        train_dataset = CFGGraphDataset(graph_index, train_indices, normalize_structural_features=normalize_structural_features)
        val_dataset = CFGGraphDataset(graph_index, val_indices, normalize_structural_features=normalize_structural_features)
        test_dataset = CFGGraphDataset(graph_index, test_indices, normalize_structural_features=normalize_structural_features)
        train_loader = make_loader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = make_loader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers) if len(val_indices) else None
        test_loader = make_loader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

        model, encoder_config = build_model(args, node_vocab, edge_vocab, device)
        best_state, history, best_info = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            labels=labels,
            train_indices=train_indices,
            args=args,
            device=device,
        )
        model.load_state_dict(best_state)
        model.to(device)
        test_loss, model_metrics, probs, targets, row_indices = evaluate_model(
            model=model,
            loader=test_loader,
            loss_fn=make_loss_fn(labels, train_indices, device),
            device=device,
        )
        majority_label, base_metrics = baseline_metrics(labels[train_indices], targets)
        embedding_rows, test_embeddings = extract_test_embeddings(model, test_loader, device, args.output_dim)

        fold_dir = folds_dir / str(test_project)
        fold_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(fold_dir / "training_history.csv", index=False)
        np.save(fold_dir / "test_embeddings.npy", test_embeddings)
        torch.save(
            {
                "model_state_dict": model.cpu().state_dict(),
                "encoder_config": asdict(encoder_config),
                "node_type_vocab": node_vocab,
                "edge_type_vocab": edge_vocab,
                "best_info": best_info,
                "protocol": "LOPO",
                "placeholder_policy": "excluded",
                "test_project": test_project,
                "script_args": vars(args),
            },
            fold_dir / "cfg_encoder.pt",
        )

        test_frame = graph_index.iloc[row_indices][
            ["graph_id", "dataset_name", "name", "source_path", "label", "extraction_mode", "num_nodes", "num_edges", "num_methods"]
        ].copy()
        test_frame["model_probability"] = probs
        test_frame["model_prediction"] = (probs >= 0.5).astype(int)
        test_frame["baseline_label"] = majority_label
        test_frame["baseline_probability"] = float(majority_label)
        test_frame.to_csv(fold_dir / "test_predictions.csv", index=False)
        graph_index.iloc[embedding_rows][["graph_id", "dataset_name", "name", "source_path", "label", "extraction_mode"]].to_csv(
            fold_dir / "test_embedding_index.csv",
            index=False,
        )
        split_payload = {
            "test_project": test_project,
            "train_indices": train_indices.astype(int).tolist(),
            "val_indices": val_indices.astype(int).tolist(),
            "test_indices": test_indices.astype(int).tolist(),
        }
        (fold_dir / "split.json").write_text(json.dumps(split_payload, indent=2), encoding="utf-8")
        prediction_frames.append(test_frame)

        train_project_labels = labels[train_indices]
        val_project_labels = labels[val_indices] if len(val_indices) else np.array([], dtype=int)
        test_labels = labels[test_indices]
        row: dict[str, Any] = {
            "test_project": test_project,
            "train_graphs": int(len(train_indices)),
            "val_graphs": int(len(val_indices)),
            "test_graphs": int(len(test_indices)),
            "train_defective": int(train_project_labels.sum()),
            "val_defective": int(val_project_labels.sum()) if len(val_project_labels) else 0,
            "test_defective": int(test_labels.sum()),
            "test_clean": int(len(test_labels) - test_labels.sum()),
            "test_defective_ratio": float(test_labels.mean()),
            "train_placeholders": 0,
            "val_placeholders": 0,
            "test_placeholders": 0,
            "best_epoch": int(best_info["best_epoch"]),
            "best_val_loss": float(best_info["best_loss"]),
            "test_loss": float(test_loss),
            "majority_label": int(majority_label),
            **prefixed_metrics("model", model_metrics),
            **prefixed_metrics("baseline", base_metrics),
        }
        row["delta_accuracy"] = row["model_accuracy"] - row["baseline_accuracy"]
        row["delta_f1"] = row["model_f1"] - row["baseline_f1"]
        fold_rows.append(row)
        print(
            f"fold={test_project} test={len(test_indices)} best_epoch={row['best_epoch']} "
            f"acc={row['model_accuracy']:.3f} f1={row['model_f1']:.3f} baseline_f1={row['baseline_f1']:.3f}",
            flush=True,
        )

    fold_metrics = pd.DataFrame(fold_rows)
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    aggregate = aggregate_results(graph_index, fold_metrics, all_predictions, original_modes)
    aggregate.update(
        {
            "graph_index": str(resolve_path(args.graph_index)),
            "output_dir": str(output_dir),
            "structural_feature_names": STRUCTURAL_FEATURE_NAMES,
            "structural_feature_transform": {
                "enabled": not args.no_normalize_structural_features,
                "line_position": "clamped to [0, 1]",
                "has_source_line": "binary flag clamped to [0, 1]",
                "is_synthetic": "binary flag clamped to [0, 1]",
            },
            "edge_type_handling": {
                "method": "trainable edge-type embeddings passed to GATv2 attention",
                "edge_types": edge_vocab,
                "add_self_loops": False,
            },
            "encoder_config": asdict(build_model(args, node_vocab, edge_vocab, torch.device("cpu"))[1]),
        }
    )
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    all_predictions.to_csv(output_dir / "all_test_predictions.csv", index=False)
    (output_dir / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    return fold_metrics, aggregate


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    output_dir = resolve_path(args.output_dir)
    prepare_output_dir(output_dir, clean=not args.no_clean)
    device = choose_device(args.device)
    graph_index, node_vocab, edge_vocab, original_modes = load_inputs(
        graph_index_path=resolve_path(args.graph_index),
        node_vocab_path=resolve_path(args.node_type_vocab),
        edge_vocab_path=resolve_path(args.edge_type_vocab),
    )

    print(
        "CFG LOPO evaluation started. "
        f"graphs={len(graph_index)} excluded_placeholders={original_modes.get('placeholder', 0)} "
        f"device={device} output_dir={output_dir}",
        flush=True,
    )
    run_lopo(
        args=args,
        device=device,
        graph_index=graph_index,
        node_vocab=node_vocab,
        edge_vocab=edge_vocab,
        original_modes=original_modes,
        output_dir=output_dir,
    )
    print(f"fold_metrics={output_dir / 'fold_metrics.csv'}", flush=True)
    print("CFG LOPO evaluation finished.", flush=True)


if __name__ == "__main__":
    main()
