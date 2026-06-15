#!/usr/bin/env python3
"""Train the AST GIN encoder and save one embedding per AST graph.

Default input:
    outputs/promise/ast/graph_index.csv

Default output:
    outputs/promise/embeddings/ast/

The script trains `ASTGINEncoder` as a supervised graph classifier using the AST
labels from the graph index. It then removes the classifier head conceptually by
saving the encoder output embedding for every AST graph.
"""

from __future__ import annotations

import argparse
import json
import random
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

    from thesis_project.models import ASTEncoderConfig, ASTGINEncoder, ASTGraphClassifier, normalize_ast_structural_features
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only in missing dependency environments.
    raise SystemExit(
        "Missing GNN dependencies. Install them with:\n"
        "  .venv/bin/pip install -r requirements.txt\n"
        f"Original error: {exc}"
    ) from exc

STRUCTURAL_FEATURE_NAMES = ["depth", "out_degree", "has_identifier"]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AST GIN encoder and save AST graph embeddings.")
    parser.add_argument("--graph-index", type=Path, default=Path("outputs/promise/ast/graph_index.csv"))
    parser.add_argument("--node-type-vocab", type=Path, default=Path("outputs/promise/ast/node_type_vocab.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/promise/embeddings/ast"))
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
    parser.add_argument("--node-type-embedding-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--no-normalize-structural-features",
        action="store_true",
        help="Use raw AST structural features instead of depth/max_depth and log1p(out_degree).",
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


def load_inputs(graph_index_path: Path, vocab_path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    if not graph_index_path.exists():
        raise FileNotFoundError(f"Missing AST graph index: {graph_index_path}. Run scripts/extract_promise_ast.py first.")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing AST node type vocabulary: {vocab_path}. Run scripts/extract_promise_ast.py first.")

    graph_index = pd.read_csv(graph_index_path)
    required = {"graph_id", "dataset_name", "name", "source_path", "label", "x_npy", "node_type_id_npy", "edge_index_npy"}
    missing = sorted(required - set(graph_index.columns))
    if missing:
        raise ValueError(f"AST graph index is missing required columns: {missing}")
    if graph_index.empty:
        raise ValueError("AST graph index is empty")

    labels = set(graph_index["label"].dropna().astype(int).unique())
    if labels - {0, 1}:
        raise ValueError(f"AST labels must be binary, found: {sorted(labels)}")

    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    return graph_index.reset_index(drop=True), {str(key): int(value) for key, value in vocab.items()}


def split_indices(labels: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    all_indices = np.arange(len(labels))
    if val_ratio <= 0.0:
        return all_indices, np.array([], dtype=np.int64)
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val-ratio must be in [0, 1)")
    train_idx, val_idx = train_test_split(
        all_indices,
        test_size=val_ratio,
        random_state=seed,
        stratify=labels,
    )
    return np.array(train_idx, dtype=np.int64), np.array(val_idx, dtype=np.int64)


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
        for batch in loader:
            batch = move_batch(batch, device)
            graph_embeddings = model.encode(batch.x, batch.node_type_id, batch.edge_index, batch.batch)
            row_indices = batch.row_idx.view(-1).detach().cpu().numpy()
            embeddings[row_indices] = graph_embeddings.detach().cpu().numpy().astype(np.float32)
    return embeddings


def save_outputs(
    output_dir: Path,
    graph_index: pd.DataFrame,
    embeddings: np.ndarray,
    model: ASTGraphClassifier,
    encoder_config: ASTEncoderConfig,
    vocab: dict[str, int],
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    history: list[dict[str, Any]],
    best_info: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "ast_embeddings.npy"
    index_path = output_dir / "ast_embedding_index.csv"
    history_path = output_dir / "ast_training_history.csv"
    checkpoint_path = output_dir / "ast_encoder.pt"
    summary_path = output_dir / "ast_embedding_summary.json"
    split_path = output_dir / "ast_train_val_split.json"

    np.save(embeddings_path, embeddings)
    embedding_index = graph_index[
        ["graph_id", "dataset_name", "name", "source_path", "label", "num_nodes", "num_edges", "parser_mode"]
    ].copy()
    embedding_index.insert(0, "embedding_row", np.arange(len(embedding_index), dtype=np.int64))
    embedding_index["embedding_npy"] = str(embeddings_path)
    embedding_index.to_csv(index_path, index=False)
    pd.DataFrame(history).to_csv(history_path, index=False)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "encoder_config": asdict(encoder_config),
            "node_type_vocab": vocab,
            "best_info": best_info,
            "script_args": vars(args),
        },
        checkpoint_path,
    )
    split_path.write_text(
        json.dumps(
            {
                "train_indices": train_indices.astype(int).tolist(),
                "val_indices": val_indices.astype(int).tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        "graph_index": str(resolve_path(args.graph_index)),
        "output_dir": str(output_dir),
        "embeddings_npy": str(embeddings_path),
        "embedding_index_csv": str(index_path),
        "checkpoint": str(checkpoint_path),
        "training_history_csv": str(history_path),
        "num_graphs": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "num_node_types": int(len(vocab)),
        "structural_feature_names": STRUCTURAL_FEATURE_NAMES,
        "structural_feature_transform": {
            "enabled": not args.no_normalize_structural_features,
            "depth": "depth / max_depth_in_graph",
            "out_degree": "log1p(out_degree)",
            "has_identifier": "binary flag unchanged",
        },
        "train_graphs": int(len(train_indices)),
        "val_graphs": int(len(val_indices)),
        "label_counts": {str(key): int(value) for key, value in graph_index["label"].value_counts().sort_index().items()},
        "datasets": graph_index["dataset_name"].value_counts().sort_index().astype(int).to_dict(),
        "encoder_config": asdict(encoder_config),
        "best_info": best_info,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"embeddings={embeddings_path}")
    print(f"embedding_index={index_path}")
    print(f"checkpoint={checkpoint_path}")
    print(f"summary={summary_path}")


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive because embeddings should come from a trained encoder")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    set_seed(args.seed)
    graph_index_path = resolve_path(args.graph_index)
    vocab_path = resolve_path(args.node_type_vocab)
    output_dir = resolve_path(args.output_dir)
    graph_index, vocab = load_inputs(graph_index_path, vocab_path)
    labels = graph_index["label"].astype(int).to_numpy()
    train_indices, val_indices = split_indices(labels, args.val_ratio, args.seed)

    encoder_config = ASTEncoderConfig(
        num_node_types=len(vocab),
        structural_feature_dim=3,
        node_type_embedding_dim=args.node_type_embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_batch_norm=True,
        bidirectional_edges=True,
    )
    device = choose_device(args.device)
    encoder = ASTGINEncoder(encoder_config)
    model = ASTGraphClassifier(encoder, dropout=args.dropout).to(device)

    normalize_structural_features = not args.no_normalize_structural_features
    train_dataset = ASTGraphDataset(graph_index, train_indices, normalize_structural_features=normalize_structural_features)
    val_dataset = (
        ASTGraphDataset(graph_index, val_indices, normalize_structural_features=normalize_structural_features)
        if len(val_indices)
        else None
    )
    full_dataset = ASTGraphDataset(graph_index, normalize_structural_features=normalize_structural_features)
    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers) if val_dataset else None

    print(
        "AST embedding training started. "
        f"graphs={len(graph_index)} train={len(train_indices)} val={len(val_indices)} "
        f"device={device} embedding_dim={args.output_dim} "
        f"normalize_structural_features={normalize_structural_features}"
    )
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
    embeddings = extract_embeddings(
        model=model,
        dataset=full_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        output_dim=args.output_dim,
    )
    save_outputs(
        output_dir=output_dir,
        graph_index=graph_index,
        embeddings=embeddings,
        model=model.cpu(),
        encoder_config=encoder_config,
        vocab=vocab,
        train_indices=train_indices,
        val_indices=val_indices,
        history=history,
        best_info=best_info,
        args=args,
    )
    print("AST embedding generation finished.")


if __name__ == "__main__":
    main()
