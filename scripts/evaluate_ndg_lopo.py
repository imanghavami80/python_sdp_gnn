#!/usr/bin/env python3
"""Train and evaluate the multi-view NDG encoder with project-level LOPO.

The script joins each NDG file node with its AST and CFG embedding by
``(dataset_name, name)``. One project graph is held out for testing, one of the
remaining projects selects the training epoch, and the final fold model is
retrained on all nine non-test projects for that selected number of epochs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import torch
    from torch import Tensor, nn

    from thesis_project.models import (
        NDGEncoderConfig,
        NDGMultiViewRelationalGATEncoder,
        NDGNodeClassifier,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing GNN dependencies. Install them with:\n"
        "  .venv/bin/pip install -r requirements.txt\n"
        f"Original error: {exc}"
    ) from exc


@dataclass
class ProjectGraph:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-view relational GAT NDG node classification with LOPO.")
    parser.add_argument("--ndg-index", type=Path, default=Path("outputs/promise/ndg/graph_index.csv"))
    parser.add_argument("--ndg-edge-vocab", type=Path, default=Path("outputs/promise/ndg/edge_type_vocab.json"))
    parser.add_argument("--ndg-feature-names", type=Path, default=Path("outputs/promise/ndg/feature_names.json"))
    parser.add_argument("--ast-index", type=Path, default=Path("outputs/promise/embeddings/ast_lopo/ast_embedding_index.csv"))
    parser.add_argument("--ast-summary", type=Path, default=Path("outputs/promise/embeddings/ast_lopo/ast_embedding_summary.json"))
    parser.add_argument("--cfg-index", type=Path, default=Path("outputs/promise/embeddings/cfg_lopo/cfg_embedding_index.csv"))
    parser.add_argument("--cfg-summary", type=Path, default=Path("outputs/promise/embeddings/cfg_lopo/cfg_embedding_summary.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/promise/embeddings/ndg_lopo"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--output-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--edge-type-embedding-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--attention-dropout", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument(
        "--allow-non-lopo-ast",
        action="store_true",
        help="Allow globally supervised AST embeddings. This makes the resulting LOPO estimate leakage-prone.",
    )
    parser.add_argument("--no-clean", action="store_true", help="Keep existing files in the output directory.")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


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


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def embedding_protocol(summary: dict[str, Any]) -> str:
    return str(summary.get("protocol", summary.get("embedding_protocol", "global_supervised_random_split")))


def validate_upstream_protocols(ast_summary: dict[str, Any], cfg_summary: dict[str, Any], allow_non_lopo_ast: bool) -> dict[str, str]:
    ast_protocol = embedding_protocol(ast_summary)
    cfg_protocol = embedding_protocol(cfg_summary)
    ast_is_lopo = "lopo" in ast_protocol.lower()
    cfg_is_lopo = "lopo" in cfg_protocol.lower()
    if not cfg_is_lopo:
        raise ValueError(f"CFG embeddings must use LOPO, found protocol={cfg_protocol!r}")
    if not ast_is_lopo and not allow_non_lopo_ast:
        raise ValueError(
            "AST embeddings are globally supervised with a random split, so held-out-project labels can leak into NDG "
            "features. Generate LOPO AST embeddings first, or use --allow-non-lopo-ast only for an exploratory run."
        )
    return {
        "ast": ast_protocol,
        "cfg": cfg_protocol,
        "ndg_result_validity": "leakage_prone_exploratory" if not ast_is_lopo else "lopo_cross_fitted_upstream_not_nested",
    }


def load_embedding_table(index_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    index = pd.read_csv(index_path)
    required = {"embedding_row", "dataset_name", "name", "embedding_npy"}
    missing = sorted(required - set(index.columns))
    if missing:
        raise ValueError(f"Embedding index {index_path} is missing columns: {missing}")
    embedding_paths = index["embedding_npy"].dropna().astype(str).unique().tolist()
    if len(embedding_paths) != 1:
        raise ValueError(f"Expected one embedding matrix in {index_path}, found {embedding_paths}")
    embeddings = np.load(embedding_paths[0]).astype(np.float32)
    rows = index["embedding_row"].to_numpy(dtype=np.int64)
    if len(index) != len(embeddings) or set(rows.tolist()) != set(range(len(embeddings))):
        raise ValueError(f"Embedding row contract is inconsistent in {index_path}")
    index = index.copy()
    index["join_key"] = list(zip(index["dataset_name"].astype(str), index["name"].astype(str), strict=True))
    if index["join_key"].duplicated().any():
        raise ValueError(f"Duplicate (dataset_name, name) keys in {index_path}")
    return index, embeddings


def embedding_lookup(index: pd.DataFrame, embeddings: np.ndarray) -> dict[tuple[str, str], np.ndarray]:
    return {
        key: embeddings[int(row)]
        for key, row in zip(index["join_key"], index["embedding_row"], strict=True)
    }


def add_inverse_relations(edge_index: np.ndarray, edge_type: np.ndarray, num_forward_relations: int) -> tuple[np.ndarray, np.ndarray]:
    if edge_index.shape[1] != edge_type.shape[0]:
        raise ValueError("NDG edge_index and edge_type sizes do not match")
    reverse_index = edge_index[[1, 0], :]
    reverse_type = edge_type + num_forward_relations
    return (
        np.concatenate([edge_index, reverse_index], axis=1).astype(np.int64),
        np.concatenate([edge_type, reverse_type], axis=0).astype(np.int64),
    )


def load_project_graphs(
    ndg_index_path: Path,
    ast_index_path: Path,
    cfg_index_path: Path,
    num_forward_relations: int,
) -> tuple[dict[str, ProjectGraph], dict[str, int]]:
    ndg_index = pd.read_csv(ndg_index_path)
    required = {"dataset_name", "graph_json", "x_npy", "y_npy", "edge_index_npy", "edge_type_npy"}
    missing = sorted(required - set(ndg_index.columns))
    if missing:
        raise ValueError(f"NDG index is missing columns: {missing}")

    ast_index, ast_embeddings = load_embedding_table(ast_index_path)
    cfg_index, cfg_embeddings = load_embedding_table(cfg_index_path)
    ast_lookup = embedding_lookup(ast_index, ast_embeddings)
    cfg_lookup = embedding_lookup(cfg_index, cfg_embeddings)
    ast_dim = int(ast_embeddings.shape[1])
    cfg_dim = int(cfg_embeddings.shape[1])
    graphs: dict[str, ProjectGraph] = {}
    missing_ast = 0
    missing_cfg = 0

    for _, row in ndg_index.sort_values("dataset_name").iterrows():
        dataset_name = str(row["dataset_name"])
        graph_json = load_json(Path(str(row["graph_json"])))
        nodes = sorted(graph_json["nodes"], key=lambda node: int(node["id"]))
        names = [str(node["name"]) for node in nodes]
        source_paths = [str(node["source_path"]) for node in nodes]
        metrics = np.load(row["x_npy"]).astype(np.float32)
        labels = np.load(row["y_npy"]).astype(np.float32)
        edge_index = np.load(row["edge_index_npy"]).astype(np.int64)
        edge_type = np.load(row["edge_type_npy"]).astype(np.int64)
        if len(nodes) != len(metrics) or len(nodes) != len(labels):
            raise ValueError(f"Node metadata/tensor mismatch for {dataset_name}")

        ast_x = np.zeros((len(nodes), ast_dim), dtype=np.float32)
        cfg_x = np.zeros((len(nodes), cfg_dim), dtype=np.float32)
        view_mask = np.zeros((len(nodes), 3), dtype=np.bool_)
        view_mask[:, 0] = True
        for node_id, name in enumerate(names):
            key = (dataset_name, name)
            if key in ast_lookup:
                ast_x[node_id] = ast_lookup[key]
                view_mask[node_id, 1] = True
            else:
                missing_ast += 1
            if key in cfg_lookup:
                cfg_x[node_id] = cfg_lookup[key]
                view_mask[node_id, 2] = True
            else:
                missing_cfg += 1

        augmented_edge_index, augmented_edge_type = add_inverse_relations(edge_index, edge_type, num_forward_relations)
        graphs[dataset_name] = ProjectGraph(
            dataset_name=dataset_name,
            names=names,
            source_paths=source_paths,
            metrics_x=torch.from_numpy(metrics),
            ast_x=torch.from_numpy(ast_x),
            cfg_x=torch.from_numpy(cfg_x),
            view_mask=torch.from_numpy(view_mask),
            y=torch.from_numpy(labels),
            edge_index=torch.from_numpy(augmented_edge_index),
            edge_type=torch.from_numpy(augmented_edge_type),
        )
    return graphs, {"missing_ast": missing_ast, "missing_cfg": missing_cfg, "ast_dim": ast_dim, "cfg_dim": cfg_dim}


def combine_graphs(graphs: list[ProjectGraph]) -> ProjectGraph:
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
    mean = train_graph.metrics_x.mean(dim=0)
    std = train_graph.metrics_x.std(dim=0, unbiased=False).clamp_min(1e-6)

    def transform(graph: ProjectGraph) -> ProjectGraph:
        return ProjectGraph(
            dataset_name=graph.dataset_name,
            names=graph.names,
            source_paths=graph.source_paths,
            metrics_x=(graph.metrics_x - mean) / std,
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
    args: argparse.Namespace,
) -> tuple[int, list[dict[str, Any]]]:
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
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss.item()),
            "validation_loss": float(validation_loss.item()),
        }
        history.append(row)
        if validation_loss.item() < best_loss - args.min_delta:
            best_loss = float(validation_loss.item())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            break
    return best_epoch, history


def retrain(model: NDGNodeClassifier, train_graph: ProjectGraph, epochs: int, args: argparse.Namespace) -> None:
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
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(**model_inputs(graph))
        logits = model.classifier(embeddings).view(-1)
    probabilities = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
    return embeddings.cpu().numpy().astype(np.float32), probabilities, graph.y.cpu().numpy().astype(np.int64)


def pooled_metric_summary(predictions: pd.DataFrame, probability_column: str) -> dict[str, float | None]:
    return binary_metrics(
        predictions["label"].to_numpy(dtype=np.int64),
        predictions[probability_column].to_numpy(dtype=np.float32),
    )


def run_lopo(
    graphs: dict[str, ProjectGraph],
    config: NDGEncoderConfig,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    provenance: dict[str, str],
    coverage: dict[str, int],
    edge_vocab: dict[str, int],
    feature_names: list[str],
) -> None:
    datasets = sorted(graphs)
    fold_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    embedding_frames: list[pd.DataFrame] = []
    embedding_arrays: list[np.ndarray] = []

    for fold_index, test_name in enumerate(datasets):
        candidate_train = [name for name in datasets if name != test_name]
        validation_name = candidate_train[fold_index % len(candidate_train)]
        fit_names = [name for name in candidate_train if name != validation_name]
        set_seed(args.seed + fold_index)

        fit_graph = combine_graphs([graphs[name] for name in fit_names])
        validation_graph = graphs[validation_name]
        fit_graph, validation_graph = standardize_metrics(fit_graph, validation_graph)
        selection_model = make_model(config, device)
        best_epoch, history = train_with_validation(
            selection_model,
            fit_graph.to(device),
            validation_graph.to(device),
            args,
        )

        set_seed(args.seed + fold_index)
        full_train = combine_graphs([graphs[name] for name in candidate_train])
        test_graph = graphs[test_name]
        full_train, test_graph = standardize_metrics(full_train, test_graph)
        final_model = make_model(config, device)
        retrain(final_model, full_train.to(device), best_epoch, args)
        embeddings, probabilities, labels = evaluate(final_model, test_graph.to(device))
        metrics = binary_metrics(labels, probabilities)
        train_labels = full_train.y.cpu().numpy().astype(np.int64)
        majority_label = int(train_labels.mean() >= 0.5)
        baseline_probabilities = np.full(len(labels), float(majority_label), dtype=np.float32)
        baseline_metrics = binary_metrics(labels, baseline_probabilities)

        fold_dir = output_dir / "folds" / test_name
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": final_model.state_dict(),
                "encoder_config": asdict(config),
                "test_project": test_name,
                "validation_project": validation_name,
                "selected_epochs": best_epoch,
                "upstream_provenance": provenance,
            },
            fold_dir / "ndg_encoder.pt",
        )
        pd.DataFrame(history).to_csv(fold_dir / "selection_history.csv", index=False)
        np.save(fold_dir / "test_embeddings.npy", embeddings)

        predictions = pd.DataFrame(
            {
                "dataset_name": test_name,
                "name": test_graph.names,
                "source_path": test_graph.source_paths,
                "label": labels,
                "probability": probabilities,
                "prediction": (probabilities >= 0.5).astype(np.int64),
                "baseline_probability": baseline_probabilities,
                "baseline_prediction": majority_label,
                "has_ast": test_graph.view_mask[:, 1].cpu().numpy().astype(np.int64),
                "has_cfg": test_graph.view_mask[:, 2].cpu().numpy().astype(np.int64),
            }
        )
        predictions.to_csv(fold_dir / "test_predictions.csv", index=False)
        prediction_frames.append(predictions)
        embedding_arrays.append(embeddings)
        embedding_frames.append(predictions[["dataset_name", "name", "source_path", "label", "has_ast", "has_cfg"]].copy())
        fold_rows.append(
            {
                "test_project": test_name,
                "validation_project": validation_name,
                "train_projects": ",".join(candidate_train),
                "selected_epochs": best_epoch,
                "train_nodes": full_train.num_nodes,
                "test_nodes": test_graph.num_nodes,
                **{f"model_{key}": value for key, value in metrics.items()},
                **{f"baseline_{key}": value for key, value in baseline_metrics.items()},
            }
        )
        print(
            f"fold={test_name} validation={validation_name} epochs={best_epoch} "
            f"f1={metrics['f1']:.4f} pr_auc={metrics['pr_auc'] if metrics['pr_auc'] is not None else 'NA'}",
            flush=True,
        )

    fold_metrics = pd.DataFrame(fold_rows)
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    all_embeddings = np.concatenate(embedding_arrays, axis=0).astype(np.float32)
    embedding_index = pd.concat(embedding_frames, ignore_index=True)
    embedding_index.insert(0, "embedding_row", np.arange(len(embedding_index), dtype=np.int64))
    embedding_index["embedding_npy"] = str(output_dir / "ndg_embeddings.npy")

    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    all_predictions.to_csv(output_dir / "all_test_predictions.csv", index=False)
    np.save(output_dir / "ndg_embeddings.npy", all_embeddings)
    embedding_index.to_csv(output_dir / "ndg_embedding_index.csv", index=False)

    summary = {
        "protocol": "LOPO node classification with project-level validation and full-training retrain",
        "protocol_limit": (
            "Upstream AST/CFG features are project-level cross-fitted, not regenerated within each outer NDG fold. "
            "Use nested upstream retraining for the strictest stacked-model evaluation."
        ),
        "upstream_embedding_provenance": provenance,
        "model": "gated multi-view relational GATv2 with residuals and layer concatenation",
        "graphs": len(graphs),
        "nodes": int(len(all_predictions)),
        "embedding_dim": int(all_embeddings.shape[1]),
        "node_views": {
            "metrics": len(feature_names),
            "ast": coverage["ast_dim"],
            "cfg": coverage["cfg_dim"],
            "missing_ast": coverage["missing_ast"],
            "missing_cfg": coverage["missing_cfg"],
        },
        "relations": {
            "forward": edge_vocab,
            "inverse_relation_offset": len(edge_vocab),
            "model_relation_count": 2 * len(edge_vocab),
        },
        "encoder_config": asdict(config),
        "model_pooled_metrics": pooled_metric_summary(all_predictions, "probability"),
        "baseline_pooled_metrics": pooled_metric_summary(all_predictions, "baseline_probability"),
        "outputs": {
            "embeddings": str(output_dir / "ndg_embeddings.npy"),
            "embedding_index": str(output_dir / "ndg_embedding_index.csv"),
            "predictions": str(output_dir / "all_test_predictions.csv"),
            "fold_metrics": str(output_dir / "fold_metrics.csv"),
        },
    }
    (output_dir / "ndg_embedding_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.patience <= 0:
        raise ValueError("epochs and patience must be positive")
    output_dir = resolve_path(args.output_dir)
    if output_dir.exists() and not args.no_clean:
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ndg_index = resolve_path(args.ndg_index)
    ast_index = resolve_path(args.ast_index)
    cfg_index = resolve_path(args.cfg_index)
    ast_summary = load_json(resolve_path(args.ast_summary))
    cfg_summary = load_json(resolve_path(args.cfg_summary))
    edge_vocab = {str(key): int(value) for key, value in load_json(resolve_path(args.ndg_edge_vocab)).items()}
    feature_names = [str(value) for value in load_json(resolve_path(args.ndg_feature_names))]
    provenance = validate_upstream_protocols(ast_summary, cfg_summary, args.allow_non_lopo_ast)
    graphs, coverage = load_project_graphs(ndg_index, ast_index, cfg_index, len(edge_vocab))
    if len(graphs) < 3:
        raise ValueError("LOPO with project-level validation requires at least three project graphs")

    config = NDGEncoderConfig(
        metrics_dim=len(feature_names),
        ast_dim=coverage["ast_dim"],
        cfg_dim=coverage["cfg_dim"],
        num_edge_types=2 * len(edge_vocab),
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        edge_type_embedding_dim=args.edge_type_embedding_dim,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
    )
    device = choose_device(args.device)
    print(
        f"NDG LOPO started: projects={len(graphs)} nodes={sum(graph.num_nodes for graph in graphs.values())} "
        f"device={device} provenance={provenance['ndg_result_validity']}",
        flush=True,
    )
    run_lopo(graphs, config, output_dir, args, device, provenance, coverage, edge_vocab, feature_names)
    print(f"embeddings={output_dir / 'ndg_embeddings.npy'}", flush=True)
    print("NDG LOPO finished.", flush=True)


if __name__ == "__main__":
    raise SystemExit(
        "This cross-fitted evaluator is deprecated for final results because its upstream embeddings are not nested "
        "inside each outer fold. Run scripts/evaluate_ndg_nested_lopo.py instead."
    )
