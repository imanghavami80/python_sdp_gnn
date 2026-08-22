#!/usr/bin/env python3
"""Generate cross-project AST embeddings with Leave-One-Project-Out training."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
for path in (SCRIPTS_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import torch

    from generate_ast_embeddings import (
        ASTGraphDataset,
        extract_embeddings,
        load_inputs,
        make_loader,
        set_seed,
        train_model,
    )
    from thesis_project.models import ASTEncoderConfig, ASTGINEncoder, ASTGraphClassifier
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing GNN dependencies. Install them with:\n"
        "  .venv/bin/pip install -r requirements.txt\n"
        f"Original error: {exc}"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AST embeddings with LOPO supervision.")
    parser.add_argument("--graph-index", type=Path, default=Path("outputs/promise/ast/graph_index.csv"))
    parser.add_argument("--node-type-vocab", type=Path, default=Path("outputs/promise/ast/node_type_vocab.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/promise/embeddings/ast_lopo"))
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
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-normalize-structural-features", action="store_true")
    parser.add_argument("--no-clean", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


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


def split_train_validation(
    graph_index: pd.DataFrame,
    train_project_indices: np.ndarray,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels = graph_index.iloc[train_project_indices]["label"].astype(int).to_numpy()
    stratify = labels if len(np.unique(labels)) == 2 and min(np.bincount(labels)) >= 2 else None
    train_indices, validation_indices = train_test_split(
        train_project_indices,
        test_size=val_ratio,
        random_state=seed,
        stratify=stratify,
    )
    return np.asarray(train_indices, dtype=np.int64), np.asarray(validation_indices, dtype=np.int64)


def build_model(args: argparse.Namespace, num_node_types: int, device: torch.device) -> tuple[ASTGraphClassifier, ASTEncoderConfig]:
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


def run_lopo(
    graph_index: pd.DataFrame,
    vocab: dict[str, int],
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    labels = graph_index["label"].astype(int).to_numpy()
    projects = sorted(graph_index["dataset_name"].astype(str).unique())
    all_embeddings = np.zeros((len(graph_index), args.output_dim), dtype=np.float32)
    fold_rows: list[dict[str, Any]] = []
    normalize = not args.no_normalize_structural_features

    for fold_index, test_project in enumerate(projects):
        set_seed(args.seed + fold_index)
        test_indices = graph_index.index[graph_index["dataset_name"].astype(str) == test_project].to_numpy(dtype=np.int64)
        candidate_train = graph_index.index[graph_index["dataset_name"].astype(str) != test_project].to_numpy(dtype=np.int64)
        train_indices, validation_indices = split_train_validation(
            graph_index, candidate_train, args.val_ratio, args.seed + fold_index
        )

        train_dataset = ASTGraphDataset(graph_index, train_indices, normalize_structural_features=normalize)
        validation_dataset = ASTGraphDataset(graph_index, validation_indices, normalize_structural_features=normalize)
        test_dataset = ASTGraphDataset(graph_index, test_indices, normalize_structural_features=normalize)
        train_loader = make_loader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
        validation_loader = make_loader(validation_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
        model, config = build_model(args, len(vocab), device)
        best_state, history, best_info = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=validation_loader,
            labels=labels,
            train_indices=train_indices,
            args=args,
            device=device,
        )
        model.load_state_dict(best_state)
        fold_embeddings_full = extract_embeddings(
            model=model,
            dataset=test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            output_dim=args.output_dim,
        )
        all_embeddings[test_indices] = fold_embeddings_full[test_indices]

        fold_dir = output_dir / "folds" / test_project
        fold_dir.mkdir(parents=True, exist_ok=True)
        np.save(fold_dir / "test_embeddings.npy", all_embeddings[test_indices])
        pd.DataFrame(history).to_csv(fold_dir / "training_history.csv", index=False)
        graph_index.iloc[test_indices][
            ["graph_id", "dataset_name", "name", "source_path", "label", "parser_mode"]
        ].to_csv(fold_dir / "test_embedding_index.csv", index=False)
        torch.save(
            {
                "model_state_dict": model.cpu().state_dict(),
                "encoder_config": asdict(config),
                "node_type_vocab": vocab,
                "best_info": best_info,
                "protocol": "LOPO",
                "test_project": test_project,
                "script_args": vars(args),
            },
            fold_dir / "ast_encoder.pt",
        )
        fold_rows.append(
            {
                "test_project": test_project,
                "train_graphs": len(train_indices),
                "validation_graphs": len(validation_indices),
                "test_graphs": len(test_indices),
                "best_epoch": best_info["best_epoch"],
                "best_validation_loss": best_info["best_loss"],
            }
        )
        print(
            f"fold={test_project} train={len(train_indices)} validation={len(validation_indices)} "
            f"test={len(test_indices)} best_epoch={best_info['best_epoch']}",
            flush=True,
        )

    embeddings_path = output_dir / "ast_embeddings.npy"
    index_path = output_dir / "ast_embedding_index.csv"
    np.save(embeddings_path, all_embeddings)
    embedding_index = graph_index[
        ["graph_id", "dataset_name", "name", "source_path", "label", "num_nodes", "num_edges", "parser_mode"]
    ].copy()
    embedding_index.insert(0, "embedding_row", np.arange(len(embedding_index), dtype=np.int64))
    embedding_index["embedding_npy"] = str(embeddings_path)
    embedding_index.to_csv(index_path, index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics.csv", index=False)
    summary = {
        "protocol": "LOPO",
        "embedding_protocol": "Each AST is encoded by a supervised model that did not train on its project.",
        "graphs": len(graph_index),
        "projects": len(projects),
        "embedding_dim": args.output_dim,
        "embeddings_npy": str(embeddings_path),
        "embedding_index_csv": str(index_path),
        "normalization": {
            "enabled": normalize,
            "depth": "depth / max_depth_in_graph",
            "out_degree": "log1p(out_degree)",
            "has_identifier": "unchanged binary flag",
        },
    }
    (output_dir / "ast_embedding_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.patience <= 0:
        raise ValueError("epochs, batch-size, and patience must be positive")
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("val-ratio must be in (0, 1)")
    output_dir = resolve_path(args.output_dir)
    if output_dir.exists() and not args.no_clean:
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_index, vocab = load_inputs(resolve_path(args.graph_index), resolve_path(args.node_type_vocab))
    device = choose_device(args.device)
    print(f"AST LOPO started: graphs={len(graph_index)} projects={graph_index['dataset_name'].nunique()} device={device}")
    run_lopo(graph_index, vocab, output_dir, args, device)
    print(f"embeddings={output_dir / 'ast_embeddings.npy'}")
    print("AST LOPO finished.")


if __name__ == "__main__":
    main()
