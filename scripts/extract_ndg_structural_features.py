#!/usr/bin/env python3
"""Extract label-free node-role descriptors from existing PROMISE NDGs."""

from __future__ import annotations

import argparse
import json
import hashlib
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from thesis_project.features import (  # noqa: E402
    NDGStructuralFeatureConfig,
    extract_ndg_structural_features,
    structural_feature_names,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract handcrafted structural features from project-level NDGs."
    )
    parser.add_argument(
        "--ndg-index", type=Path, default=Path("outputs/promise/ndg/graph_index.csv")
    )
    parser.add_argument(
        "--edge-vocab", type=Path, default=Path("outputs/promise/ndg/edge_type_vocab.json")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/promise/ndg_structural")
    )
    parser.add_argument("--betweenness-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def prepare_output_dir(path: Path) -> None:
    forbidden = {Path(path.anchor), Path.home().resolve(), REPO_ROOT.resolve()}
    if path.resolve() in forbidden:
        raise ValueError(f"Refusing to clean unsafe output directory: {path}")
    if path.exists():
        shutil.rmtree(path)
    (path / "tensors").mkdir(parents=True, exist_ok=True)


def load_edge_types(path: Path) -> list[str]:
    vocabulary = json.loads(path.read_text(encoding="utf-8"))
    ids = sorted(int(value) for value in vocabulary.values())
    if ids != list(range(len(ids))):
        raise ValueError("NDG edge vocabulary ids must be contiguous from zero")
    return [
        str(name)
        for name, _ in sorted(vocabulary.items(), key=lambda item: int(item[1]))
    ]


def main() -> None:
    args = parse_args()
    index_path = resolve(args.ndg_index)
    edge_vocab_path = resolve(args.edge_vocab)
    output_dir = resolve(args.output_dir)
    if not index_path.exists() or not edge_vocab_path.exists():
        raise FileNotFoundError("NDG outputs are missing; run scripts/extract_promise_ndg.py first")
    config = NDGStructuralFeatureConfig(
        betweenness_samples=args.betweenness_samples,
        random_state=args.seed,
    )
    index = pd.read_csv(index_path)
    required = {"dataset_name", "num_nodes", "edge_index_npy", "edge_type_npy", "graph_json"}
    missing = sorted(required - set(index.columns))
    if missing:
        raise ValueError(f"NDG index is missing columns: {missing}")
    edge_type_names = load_edge_types(edge_vocab_path)
    feature_names = structural_feature_names(edge_type_names)
    prepare_output_dir(output_dir)

    index_rows: list[dict[str, Any]] = []
    matrices: list[np.ndarray] = []
    started = time.perf_counter()
    for _, row in index.sort_values("dataset_name").iterrows():
        dataset_name = str(row["dataset_name"])
        project_started = time.perf_counter()
        matrix = extract_ndg_structural_features(
            num_nodes=int(row["num_nodes"]),
            edge_index=np.load(str(row["edge_index_npy"])),
            edge_type=np.load(str(row["edge_type_npy"])),
            edge_type_names=edge_type_names,
            config=config,
        )
        output_path = output_dir / "tensors" / f"{dataset_name}_structural_x.npy"
        np.save(output_path, matrix)
        matrices.append(matrix)
        index_rows.append(
            {
                "dataset_name": dataset_name,
                "num_nodes": len(matrix),
                "feature_dim": matrix.shape[1],
                "structural_x_npy": str(output_path),
                "source_graph_sha256": hashlib.sha256(Path(str(row["graph_json"])).read_bytes()).hexdigest(),
            }
        )
        print(
            f"project={dataset_name} nodes={len(matrix)} features={matrix.shape[1]} "
            f"seconds={time.perf_counter() - project_started:.2f}",
            flush=True,
        )

    feature_index = pd.DataFrame(index_rows)
    feature_index.to_csv(output_dir / "feature_index.csv", index=False)
    (output_dir / "feature_names.json").write_text(
        json.dumps(feature_names, indent=2), encoding="utf-8"
    )
    combined = np.concatenate(matrices, axis=0)
    summary = {
        "source_ndg_index": str(index_path),
        "edge_vocabulary": str(edge_vocab_path),
        "configuration": asdict(config),
        "uses_node_labels": False,
        "projects": len(index_rows),
        "nodes": len(combined),
        "feature_dim": len(feature_names),
        "feature_names": feature_names,
        "all_finite": bool(np.isfinite(combined).all()),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"NDG structural extraction finished: projects={len(index_rows)} "
        f"nodes={len(combined)} features={len(feature_names)} "
        f"seconds={summary['elapsed_seconds']:.2f}",
        flush=True,
    )
    print(f"index={output_dir / 'feature_index.csv'}", flush=True)


if __name__ == "__main__":
    main()
