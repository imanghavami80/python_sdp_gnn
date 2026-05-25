#!/usr/bin/env python3
"""Preprocess PROMISE Log4j metrics for multi-view SDP experiments.

This script:
1. Loads the PROMISE CSV and validates expected structure.
2. Handles missing numeric values (median imputation).
3. Scales numeric features (excluding the supervised label `bug`).
4. Maps each class `name` to a Java source file path in local Log4j sources.
5. Writes preprocessed data and mapping artifacts to disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


EXPECTED_COLUMNS = [
    "name",
    "wmc",
    "dit",
    "noc",
    "cbo",
    "rfc",
    "lcom",
    "ca",
    "ce",
    "npm",
    "lcom3",
    "loc",
    "dam",
    "moa",
    "mfa",
    "cam",
    "ic",
    "cbm",
    "amc",
    "max_cc",
    "avg_cc",
    "bug",
]


def infer_fqcn(java_file: Path, source_root: Path) -> str:
    """Infer fully-qualified class name from source-relative Java path."""
    rel = java_file.relative_to(source_root)
    return str(rel.with_suffix("")).replace("/", ".")


def build_source_index(source_root: Path) -> Tuple[Dict[str, Path], Dict[str, List[Path]]]:
    """Create lookup maps for exact FQCN and fallback simple class name matching."""
    fqcn_to_path: Dict[str, Path] = {}
    simple_to_paths: Dict[str, List[Path]] = {}

    for java_path in source_root.rglob("*.java"):
        fqcn = infer_fqcn(java_path, source_root)
        fqcn_to_path[fqcn] = java_path

        simple = java_path.stem
        simple_to_paths.setdefault(simple, []).append(java_path)

    return fqcn_to_path, simple_to_paths


def map_class_to_file(
    class_name: str,
    fqcn_to_path: Dict[str, Path],
    simple_to_paths: Dict[str, List[Path]],
) -> Tuple[str | None, str]:
    """Resolve class -> file path with strategy metadata.

    Strategy order:
    1) Exact fully-qualified name match.
    2) Unique simple class name match.
    3) Not found.
    """
    exact = fqcn_to_path.get(class_name)
    if exact:
        return str(exact), "exact_fqcn"

    simple_name = class_name.split(".")[-1]
    candidates = simple_to_paths.get(simple_name, [])
    if len(candidates) == 1:
        return str(candidates[0]), "unique_simple_name"

    return None, "not_found"


def preprocess_log4j(
    csv_path: Path,
    source_root: Path,
    output_dir: Path,
    scaler_name: str = "standard",
) -> None:
    """Execute full preprocessing pipeline and write artifacts."""
    df = pd.read_csv(csv_path)

    # Basic structure validation.
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    df = df[EXPECTED_COLUMNS].copy()

    # Ensure `bug` is the supervised label (integer 0/1).
    df["bug"] = pd.to_numeric(df["bug"], errors="coerce").fillna(0).astype(int)

    # Numeric feature columns exclude identifier + label.
    feature_cols = [c for c in df.columns if c not in ("name", "bug")]

    # Graceful missing-value handling via median imputation.
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    if scaler_name == "standard":
        scaler = StandardScaler()
    elif scaler_name == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("scaler_name must be 'standard' or 'minmax'")

    # Scale only metric features; keep `name` and `bug` unchanged.
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df_scaled[feature_cols])

    # Build name -> source file mapping.
    fqcn_to_path, simple_to_paths = build_source_index(source_root)
    mapping_rows = []
    for class_name in df_scaled["name"]:
        source_path, match_strategy = map_class_to_file(class_name, fqcn_to_path, simple_to_paths)
        mapping_rows.append(
            {
                "name": class_name,
                "source_path": source_path,
                "match_strategy": match_strategy,
            }
        )

    mapping_df = pd.DataFrame(mapping_rows)

    # Merge mapping back into preprocessed frame.
    final_df = df_scaled.merge(mapping_df, on="name", how="left")

    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessed_path = output_dir / f"log4j_preprocessed_{scaler_name}.csv"
    mapping_path = output_dir / "log4j_name_to_source_mapping.csv"
    mapping_json_path = output_dir / "log4j_name_to_source_mapping.json"
    summary_path = output_dir / "log4j_preprocess_summary.txt"

    final_df.to_csv(preprocessed_path, index=False)
    mapping_df.to_csv(mapping_path, index=False)
    mapping_json_path.write_text(json.dumps(mapping_df.set_index("name")["source_path"].to_dict(), indent=2))

    matched = int(mapping_df["source_path"].notna().sum())
    total = int(len(mapping_df))
    strategy_counts = mapping_df["match_strategy"].value_counts(dropna=False).to_dict()

    summary_lines = [
        f"rows={len(df)}",
        f"columns={len(df.columns)}",
        f"feature_columns={len(feature_cols)}",
        f"scaler={scaler_name}",
        f"mapped_classes={matched}/{total}",
        f"match_strategy_counts={strategy_counts}",
        f"preprocessed_file={preprocessed_path}",
        f"mapping_csv={mapping_path}",
        f"mapping_json={mapping_json_path}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n")

    print("Preprocessing complete")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    preprocess_log4j(
        csv_path=repo_root / "projects/log4j/log4j-1.0.csv",
        source_root=repo_root / "projects/log4j/logging-log4j1-v_1_0/src/java",
        output_dir=repo_root / "outputs/log4j",
        scaler_name="standard",
    )
