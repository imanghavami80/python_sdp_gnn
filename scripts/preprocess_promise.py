#!/usr/bin/env python3
"""Preprocess one or more PROMISE Java SDP datasets.

The pipeline preserves the original Log4j preprocessing choices:
1. validate PROMISE metric schema,
2. convert defect counts to a binary classification label,
3. median-impute numeric metrics,
4. apply log1p to metric columns,
5. scale metric columns,
6. map each class name to a Java source file,
7. write per-project outputs and one combined SDP CSV.

By default this script discovers every `projects/*/*.csv` dataset and treats the
dataset folder as the source root. Source mapping reads Java package
declarations, so it works across projects with different source layouts.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

METRIC_COLUMNS = [
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
]
EXPECTED_COLUMNS = ["name", *METRIC_COLUMNS, "bug"]
PACKAGE_RE = re.compile(r"^\s*package\s+([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)*)\s*;", re.MULTILINE)


@dataclass(frozen=True)
class ProjectSpec:
    dataset_name: str
    csv_path: Path
    source_root: Path


@dataclass
class PreprocessResult:
    dataset_name: str
    dataframe: pd.DataFrame
    mapping: pd.DataFrame
    summary: dict[str, Any]


def clean_dataset_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())


def read_package(java_path: Path) -> str:
    text = java_path.read_text(encoding="utf-8", errors="ignore")
    match = PACKAGE_RE.search(text)
    return match.group(1) if match else ""


def infer_path_fqcn(java_path: Path, source_root: Path) -> str:
    return ".".join(java_path.relative_to(source_root).with_suffix("").parts)


def build_source_index(source_root: Path) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    """Index Java files by package-derived FQCN and simple class name."""
    fqcn_to_path: dict[str, Path] = {}
    simple_to_paths: dict[str, list[Path]] = {}

    for java_path in sorted(source_root.rglob("*.java")):
        package = read_package(java_path)
        package_fqcn = f"{package}.{java_path.stem}" if package else java_path.stem
        path_fqcn = infer_path_fqcn(java_path, source_root)

        fqcn_to_path.setdefault(package_fqcn, java_path.resolve())
        fqcn_to_path.setdefault(path_fqcn, java_path.resolve())
        simple_to_paths.setdefault(java_path.stem, []).append(java_path.resolve())

    return fqcn_to_path, simple_to_paths


def map_class_to_file(
    class_name: str,
    fqcn_to_path: dict[str, Path],
    simple_to_paths: dict[str, list[Path]],
) -> tuple[str | None, str]:
    """Resolve PROMISE class name to a Java source file path."""
    normalized = str(class_name).replace("$", ".")
    exact = fqcn_to_path.get(normalized)
    if exact is not None:
        return str(exact), "exact_fqcn"

    # Inner/nested classes are stored in the outer class source file.
    owner = normalized
    while "." in owner:
        owner = owner.rsplit(".", 1)[0]
        if owner in fqcn_to_path:
            return str(fqcn_to_path[owner]), "outer_class"

    simple_name = normalized.split(".")[-1]
    candidates = simple_to_paths.get(simple_name, [])
    if len(candidates) == 1:
        return str(candidates[0]), "unique_simple_name"
    if len(candidates) > 1:
        return None, "ambiguous_simple_name"
    return None, "not_found"


def validate_project_frame(dataset_name: str, df: pd.DataFrame) -> None:
    missing = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name}: missing expected columns: {missing}")


def discover_projects(projects_root: Path) -> list[ProjectSpec]:
    specs: list[ProjectSpec] = []
    for csv_path in sorted(projects_root.glob("*/*.csv")):
        if csv_path.name == "promise_dataset_summary.csv":
            continue
        dataset_name = clean_dataset_name(csv_path.stem)
        specs.append(ProjectSpec(dataset_name=dataset_name, csv_path=csv_path.resolve(), source_root=csv_path.parent.resolve()))
    return specs


def parse_project_arg(value: str, repo_root: Path) -> ProjectSpec:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--project must use DATASET_NAME:CSV_PATH:SOURCE_ROOT")
    dataset_name, csv_path, source_root = parts
    return ProjectSpec(
        dataset_name=clean_dataset_name(dataset_name),
        csv_path=(repo_root / csv_path).resolve() if not Path(csv_path).is_absolute() else Path(csv_path).resolve(),
        source_root=(repo_root / source_root).resolve() if not Path(source_root).is_absolute() else Path(source_root).resolve(),
    )


def make_scaler(name: str) -> StandardScaler | MinMaxScaler:
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    raise ValueError("scaler must be 'standard' or 'minmax'")


def load_raw_projects(specs: list[ProjectSpec]) -> list[pd.DataFrame]:
    raw_frames: list[pd.DataFrame] = []
    for spec in specs:
        if not spec.csv_path.exists():
            raise FileNotFoundError(f"{spec.dataset_name}: missing CSV file: {spec.csv_path}")
        if not spec.source_root.exists():
            raise FileNotFoundError(f"{spec.dataset_name}: missing source root: {spec.source_root}")
        df = pd.read_csv(spec.csv_path)
        validate_project_frame(spec.dataset_name, df)
        df = df[EXPECTED_COLUMNS].copy()
        df.insert(0, "dataset_name", spec.dataset_name)
        raw_frames.append(df)
    return raw_frames


def preprocess_frames(
    specs: list[ProjectSpec],
    raw_frames: list[pd.DataFrame],
    scaler_name: str,
    scale_scope: str,
) -> list[PreprocessResult]:
    """Preprocess all frames with either global or per-project scaling."""
    if scale_scope not in {"global", "project"}:
        raise ValueError("scale_scope must be 'global' or 'project'")

    work_frames = []
    for frame in raw_frames:
        df = frame.copy()
        df["bug"] = (pd.to_numeric(df["bug"], errors="coerce").fillna(0) > 0).astype(int)
        for column in METRIC_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        work_frames.append(df)

    if scale_scope == "global":
        combined_metrics = pd.concat([df[METRIC_COLUMNS] for df in work_frames], ignore_index=True)
        medians = combined_metrics.median(numeric_only=True)
        processed_metrics = combined_metrics.fillna(medians)
        if (processed_metrics < 0).any().any():
            raise ValueError("log1p transformation requires non-negative metric features")
        processed_metrics = np.log1p(processed_metrics)
        scaler = make_scaler(scaler_name)
        scaled_all = pd.DataFrame(scaler.fit_transform(processed_metrics), columns=METRIC_COLUMNS)

        offset = 0
        scaled_by_project: list[pd.DataFrame] = []
        for df in work_frames:
            n_rows = len(df)
            scaled_by_project.append(scaled_all.iloc[offset : offset + n_rows].reset_index(drop=True))
            offset += n_rows
    else:
        scaled_by_project = []
        for df in work_frames:
            metrics = df[METRIC_COLUMNS].copy()
            metrics = metrics.fillna(metrics.median(numeric_only=True))
            if (metrics < 0).any().any():
                raise ValueError("log1p transformation requires non-negative metric features")
            metrics = np.log1p(metrics)
            scaler = make_scaler(scaler_name)
            scaled_by_project.append(pd.DataFrame(scaler.fit_transform(metrics), columns=METRIC_COLUMNS))

    results: list[PreprocessResult] = []
    for spec, df, scaled_metrics in zip(specs, work_frames, scaled_by_project):
        scaled_df = pd.DataFrame(
            {
                "dataset_name": df["dataset_name"].astype(str),
                "name": df["name"].astype(str),
                "bug": df["bug"].astype(int),
            }
        )
        for column in METRIC_COLUMNS:
            scaled_df[column] = scaled_metrics[column].astype(float)

        fqcn_to_path, simple_to_paths = build_source_index(spec.source_root)
        mapping_rows = []
        for class_name in scaled_df["name"]:
            source_path, match_strategy = map_class_to_file(class_name, fqcn_to_path, simple_to_paths)
            mapping_rows.append(
                {
                    "dataset_name": spec.dataset_name,
                    "name": class_name,
                    "source_path": source_path,
                    "match_strategy": match_strategy,
                }
            )
        mapping_df = pd.DataFrame(mapping_rows)
        final_df = scaled_df.merge(mapping_df, on=["dataset_name", "name"], how="left")

        strategy_counts = mapping_df["match_strategy"].value_counts(dropna=False).to_dict()
        bug_label_counts = scaled_df["bug"].value_counts(dropna=False).sort_index().to_dict()
        summary = {
            "dataset_name": spec.dataset_name,
            "csv_path": str(spec.csv_path),
            "source_root": str(spec.source_root),
            "rows": int(len(final_df)),
            "columns": int(len(final_df.columns)),
            "feature_columns": len(METRIC_COLUMNS),
            "feature_transform": "log1p",
            "scaler": scaler_name,
            "scale_scope": scale_scope,
            "bug_label_counts": {str(k): int(v) for k, v in bug_label_counts.items()},
            "mapped_classes": int(mapping_df["source_path"].notna().sum()),
            "total_classes": int(len(mapping_df)),
            "match_strategy_counts": {str(k): int(v) for k, v in strategy_counts.items()},
        }
        results.append(PreprocessResult(spec.dataset_name, final_df, mapping_df, summary))

    return results


def write_outputs(results: list[PreprocessResult], output_root: Path, scaler_name: str, scale_scope: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    combined_dir = output_root / "promise"
    combined_dir.mkdir(parents=True, exist_ok=True)

    combined_frames = []
    summary_rows = []
    for result in results:
        dataset_dir = output_root / result.dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        preprocessed_path = dataset_dir / f"{result.dataset_name}_preprocessed_{scaler_name}.csv"
        mapping_path = dataset_dir / f"{result.dataset_name}_name_to_source_mapping.csv"
        mapping_json_path = dataset_dir / f"{result.dataset_name}_name_to_source_mapping.json"
        summary_path = dataset_dir / f"{result.dataset_name}_preprocess_summary.json"

        result.dataframe.to_csv(preprocessed_path, index=False)
        result.mapping.to_csv(mapping_path, index=False)
        mapping_json_path.write_text(
            json.dumps(result.mapping.set_index("name")["source_path"].to_dict(), indent=2),
            encoding="utf-8",
        )

        result.summary.update(
            {
                "preprocessed_file": str(preprocessed_path),
                "mapping_csv": str(mapping_path),
                "mapping_json": str(mapping_json_path),
            }
        )
        summary_path.write_text(json.dumps(result.summary, indent=2), encoding="utf-8")

        combined_frames.append(result.dataframe)
        summary_rows.append(result.summary)

    combined = pd.concat(combined_frames, ignore_index=True)
    combined_path = combined_dir / f"promise_preprocessed_{scaler_name}.csv"
    combined_summary_path = combined_dir / "promise_preprocess_summary.json"
    combined.to_csv(combined_path, index=False)

    combined_summary = {
        "datasets": [row["dataset_name"] for row in summary_rows],
        "num_datasets": len(summary_rows),
        "rows": int(len(combined)),
        "columns": int(len(combined.columns)),
        "feature_columns": len(METRIC_COLUMNS),
        "feature_transform": "log1p",
        "scaler": scaler_name,
        "scale_scope": scale_scope,
        "defective_samples": int(combined["bug"].sum()),
        "non_defective_samples": int((combined["bug"] == 0).sum()),
        "mapped_classes": int(combined["source_path"].notna().sum()),
        "unmapped_classes": int(combined["source_path"].isna().sum()),
        "combined_preprocessed_file": str(combined_path),
        "project_summaries": summary_rows,
    }
    combined_summary_path.write_text(json.dumps(combined_summary, indent=2), encoding="utf-8")

    print("PROMISE preprocessing complete")
    print(f"datasets={len(summary_rows)} rows={len(combined)} columns={len(combined.columns)}")
    print(f"defective_samples={combined_summary['defective_samples']}")
    print(f"mapped_classes={combined_summary['mapped_classes']}/{len(combined)}")
    print(f"combined_preprocessed_file={combined_path}")
    print(f"combined_summary={combined_summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess PROMISE Java SDP datasets.")
    parser.add_argument("--projects-root", type=Path, default=Path("projects"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--scaler", choices=["standard", "minmax"], default="standard")
    parser.add_argument(
        "--scale-scope",
        choices=["global", "project"],
        default="global",
        help="Use global scaling across all datasets or fit one scaler per project.",
    )
    parser.add_argument(
        "--project",
        action="append",
        default=[],
        metavar="DATASET:CSV_PATH:SOURCE_ROOT",
        help="Explicit project specification. Can be repeated. Defaults to auto-discovery.",
    )
    parser.add_argument("--dataset-name", help="Single-project dataset name.")
    parser.add_argument("--csv-path", type=Path, help="Single-project PROMISE CSV path.")
    parser.add_argument("--source-root", type=Path, help="Single-project source root.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    projects_root = (repo_root / args.projects_root).resolve() if not args.projects_root.is_absolute() else args.projects_root
    output_root = (repo_root / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root

    specs: list[ProjectSpec]
    if args.csv_path or args.source_root or args.dataset_name:
        if not (args.csv_path and args.source_root and args.dataset_name):
            raise SystemExit("--dataset-name, --csv-path, and --source-root must be provided together")
        csv_path = (repo_root / args.csv_path).resolve() if not args.csv_path.is_absolute() else args.csv_path.resolve()
        source_root = (repo_root / args.source_root).resolve() if not args.source_root.is_absolute() else args.source_root.resolve()
        specs = [ProjectSpec(clean_dataset_name(args.dataset_name), csv_path, source_root)]
    elif args.project:
        specs = [parse_project_arg(value, repo_root) for value in args.project]
    else:
        specs = discover_projects(projects_root)

    if not specs:
        raise SystemExit("No PROMISE datasets found")

    raw_frames = load_raw_projects(specs)
    results = preprocess_frames(specs, raw_frames, scaler_name=args.scaler, scale_scope=args.scale_scope)
    write_outputs(results, output_root=output_root, scaler_name=args.scaler, scale_scope=args.scale_scope)


if __name__ == "__main__":
    main()
