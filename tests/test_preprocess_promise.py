import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from preprocess_promise import map_class_to_file, normalize_promise_schema, validate_project_frame


def test_packaged_class_does_not_use_simple_name_fallback(tmp_path: Path) -> None:
    source = tmp_path / "OldPackage.java"
    source.write_text("class OldPackage {}", encoding="utf-8")

    path, strategy = map_class_to_file(
        "new.package.OldPackage",
        fqcn_to_paths={},
        simple_to_paths={"OldPackage": [source]},
    )

    assert path is None
    assert strategy == "not_found"


def test_duplicate_fqcn_is_not_resolved_arbitrarily(tmp_path: Path) -> None:
    first = tmp_path / "first.java"
    second = tmp_path / "second.java"

    path, strategy = map_class_to_file(
        "example.Duplicate",
        fqcn_to_paths={"example.Duplicate": [first, second]},
        simple_to_paths={"Duplicate": [first, second]},
    )

    assert path is None
    assert strategy == "ambiguous_fqcn"


def test_duplicate_dataset_class_names_are_rejected() -> None:
    columns = {
        "name": ["example.A", "example.A"],
        "bug": [0, 1],
        **{name: [0.0, 1.0] for name in [
            "wmc", "dit", "noc", "cbo", "rfc", "lcom", "ca", "ce", "npm", "lcom3",
            "loc", "dam", "moa", "mfa", "cam", "ic", "cbm", "amc", "max_cc", "avg_cc",
        ]},
    }
    with pytest.raises(ValueError, match="duplicate class names"):
        validate_project_frame("sample", pd.DataFrame(columns))


def test_duplicate_name_header_uses_qualified_class_column() -> None:
    frame = pd.DataFrame(
        {"name": ["ant"], "version": ["1.7"], "name.1": ["org.example.Widget"]}
    )

    normalized = normalize_promise_schema(frame)

    assert normalized.loc[0, "name"] == "org.example.Widget"
