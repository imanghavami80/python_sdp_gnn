import hashlib
import io
import sys
import tarfile
from pathlib import Path

import pytest

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import extract_promise_cfg
from extract_promise_cfg import ensure_official_release_bytecode


def test_release_bytecode_uses_valid_cached_jar(tmp_path: Path, monkeypatch) -> None:
    jar_bytes = b"exact release bytecode"
    digest = hashlib.sha256(jar_bytes).hexdigest()
    monkeypatch.setitem(
        extract_promise_cfg.OFFICIAL_RELEASE_BYTECODE,
        "sample-1.0",
        {
            "url": "https://invalid.example/release.tar.gz",
            "archive_sha256": "unused",
            "members": [{"path": "sample/lib/sample.jar", "sha256": digest}],
        },
    )
    cached = tmp_path / "sample-1.0" / "sample.jar"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(jar_bytes)

    paths, status = ensure_official_release_bytecode("sample-1.0", tmp_path)

    assert paths == [cached]
    assert status == "cache"


def test_release_bytecode_extracts_only_verified_member(tmp_path: Path, monkeypatch) -> None:
    jar_bytes = b"verified jar"
    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo("sample/lib/sample.jar")
        info.size = len(jar_bytes)
        archive.addfile(info, io.BytesIO(jar_bytes))
    archive_bytes = archive_buffer.getvalue()
    archive_path = tmp_path / "source.tar.gz"
    archive_path.write_bytes(archive_bytes)
    monkeypatch.setitem(
        extract_promise_cfg.OFFICIAL_RELEASE_BYTECODE,
        "sample-1.0",
        {
            "url": archive_path.as_uri(),
            "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
            "members": [
                {
                    "path": "sample/lib/sample.jar",
                    "sha256": hashlib.sha256(jar_bytes).hexdigest(),
                }
            ],
        },
    )

    paths, status = ensure_official_release_bytecode("sample-1.0", tmp_path / "cache")

    assert status == "download"
    assert paths[0].read_bytes() == jar_bytes


def test_unconfigured_dataset_uses_source_build(tmp_path: Path) -> None:
    paths, status = ensure_official_release_bytecode("sample-without-release", tmp_path)

    assert paths == []
    assert status == "not_configured"


def test_configured_release_failure_cannot_silently_fall_back(tmp_path: Path, monkeypatch) -> None:
    invalid_archive = tmp_path / "invalid.tar.gz"
    invalid_archive.write_bytes(b"not the configured archive")
    monkeypatch.setitem(
        extract_promise_cfg.OFFICIAL_RELEASE_BYTECODE,
        "sample-1.0",
        {
            "url": invalid_archive.as_uri(),
            "archive_sha256": "0" * 64,
            "members": [{"path": "sample.jar", "sha256": "0" * 64}],
        },
    )

    with pytest.raises(RuntimeError, match="Official release bytecode unavailable"):
        ensure_official_release_bytecode("sample-1.0", tmp_path / "cache")
