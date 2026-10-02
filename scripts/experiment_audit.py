"""Small, read-only provenance helpers for local experiments."""
import hashlib
import importlib.metadata
import platform
import sys
from pathlib import Path

import pandas as pd
import torch


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def environment(device):
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                machine=platform.machine(), device=str(device),
                torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                packages={p: importlib.metadata.version(p) for p in
                          ['numpy', 'pandas', 'scikit-learn', 'torch', 'torch-geometric', 'networkx']})


def fingerprint_inputs(paths, selected, resolve):
    """Hash indices/vocabularies and selected projects' referenced extraction artifacts."""
    files = {resolve(p) for p in paths}
    for path in paths:
        if path.suffix != '.csv':
            continue
        index = pd.read_csv(resolve(path))
        if 'dataset_name' in index:
            index = index[index.dataset_name.isin(selected)]
        for column in index:
            if column.endswith('_npy') or column == 'graph_json':
                files.update(resolve(Path(str(p))) for p in index[column].dropna().unique())
    return {str(path): sha256(path) for path in sorted(files)}


def fingerprint_source(root):
    files = list((root/'scripts').glob('*.py')) + list((root/'src').rglob('*.py'))
    files += [root/'requirements.txt', root/'pyproject.toml']
    return {str(p.relative_to(root)): sha256(p) for p in sorted(files)}
