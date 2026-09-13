import inspect

import numpy as np
import pytest

pytest.importorskip("networkx")

from thesis_project.features import (
    NDGStructuralFeatureConfig,
    extract_ndg_structural_features,
    structural_feature_names,
)


def sample_graph() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([[0, 2, 1], [1, 1, 2]], dtype=np.int64),
        np.asarray([0, 1, 0], dtype=np.int64),
    )


def test_structural_features_cover_typed_local_and_global_roles() -> None:
    edge_index, edge_type = sample_graph()
    names = structural_feature_names(["CALL", "TYPE"])

    features = extract_ndg_structural_features(
        4,
        edge_index,
        edge_type,
        ["CALL", "TYPE"],
        NDGStructuralFeatureConfig(betweenness_samples=2, random_state=7),
    )

    assert features.shape == (4, len(names))
    assert np.isfinite(features).all()
    assert features[3, names.index("ndg_is_isolate")] == 1.0
    assert features[1, names.index("ndg_in_call_log_count")] == pytest.approx(np.log(2.0))
    assert features[1, names.index("ndg_in_type_log_count")] == pytest.approx(np.log(2.0))
    assert features[1, names.index("ndg_out_call_log_count")] == pytest.approx(np.log(2.0))


def test_structural_extraction_is_deterministic() -> None:
    edge_index, edge_type = sample_graph()
    config = NDGStructuralFeatureConfig(betweenness_samples=2, random_state=11)

    first = extract_ndg_structural_features(4, edge_index, edge_type, ["CALL", "TYPE"], config)
    second = extract_ndg_structural_features(4, edge_index, edge_type, ["CALL", "TYPE"], config)

    assert np.array_equal(first, second)


def test_structural_extractor_cannot_receive_defect_labels() -> None:
    assert "labels" not in inspect.signature(extract_ndg_structural_features).parameters


def test_structural_extractor_rejects_invalid_relation_ids() -> None:
    edge_index, edge_type = sample_graph()
    edge_type[0] = 2

    with pytest.raises(ValueError, match="relation"):
        extract_ndg_structural_features(4, edge_index, edge_type, ["CALL", "TYPE"])
