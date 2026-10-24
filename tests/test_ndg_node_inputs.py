import numpy as np
import pytest

from thesis_project.features.ndg_node_inputs import (
    dependency_feature_names, dependency_node_inputs,
)
from thesis_project.training.ndg import add_inverse_relations


def test_dependency_inputs_count_original_direction_and_relation_only():
    forward_edges = np.array([[0, 2], [1, 1]], dtype=np.int64)
    forward_types = np.array([0, 1], dtype=np.int64)
    edges, types = add_inverse_relations(forward_edges, forward_types, 2)
    features = dependency_node_inputs(4, edges, types, 2)

    assert features.shape == (4, len(dependency_feature_names(['CALL', 'FIELD'])))
    np.testing.assert_allclose(features[0, :3], [0, np.log(2), 0])
    np.testing.assert_allclose(features[1, :3], [np.log(3), 0, 0])
    np.testing.assert_allclose(features[1, 3:7], [np.log(2), np.log(2), 0, 0])
    np.testing.assert_allclose(features[1, -2:], [1, 0])
    np.testing.assert_allclose(features[3], [0, 0, 1, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(features, dependency_node_inputs(4, forward_edges, forward_types, 2))


def test_dependency_inputs_validate_edges():
    with pytest.raises(ValueError, match='outside the node range'):
        dependency_node_inputs(2, np.array([[0], [2]]), np.array([0]), 1)
    with pytest.raises(ValueError, match='outside the forward/inverse'):
        dependency_node_inputs(2, np.array([[0], [1]]), np.array([2]), 1)
