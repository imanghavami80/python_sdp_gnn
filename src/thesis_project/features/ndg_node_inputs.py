"""Label-free NDG node inputs derived from extracted typed dependencies."""

import numpy as np


def dependency_feature_names(relation_names: list[str]) -> list[str]:
    """Stable order for the compact dependency signature of one file."""
    if not relation_names or len(set(relation_names)) != len(relation_names):
        raise ValueError("Dependency relation names must be nonempty and unique")
    return [
        "log_in_dependencies", "log_out_dependencies", "is_isolate",
        *[f"log_in_{name.lower()}" for name in relation_names],
        *[f"log_out_{name.lower()}" for name in relation_names],
        "in_relation_diversity", "out_relation_diversity",
    ]


def dependency_node_inputs(
    num_nodes: int,
    edge_index: np.ndarray,
    edge_type: np.ndarray,
    num_forward_relations: int,
) -> np.ndarray:
    """Summarize each node's original directed and typed dependency edges.

    ``edge_index`` may contain the separately typed inverse edges used for
    message passing. Only forward relation IDs are counted, avoiding duplicates.
    No labels, dataset metrics, or learned embeddings enter these inputs.
    """
    if num_nodes <= 0 or num_forward_relations <= 0:
        raise ValueError("Node and forward-relation counts must be positive")
    edges = np.asarray(edge_index, dtype=np.int64)
    types = np.asarray(edge_type, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2 or types.shape != (edges.shape[1],):
        raise ValueError("Typed NDG edges must have aligned [2, edges] indices and IDs")
    if edges.size and (edges.min() < 0 or edges.max() >= num_nodes):
        raise ValueError("NDG edge endpoint is outside the node range")
    if types.size and (types.min() < 0 or types.max() >= 2 * num_forward_relations):
        raise ValueError("NDG edge relation is outside the forward/inverse vocabulary")

    forward = types < num_forward_relations
    source, target, relation = edges[0, forward], edges[1, forward], types[forward]
    incoming = np.zeros((num_nodes, num_forward_relations), dtype=np.float32)
    outgoing = np.zeros_like(incoming)
    np.add.at(outgoing, (source, relation), 1)
    np.add.at(incoming, (target, relation), 1)
    in_total, out_total = incoming.sum(axis=1), outgoing.sum(axis=1)
    features = np.column_stack((
        np.log1p(in_total), np.log1p(out_total),
        ((in_total + out_total) == 0).astype(np.float32),
        np.log1p(incoming), np.log1p(outgoing),
        (incoming > 0).sum(axis=1) / num_forward_relations,
        (outgoing > 0).sum(axis=1) / num_forward_relations,
    )).astype(np.float32)
    if features.shape != (num_nodes, 5 + 2 * num_forward_relations):
        raise AssertionError("Unexpected dependency signature dimension")
    return features
