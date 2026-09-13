"""Node-level structural descriptors for directed, typed software NDGs."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class NDGStructuralFeatureConfig:
    """Configuration for deterministic NDG topology descriptors."""

    betweenness_samples: int = 128
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.betweenness_samples <= 0:
            raise ValueError("betweenness_samples must be positive")


BASE_FEATURE_NAMES = [
    "ndg_log_in_edge_count",
    "ndg_log_out_edge_count",
    "ndg_in_degree_normalized",
    "ndg_out_degree_normalized",
    "ndg_total_degree_normalized",
    "ndg_is_isolate",
    "ndg_in_relation_diversity",
    "ndg_out_relation_diversity",
    "ndg_in_relation_entropy",
    "ndg_out_relation_entropy",
    "ndg_pagerank_influence",
    "ndg_reverse_pagerank_influence",
    "ndg_betweenness",
    "ndg_in_closeness",
    "ndg_out_closeness",
    "ndg_directed_clustering",
    "ndg_core_number_normalized",
    "ndg_reciprocal_neighbor_ratio",
    "ndg_two_hop_predecessor_fraction",
    "ndg_two_hop_successor_fraction",
    "ndg_mean_predecessor_out_degree",
    "ndg_mean_successor_in_degree",
    "ndg_weak_component_fraction",
    "ndg_strong_component_fraction",
]


def structural_feature_names(edge_type_names: list[str]) -> list[str]:
    """Return stable feature order for a supplied edge-relation vocabulary."""
    if not edge_type_names or len(set(edge_type_names)) != len(edge_type_names):
        raise ValueError("edge_type_names must be non-empty and unique")
    typed = [
        feature
        for relation in edge_type_names
        for feature in (
            f"ndg_in_{relation.lower()}_log_count",
            f"ndg_out_{relation.lower()}_log_count",
        )
    ]
    return [*BASE_FEATURE_NAMES[:6], *typed, *BASE_FEATURE_NAMES[6:]]


def _relation_entropy(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=1, keepdims=True)
    probabilities = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    log_probabilities = np.zeros_like(probabilities)
    np.log(probabilities, out=log_probabilities, where=probabilities > 0)
    entropy = -(probabilities * log_probabilities).sum(axis=1)
    if counts.shape[1] > 1:
        entropy /= np.log(counts.shape[1])
    return entropy.astype(np.float32)


def _component_fraction(
    components: list[set[int]] | nx.classes.reportviews.NodeView,
    num_nodes: int,
) -> np.ndarray:
    values = np.zeros(num_nodes, dtype=np.float32)
    for component in components:
        fraction = len(component) / max(num_nodes, 1)
        for node in component:
            values[int(node)] = fraction
    return values


def _two_hop_fraction(graph: nx.DiGraph, node: int, reverse: bool) -> float:
    adjacency = graph.predecessors if reverse else graph.successors
    first_hop = set(adjacency(node))
    reached = set(first_hop)
    for neighbor in first_hop:
        reached.update(adjacency(neighbor))
    reached.discard(node)
    return len(reached) / max(graph.number_of_nodes() - 1, 1)


def extract_ndg_structural_features(
    num_nodes: int,
    edge_index: np.ndarray,
    edge_type: np.ndarray,
    edge_type_names: list[str],
    config: NDGStructuralFeatureConfig | None = None,
) -> np.ndarray:
    """Extract label-free ego, typed-relation, and global network descriptors."""
    config = config or NDGStructuralFeatureConfig()
    edge_index = np.asarray(edge_index, dtype=np.int64)
    edge_type = np.asarray(edge_type, dtype=np.int64)
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if edge_type.shape != (edge_index.shape[1],):
        raise ValueError("edge_type must align with edge_index")
    if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= num_nodes):
        raise ValueError("edge_index contains an out-of-range node")
    if edge_type.size and (edge_type.min() < 0 or edge_type.max() >= len(edge_type_names)):
        raise ValueError("edge_type contains an out-of-range relation")

    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_nodes))
    for source, target in edge_index.T:
        source_id, target_id = int(source), int(target)
        if source_id == target_id:
            continue
        if graph.has_edge(source_id, target_id):
            graph[source_id][target_id]["multiplicity"] += 1.0
        else:
            graph.add_edge(source_id, target_id, multiplicity=1.0)

    num_relations = len(edge_type_names)
    in_relation_counts = np.zeros((num_nodes, num_relations), dtype=np.float32)
    out_relation_counts = np.zeros((num_nodes, num_relations), dtype=np.float32)
    for (source, target), relation in zip(edge_index.T, edge_type, strict=True):
        if source == target:
            continue
        out_relation_counts[int(source), int(relation)] += 1.0
        in_relation_counts[int(target), int(relation)] += 1.0

    in_edge_count = in_relation_counts.sum(axis=1)
    out_edge_count = out_relation_counts.sum(axis=1)
    in_degree = np.asarray([graph.in_degree(node) for node in range(num_nodes)], dtype=np.float32)
    out_degree = np.asarray([graph.out_degree(node) for node in range(num_nodes)], dtype=np.float32)
    total_degree = np.asarray(
        [len(set(graph.predecessors(node)) | set(graph.successors(node))) for node in range(num_nodes)],
        dtype=np.float32,
    )
    degree_scale = float(max(num_nodes - 1, 1))
    isolate = ((in_degree + out_degree) == 0).astype(np.float32)
    typed_log_counts = np.stack(
        [
            values
            for relation in range(num_relations)
            for values in (
                np.log1p(in_relation_counts[:, relation]),
                np.log1p(out_relation_counts[:, relation]),
            )
        ],
        axis=1,
    ).astype(np.float32)
    in_diversity = (in_relation_counts > 0).sum(axis=1).astype(np.float32) / num_relations
    out_diversity = (out_relation_counts > 0).sum(axis=1).astype(np.float32) / num_relations

    pagerank = nx.pagerank(graph, alpha=0.85, max_iter=500, tol=1e-10, weight="multiplicity")
    reverse_pagerank = nx.pagerank(
        graph.reverse(copy=False), alpha=0.85, max_iter=500, tol=1e-10, weight="multiplicity"
    )
    betweenness_k = min(config.betweenness_samples, num_nodes)
    betweenness = nx.betweenness_centrality(
        graph,
        k=None if betweenness_k == num_nodes else betweenness_k,
        normalized=True,
        weight=None,
        seed=config.random_state,
    )
    in_closeness = nx.closeness_centrality(graph)
    out_closeness = nx.closeness_centrality(graph.reverse(copy=False))
    directed_clustering = nx.clustering(graph)
    undirected = graph.to_undirected()
    core_numbers = nx.core_number(undirected) if graph.number_of_edges() else dict.fromkeys(graph, 0)
    max_core = max(core_numbers.values(), default=0)

    reciprocal = np.zeros(num_nodes, dtype=np.float32)
    two_hop_in = np.zeros(num_nodes, dtype=np.float32)
    two_hop_out = np.zeros(num_nodes, dtype=np.float32)
    mean_predecessor_out = np.zeros(num_nodes, dtype=np.float32)
    mean_successor_in = np.zeros(num_nodes, dtype=np.float32)
    for node in range(num_nodes):
        predecessors = set(graph.predecessors(node))
        successors = set(graph.successors(node))
        neighbors = predecessors | successors
        reciprocal[node] = len(predecessors & successors) / max(len(neighbors), 1)
        two_hop_in[node] = _two_hop_fraction(graph, node, reverse=True)
        two_hop_out[node] = _two_hop_fraction(graph, node, reverse=False)
        if predecessors:
            mean_predecessor_out[node] = np.mean([graph.out_degree(value) for value in predecessors]) / degree_scale
        if successors:
            mean_successor_in[node] = np.mean([graph.in_degree(value) for value in successors]) / degree_scale

    weak_fraction = _component_fraction(list(nx.weakly_connected_components(graph)), num_nodes)
    strong_fraction = _component_fraction(list(nx.strongly_connected_components(graph)), num_nodes)
    global_features = np.column_stack(
        [
            in_diversity,
            out_diversity,
            _relation_entropy(in_relation_counts),
            _relation_entropy(out_relation_counts),
            np.log1p([num_nodes * pagerank[node] for node in range(num_nodes)]),
            np.log1p([num_nodes * reverse_pagerank[node] for node in range(num_nodes)]),
            [betweenness[node] for node in range(num_nodes)],
            [in_closeness[node] for node in range(num_nodes)],
            [out_closeness[node] for node in range(num_nodes)],
            [directed_clustering[node] for node in range(num_nodes)],
            [core_numbers[node] / max(max_core, 1) for node in range(num_nodes)],
            reciprocal,
            two_hop_in,
            two_hop_out,
            mean_predecessor_out,
            mean_successor_in,
            weak_fraction,
            strong_fraction,
        ]
    ).astype(np.float32)
    features = np.column_stack(
        [
            np.log1p(in_edge_count),
            np.log1p(out_edge_count),
            in_degree / degree_scale,
            out_degree / degree_scale,
            total_degree / degree_scale,
            isolate,
            typed_log_counts,
            global_features,
        ]
    ).astype(np.float32)
    expected_dim = len(structural_feature_names(edge_type_names))
    if features.shape != (num_nodes, expected_dim):
        raise AssertionError(f"Expected structural feature shape {(num_nodes, expected_dim)}, got {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("Structural feature extraction produced non-finite values")
    return features
