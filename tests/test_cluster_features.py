import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")
pytest.importorskip("hdbscan")

from thesis_project.training import (
    ClusterFeatureConfig,
    ProjectGraph,
    attach_cluster_features,
    fit_cluster_features,
)


def make_graph(metrics: np.ndarray, labels: list[float] | None = None) -> ProjectGraph:
    count = len(metrics)
    return ProjectGraph(
        dataset_name="sample",
        names=[str(index) for index in range(count)],
        source_paths=[str(index) for index in range(count)],
        metrics_x=torch.tensor(metrics, dtype=torch.float32),
        ast_x=torch.zeros(count, 2),
        cfg_x=torch.zeros(count, 2),
        view_mask=torch.ones(count, 3, dtype=torch.bool),
        loss_weight=torch.ones(count),
        y=torch.tensor(labels if labels is not None else [0.0] * count),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_type=torch.zeros(0, dtype=torch.long),
    )


def separated_metrics() -> np.ndarray:
    rng = np.random.default_rng(7)
    return np.concatenate(
        [rng.normal(-3.0, 0.15, size=(30, 3)), rng.normal(3.0, 0.15, size=(30, 3))]
    ).astype(np.float32)


def two_project_groups() -> np.ndarray:
    return np.asarray(["project_a"] * 30 + ["project_b"] * 30)


def test_cluster_features_have_distances_memberships_and_outlier_score() -> None:
    labels = np.asarray([0.0, 1.0] * 30)
    transformer = fit_cluster_features(
        separated_metrics(),
        labels,
        two_project_groups(),
        ClusterFeatureConfig(fixed_clusters=2, random_state=3),
    )

    features = transformer.transform(separated_metrics())

    assert features.shape == (60, 6)
    assert np.isfinite(features).all()
    assert np.allclose(features[:, 2:4].sum(axis=1), 1.0, atol=1e-6)
    assert transformer.feature_names[-2:] == ["cluster_outlier_distance", "cluster_defect_risk"]


def test_automatic_cluster_count_uses_training_structure() -> None:
    transformer = fit_cluster_features(
        separated_metrics(),
        np.asarray([0.0, 1.0] * 30),
        two_project_groups(),
        ClusterFeatureConfig(min_clusters=2, max_clusters=4, silhouette_sample_size=60),
    )

    assert transformer.num_clusters == 2
    assert set(transformer.selection_scores) == {2, 3, 4}


def test_append_preserves_labels_edges_and_original_metrics() -> None:
    metrics = separated_metrics()
    graph = make_graph(metrics, labels=[0.0, 1.0] * 30)
    transformer = fit_cluster_features(
        metrics, graph.y, two_project_groups(), ClusterFeatureConfig(fixed_clusters=2)
    )

    augmented = attach_cluster_features(transformer, graph)

    assert augmented.metrics_x.shape == (60, 3)
    assert augmented.cluster_x is not None
    assert augmented.cluster_x.shape == (60, 6)
    assert torch.equal(augmented.metrics_x, graph.metrics_x)
    assert torch.equal(augmented.y, graph.y)
    assert torch.equal(augmented.edge_index, graph.edge_index)


def test_transforming_test_data_does_not_refit_training_centers() -> None:
    train = separated_metrics()
    transformer = fit_cluster_features(
        train,
        np.asarray([0.0, 1.0] * 30),
        two_project_groups(),
        ClusterFeatureConfig(fixed_clusters=2),
    )
    centers_before = transformer.model.cluster_centers_.copy()

    transformer.transform(np.full((4, 3), 1000.0, dtype=np.float32))

    assert np.array_equal(transformer.model.cluster_centers_, centers_before)


def test_clustering_rejects_non_finite_unprocessed_metrics() -> None:
    metrics = separated_metrics()
    metrics[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        fit_cluster_features(
            metrics,
            np.asarray([0.0, 1.0] * 30),
            two_project_groups(),
            ClusterFeatureConfig(fixed_clusters=2),
        )


@pytest.mark.parametrize("method", ["kmeans", "gmm"])
def test_training_defect_risk_is_leave_one_project_out(method: str) -> None:
    metrics = separated_metrics()
    labels = np.asarray([0.0] * 30 + [1.0] * 30)
    groups = np.asarray(["clean_project"] * 30 + ["defect_project"] * 30)
    graph = make_graph(metrics, labels=labels.tolist())
    transformer = fit_cluster_features(
        metrics,
        labels,
        groups,
        ClusterFeatureConfig(method=method, fixed_clusters=2, gmm_n_init=1),
    )

    training_graph = attach_cluster_features(transformer, graph, training_groups=groups)
    inference_graph = attach_cluster_features(transformer, graph)

    assert training_graph.cluster_x is not None
    assert inference_graph.cluster_x is not None
    training_risk = training_graph.cluster_x[:, -1].numpy()
    inference_risk = inference_graph.cluster_x[:, -1].numpy()
    assert np.all(training_risk[:30] > 0.99)
    assert np.all(training_risk[30:] < 0.01)
    assert not np.allclose(training_risk, inference_risk)


def test_cluster_risk_gives_projects_equal_total_weight() -> None:
    metrics = separated_metrics()
    labels = np.asarray([0.0] * 10 + [1.0] * 50)
    groups = np.asarray(["small_clean"] * 10 + ["large_defective"] * 50)

    transformer = fit_cluster_features(
        metrics,
        labels,
        groups,
        ClusterFeatureConfig(fixed_clusters=2),
    )

    assert transformer.global_defect_rate == pytest.approx(0.5)
    assert (
        transformer.metadata()["defect_risk"]["training_project_weighting"]
        == "equal total weight per project"
    )


@pytest.mark.parametrize("covariance_type", ["full", "tied", "diag", "spherical"])
def test_gmm_features_are_finite_for_supported_covariances(covariance_type: str) -> None:
    metrics = separated_metrics()
    transformer = fit_cluster_features(
        metrics,
        np.asarray([0.0, 1.0] * 30),
        two_project_groups(),
        ClusterFeatureConfig(
            method="gmm",
            fixed_clusters=2,
            gmm_n_init=1,
            gmm_covariance_type=covariance_type,
            random_state=11,
        ),
    )

    features = transformer.transform(metrics)

    assert features.shape == (60, 6)
    assert np.isfinite(features).all()
    assert np.allclose(features[:, 2:4].sum(axis=1), 1.0, atol=1e-6)
    assert transformer.metadata()["algorithm"] == "gaussian-mixture"
    assert transformer.metadata()["selection_criterion"] == "minimum BIC"


def test_gmm_automatic_component_count_uses_bic() -> None:
    transformer = fit_cluster_features(
        separated_metrics(),
        np.asarray([0.0, 1.0] * 30),
        two_project_groups(),
        ClusterFeatureConfig(
            method="gmm",
            min_clusters=2,
            max_clusters=4,
            gmm_n_init=1,
            random_state=13,
        ),
    )

    assert transformer.num_clusters == 2
    assert set(transformer.selection_scores) == {2, 3, 4}
    assert transformer.num_clusters == min(
        transformer.selection_scores,
        key=transformer.selection_scores.get,
    )


def test_hdbscan_produces_fixed_density_features_and_metadata() -> None:
    transformer = fit_cluster_features(
        separated_metrics(),
        np.asarray([0.0, 1.0] * 30),
        two_project_groups(),
        ClusterFeatureConfig(
            method="hdbscan",
            hdbscan_min_cluster_sizes=(5, 10, 15),
            hdbscan_min_samples=3,
            random_state=17,
        ),
    )

    features = transformer.transform(separated_metrics())
    metadata = transformer.metadata()

    assert features.shape == (60, 6)
    assert np.isfinite(features).all()
    assert transformer.density_min_cluster_size in {5, 10, 15}
    assert metadata["algorithm"] == "hdbscan"
    assert metadata["selection_criterion"] == "maximum relative DBCV"
    assert len(metadata["feature_names"]) == 6


def test_hdbscan_all_noise_falls_back_to_finite_features() -> None:
    transformer = fit_cluster_features(
        separated_metrics(),
        np.asarray([0.0, 1.0] * 30),
        two_project_groups(),
        ClusterFeatureConfig(
            method="hdbscan",
            hdbscan_min_cluster_sizes=(59,),
            hdbscan_min_samples=3,
            random_state=19,
        ),
    )

    features = transformer.transform(separated_metrics())

    assert transformer.num_clusters == 0
    assert features.shape == (60, 6)
    assert np.isfinite(features).all()
    assert np.all(transformer.assignments(separated_metrics()) == -1)
