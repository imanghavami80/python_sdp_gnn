"""Leakage-safe cluster-derived features for NDG file nodes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from thesis_project.training.ndg import ProjectGraph


@dataclass(frozen=True)
class ClusterFeatureConfig:
    """Configuration for training-only cluster feature extraction."""

    method: str = "kmeans"
    min_clusters: int = 2
    max_clusters: int = 10
    fixed_clusters: int | None = None
    silhouette_sample_size: int = 2000
    n_init: int = 20
    gmm_n_init: int = 5
    gmm_covariance_type: str = "full"
    gmm_reg_covar: float = 1e-4
    risk_smoothing: float = 20.0
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.method not in {"kmeans", "gmm"}:
            raise ValueError("method must be 'kmeans' or 'gmm'")
        if self.min_clusters < 2:
            raise ValueError("min_clusters must be at least 2")
        if self.max_clusters < self.min_clusters:
            raise ValueError("max_clusters must be at least min_clusters")
        if self.fixed_clusters is not None and self.fixed_clusters < 2:
            raise ValueError("fixed_clusters must be at least 2")
        if self.silhouette_sample_size <= 0:
            raise ValueError("silhouette_sample_size must be positive")
        if self.n_init <= 0:
            raise ValueError("n_init must be positive")
        if self.gmm_n_init <= 0:
            raise ValueError("gmm_n_init must be positive")
        if self.gmm_covariance_type not in {"full", "tied", "diag", "spherical"}:
            raise ValueError("gmm_covariance_type must be full, tied, diag, or spherical")
        if self.gmm_reg_covar <= 0:
            raise ValueError("gmm_reg_covar must be positive")
        if self.risk_smoothing <= 0:
            raise ValueError("risk_smoothing must be positive")


@dataclass
class ClusterFeatureTransformer:
    """Fitted cluster model and statistics for out-of-sample features."""

    model: KMeans | GaussianMixture
    algorithm: str
    distance_mean: np.ndarray
    distance_std: np.ndarray
    nearest_mean: float
    nearest_std: float
    temperature: np.ndarray
    selection_scores: dict[int, float]
    defect_rates: np.ndarray
    global_defect_rate: float
    risk_smoothing: float

    @property
    def num_clusters(self) -> int:
        if isinstance(self.model, KMeans):
            return int(self.model.n_clusters)
        return int(self.model.n_components)

    @property
    def feature_names(self) -> list[str]:
        return [
            *[f"cluster_distance_{index}" for index in range(self.num_clusters)],
            *[f"cluster_membership_{index}" for index in range(self.num_clusters)],
            "cluster_outlier_distance",
            "cluster_defect_risk",
        ]

    def _geometry(self, values: np.ndarray | torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        matrix = _as_finite_matrix(values)
        if isinstance(self.model, KMeans):
            distances = self.model.transform(matrix).astype(np.float64)
            scaled_distances = distances / self.temperature
            logits = -(scaled_distances**2) / 2.0
            logits -= logits.max(axis=1, keepdims=True)
            memberships = np.exp(logits)
            memberships /= memberships.sum(axis=1, keepdims=True).clip(min=1e-12)
            outlier_raw = np.log1p(scaled_distances.min(axis=1, keepdims=True))
        else:
            distances = _gmm_mahalanobis_distances(self.model, matrix)
            memberships = self.model.predict_proba(matrix).astype(np.float64)
            outlier_raw = (-self.model.score_samples(matrix)).reshape(-1, 1)
        log_distances = np.log1p(distances)
        normalized_distances = (log_distances - self.distance_mean) / self.distance_std
        outlier_distance = (outlier_raw - self.nearest_mean) / self.nearest_std
        geometry = np.concatenate(
            [normalized_distances, memberships, outlier_distance], axis=1
        ).astype(np.float32)
        return geometry, memberships

    def transform(self, values: np.ndarray | torch.Tensor) -> np.ndarray:
        """Transform unseen nodes using risk learned only from training labels."""
        geometry, memberships = self._geometry(values)
        risk = (memberships @ self.defect_rates).reshape(-1, 1)
        features = np.concatenate([geometry, risk], axis=1).astype(np.float32)
        if not np.isfinite(features).all():
            raise ValueError("Non-finite cluster features were produced")
        return features

    def transform_training(
        self,
        values: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor,
        groups: np.ndarray,
    ) -> np.ndarray:
        """Use leave-one-project-out defect rates for every training node."""
        geometry, memberships = self._geometry(values)
        label_array = _as_binary_labels(labels, len(geometry))
        group_array = np.asarray(groups)
        if group_array.shape != (len(geometry),):
            raise ValueError("groups must provide one project identifier per training node")
        unique_groups = np.unique(group_array)
        if len(unique_groups) < 2:
            raise ValueError("Cross-fitted cluster risk requires at least two training projects")
        risk = np.empty(len(geometry), dtype=np.float64)
        for group in unique_groups:
            held_out = group_array == group
            cross_fit_weights = _project_balanced_weights(
                group_array[~held_out],
                int((~held_out).sum()),
            )
            rates, _ = _smoothed_defect_rates(
                memberships[~held_out],
                label_array[~held_out],
                self.risk_smoothing,
                cross_fit_weights,
            )
            risk[held_out] = memberships[held_out] @ rates
        features = np.concatenate([geometry, risk.reshape(-1, 1)], axis=1).astype(np.float32)
        if not np.isfinite(features).all():
            raise ValueError("Non-finite cross-fitted cluster features were produced")
        return features

    def assignments(self, values: np.ndarray | torch.Tensor) -> np.ndarray:
        return self.model.predict(_as_finite_matrix(values)).astype(np.int64)

    def metadata(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "algorithm": self.algorithm,
            "num_clusters": self.num_clusters,
            "training_project_weighting": "equal total weight per project",
            "feature_names": self.feature_names,
            "selection_scores": {str(key): value for key, value in self.selection_scores.items()},
            "selection_at_search_boundary": bool(
                self.selection_scores
                and self.num_clusters
                in {min(self.selection_scores), max(self.selection_scores)}
            ),
            "selection_at_upper_search_boundary": bool(
                self.selection_scores
                and self.num_clusters == max(self.selection_scores)
            ),
            "distance_log_mean": self.distance_mean.astype(float).tolist(),
            "distance_log_std": self.distance_std.astype(float).tolist(),
            "outlier_mean": self.nearest_mean,
            "outlier_std": self.nearest_std,
            "defect_risk": {
                "smoothing": self.risk_smoothing,
                "global_training_rate": self.global_defect_rate,
                "component_rates": self.defect_rates.astype(float).tolist(),
                "training_encoding": "leave-one-project-out",
                "validation_test_encoding": "all corresponding training labels",
                "training_project_weighting": "equal total weight per project",
            },
        }
        if isinstance(self.model, KMeans):
            result.update(
                {
                    "selection_criterion": "maximum silhouette score",
                    "cluster_centers": self.model.cluster_centers_.astype(float).tolist(),
                    "soft_membership_cluster_scales": self.temperature.astype(float).tolist(),
                }
            )
        else:
            result.update(
                {
                    "selection_criterion": "minimum BIC",
                    "project_balancing": "deterministic equal-project resampling",
                    "covariance_type": self.model.covariance_type,
                    "mixture_weights": self.model.weights_.astype(float).tolist(),
                    "component_means": self.model.means_.astype(float).tolist(),
                    "component_covariances": self.model.covariances_.astype(float).tolist(),
                    "lower_bound": float(self.model.lower_bound_),
                    "converged": bool(self.model.converged_),
                    "iterations": int(self.model.n_iter_),
                }
            )
        return result


def _as_finite_matrix(values: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        matrix = values.detach().cpu().numpy()
    else:
        matrix = np.asarray(values)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError("Clustering input must have shape [at least 2 nodes, at least 1 feature]")
    if not np.isfinite(matrix).all():
        raise ValueError("Clustering input must be finite; standardize and impute metrics first")
    return matrix


def _as_binary_labels(values: np.ndarray | torch.Tensor, expected_length: int) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        labels = values.detach().cpu().numpy()
    else:
        labels = np.asarray(values)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if labels.shape != (expected_length,) or not np.isfinite(labels).all():
        raise ValueError("training_labels must contain one finite value per training node")
    if not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError("training_labels must be binary")
    return labels


def _project_balanced_weights(groups: np.ndarray, expected_length: int) -> np.ndarray:
    """Give every project equal total influence while keeping mean node weight at one."""
    group_array = np.asarray(groups)
    if group_array.shape != (expected_length,):
        raise ValueError("training_groups must provide one project identifier per training node")
    unique_groups, inverse, counts = np.unique(
        group_array,
        return_inverse=True,
        return_counts=True,
    )
    mean_project_size = expected_length / len(unique_groups)
    return (mean_project_size / counts[inverse]).astype(np.float64)


def _project_balanced_resample(
    matrix: np.ndarray,
    groups: np.ndarray,
    random_state: int,
) -> np.ndarray:
    """Create a deterministic equal-project sample for estimators without weights."""
    group_array = np.asarray(groups)
    if group_array.shape != (len(matrix),):
        raise ValueError("training_groups must provide one project identifier per training node")
    unique_groups = np.unique(group_array)
    base_count, remainder = divmod(len(matrix), len(unique_groups))
    generator = np.random.default_rng(random_state)
    selected: list[np.ndarray] = []
    for group_index, group in enumerate(unique_groups):
        candidates = np.flatnonzero(group_array == group)
        target_count = base_count + int(group_index < remainder)
        selected.append(
            generator.choice(
                candidates,
                size=target_count,
                replace=target_count > len(candidates),
            )
        )
    indices = np.concatenate(selected)
    generator.shuffle(indices)
    return matrix[indices]


def _gmm_mahalanobis_distances(model: GaussianMixture, matrix: np.ndarray) -> np.ndarray:
    """Return one covariance-aware distance per node and mixture component."""
    differences = matrix[:, None, :] - model.means_[None, :, :]
    if model.covariance_type == "full":
        squared = np.empty((len(matrix), model.n_components), dtype=np.float64)
        for component in range(model.n_components):
            solved = np.linalg.solve(
                model.covariances_[component],
                differences[:, component].T,
            ).T
            squared[:, component] = np.einsum(
                "ij,ij->i", differences[:, component], solved
            )
    elif model.covariance_type == "tied":
        solved = np.linalg.solve(
            model.covariances_,
            differences.reshape(-1, matrix.shape[1]).T,
        ).T.reshape(differences.shape)
        squared = np.einsum("nkd,nkd->nk", differences, solved)
    elif model.covariance_type == "diag":
        squared = ((differences**2) / model.covariances_[None, :, :]).sum(axis=2)
    else:
        squared = (differences**2).sum(axis=2) / model.covariances_[None, :]
    return np.sqrt(np.clip(squared, 0.0, None)).astype(np.float64)


def _smoothed_defect_rates(
    memberships: np.ndarray,
    labels: np.ndarray,
    smoothing: float,
    sample_weights: np.ndarray,
) -> tuple[np.ndarray, float]:
    if len(labels) == 0:
        raise ValueError("Cannot estimate cluster defect risk from an empty training set")
    if sample_weights.shape != labels.shape or np.any(sample_weights <= 0):
        raise ValueError("sample_weights must contain one positive value per training node")
    global_rate = float(np.average(labels, weights=sample_weights))
    effective_counts = memberships.T @ sample_weights
    defect_counts = memberships.T @ (sample_weights * labels)
    rates = (defect_counts + smoothing * global_rate) / (effective_counts + smoothing)
    return rates.astype(np.float64), global_rate


def _candidate_counts(num_nodes: int, config: ClusterFeatureConfig) -> list[int]:
    if config.fixed_clusters is not None:
        if config.fixed_clusters >= num_nodes:
            raise ValueError("fixed_clusters must be smaller than the number of training nodes")
        return [config.fixed_clusters]
    maximum = min(config.max_clusters, num_nodes - 1)
    candidates = list(range(config.min_clusters, maximum + 1))
    if not candidates:
        raise ValueError("Not enough training nodes for the requested cluster-count range")
    return candidates


def fit_cluster_features(
    training_values: np.ndarray | torch.Tensor,
    training_labels: np.ndarray | torch.Tensor,
    training_groups: np.ndarray,
    config: ClusterFeatureConfig,
) -> ClusterFeatureTransformer:
    """Fit project-balanced geometry without labels, then smoothed training risk."""
    matrix = _as_finite_matrix(training_values)
    label_array = _as_binary_labels(training_labels, len(matrix))
    if len(np.unique(np.asarray(training_groups))) < 2:
        raise ValueError("Cluster fitting requires at least two training projects")
    project_weights = _project_balanced_weights(training_groups, len(matrix))
    candidates = _candidate_counts(len(matrix), config)
    models: dict[int, KMeans | GaussianMixture] = {}
    scores: dict[int, float] = {}
    balanced_matrix: np.ndarray | None = None
    if config.method == "gmm":
        balanced_matrix = _project_balanced_resample(
            matrix,
            training_groups,
            config.random_state,
        )
    for count in candidates:
        if config.method == "kmeans":
            model: KMeans | GaussianMixture = KMeans(
                n_clusters=count,
                init="k-means++",
                n_init=config.n_init,
                random_state=config.random_state,
                algorithm="lloyd",
            )
            model.fit(matrix, sample_weight=project_weights)
            if len(candidates) > 1:
                scores[count] = float(
                    silhouette_score(
                        matrix,
                        model.labels_,
                        sample_size=min(config.silhouette_sample_size, len(matrix)),
                        random_state=config.random_state,
                    )
                )
        else:
            if balanced_matrix is None:
                raise AssertionError("Balanced GMM training matrix was not created")
            model = GaussianMixture(
                n_components=count,
                covariance_type=config.gmm_covariance_type,
                reg_covar=config.gmm_reg_covar,
                n_init=config.gmm_n_init,
                random_state=config.random_state,
                init_params="kmeans",
            )
            model.fit(balanced_matrix)
            if len(candidates) > 1:
                scores[count] = float(model.bic(balanced_matrix))
        models[count] = model

    if len(candidates) == 1:
        selected_count = candidates[0]
    elif config.method == "kmeans":
        selected_count = max(candidates, key=lambda count: (scores[count], -count))
    else:
        selected_count = min(candidates, key=lambda count: (scores[count], count))
    selected_model = models[selected_count]
    if isinstance(selected_model, KMeans):
        distances = selected_model.transform(matrix).astype(np.float64)
        assigned = selected_model.labels_.astype(np.int64)
    else:
        distances = _gmm_mahalanobis_distances(selected_model, matrix)
        assigned = selected_model.predict(matrix).astype(np.int64)
    log_distances = np.log1p(distances)
    distance_mean = log_distances.mean(axis=0)
    distance_std = log_distances.std(axis=0).clip(min=1e-6)
    if isinstance(selected_model, KMeans):
        positive_nearest = distances[np.arange(len(distances)), assigned]
        positive_nearest = positive_nearest[positive_nearest > 1e-12]
        fallback_scale = float(np.median(positive_nearest)) if len(positive_nearest) else 1.0
        temperature = np.empty(selected_count, dtype=np.float64)
        for cluster_index in range(selected_count):
            within_cluster = distances[assigned == cluster_index, cluster_index]
            within_cluster = within_cluster[within_cluster > 1e-12]
            temperature[cluster_index] = (
                float(np.median(within_cluster)) if len(within_cluster) else fallback_scale
            )
        temperature = temperature.clip(min=1e-6)
        scaled_nearest = (distances / temperature).min(axis=1)
        outlier_raw = np.log1p(scaled_nearest)
        logits = -((distances / temperature) ** 2) / 2.0
        logits -= logits.max(axis=1, keepdims=True)
        memberships = np.exp(logits)
        memberships /= memberships.sum(axis=1, keepdims=True).clip(min=1e-12)
    else:
        temperature = np.ones(selected_count, dtype=np.float64)
        outlier_raw = -selected_model.score_samples(matrix)
        memberships = selected_model.predict_proba(matrix).astype(np.float64)
    nearest_mean = float(outlier_raw.mean())
    nearest_std = float(max(outlier_raw.std(), 1e-6))
    defect_rates, global_defect_rate = _smoothed_defect_rates(
        memberships,
        label_array,
        config.risk_smoothing,
        project_weights,
    )
    return ClusterFeatureTransformer(
        model=selected_model,
        algorithm="k-means++" if config.method == "kmeans" else "gaussian-mixture",
        distance_mean=distance_mean,
        distance_std=distance_std,
        nearest_mean=nearest_mean,
        nearest_std=nearest_std,
        temperature=temperature,
        selection_scores=scores,
        defect_rates=defect_rates,
        global_defect_rate=global_defect_rate,
        risk_smoothing=config.risk_smoothing,
    )


def attach_cluster_features(
    transformer: ClusterFeatureTransformer,
    graph: ProjectGraph,
    training_groups: np.ndarray | None = None,
) -> ProjectGraph:
    """Attach a separate cluster view without modifying original metrics."""
    if training_groups is None:
        matrix = transformer.transform(graph.metrics_x)
    else:
        matrix = transformer.transform_training(graph.metrics_x, graph.y, training_groups)
    features = torch.from_numpy(matrix).to(
        device=graph.metrics_x.device,
        dtype=graph.metrics_x.dtype,
    )
    return replace(graph, cluster_x=features)
