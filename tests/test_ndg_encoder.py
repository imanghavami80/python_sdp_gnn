import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from thesis_project.models import NDGEncoderConfig, NDGMultiViewRelationalGATEncoder, NDGNodeClassifier

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import pandas as pd

from evaluate_ndg_nested_lopo import assert_outer_boundary, choose_validation_project, project_indices
from thesis_project.training import ProjectGraph, standardize_metrics


def make_config() -> NDGEncoderConfig:
    return NDGEncoderConfig(
        metrics_dim=20,
        ast_dim=16,
        cfg_dim=16,
        num_edge_types=14,
        hidden_dim=32,
        output_dim=24,
        edge_type_embedding_dim=8,
        num_layers=2,
        heads=4,
        dropout=0.0,
        attention_dropout=0.0,
    )


def graph_inputs() -> dict[str, torch.Tensor]:
    return {
        "metrics_x": torch.randn(5, 20),
        "ast_x": torch.randn(5, 16),
        "cfg_x": torch.randn(5, 16),
        "view_mask": torch.tensor(
            [[1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=torch.bool
        ),
        "edge_index": torch.tensor([[0, 1, 2, 3, 1, 2], [1, 2, 3, 4, 0, 1]], dtype=torch.long),
        "edge_type": torch.tensor([0, 1, 2, 3, 7, 8], dtype=torch.long),
    }


def test_ndg_encoder_returns_one_embedding_per_file() -> None:
    encoder = NDGMultiViewRelationalGATEncoder(make_config())
    encoder.eval()
    embeddings, attention = encoder(**graph_inputs(), return_attention=True)

    assert embeddings.shape == (5, 24)
    assert attention["view_weights"].shape == (5, 3)
    assert torch.allclose(attention["view_weights"].sum(dim=1), torch.ones(5), atol=1e-6)
    assert attention["view_weights"][1, 2].item() == 0.0
    assert "edge_attention" in attention


def test_ndg_classifier_returns_node_logits() -> None:
    model = NDGNodeClassifier(NDGMultiViewRelationalGATEncoder(make_config()), dropout=0.0)
    logits = model(**graph_inputs())
    assert logits.shape == (5,)


def test_metrics_view_is_required() -> None:
    encoder = NDGMultiViewRelationalGATEncoder(make_config())
    inputs = graph_inputs()
    inputs["view_mask"][0, 0] = False
    with pytest.raises(ValueError, match="metrics view"):
        encoder(**inputs)


def test_outer_project_is_excluded_from_training_indices() -> None:
    index = pd.DataFrame({"dataset_name": ["ant", "ant", "camel", "ivy"]})
    train_indices = project_indices(index, ["ant", "camel"])
    assert_outer_boundary(index, train_indices, "ivy", "test stage")


def test_outer_boundary_rejects_test_project_leakage() -> None:
    index = pd.DataFrame({"dataset_name": ["ant", "camel", "ivy"]})
    leaking_indices = project_indices(index, ["ant", "ivy"])
    with pytest.raises(AssertionError, match="Leakage guard failed"):
        assert_outer_boundary(index, leaking_indices, "ivy", "test stage")


def test_validation_project_requires_both_classes() -> None:
    projects = {
        "balanced": SimpleNamespace(y=torch.tensor([0, 1] * 6).numpy()),
        "mostly_positive": SimpleNamespace(y=torch.tensor([0] + [1] * 20).numpy()),
        "other": SimpleNamespace(y=torch.tensor([0] * 7 + [1] * 5).numpy()),
    }

    selected = choose_validation_project(projects, list(projects), min_class_nodes=5)

    assert selected in {"balanced", "other"}


def test_metric_transform_is_fit_on_training_nodes_only() -> None:
    def graph(values: list[list[float]]) -> ProjectGraph:
        count = len(values)
        return ProjectGraph(
            dataset_name="sample",
            names=[str(i) for i in range(count)],
            source_paths=[str(i) for i in range(count)],
            metrics_x=torch.tensor(values),
            ast_x=torch.zeros(count, 1),
            cfg_x=torch.zeros(count, 1),
            view_mask=torch.ones(count, 3, dtype=torch.bool),
            y=torch.zeros(count),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_type=torch.zeros(0, dtype=torch.long),
        )

    train, test = standardize_metrics(graph([[1.0], [3.0], [float("nan")]]), graph([[100.0]]))

    assert torch.isfinite(train.metrics_x).all()
    assert torch.allclose(train.metrics_x.mean(dim=0), torch.zeros(1), atol=1e-6)
    assert test.metrics_x.item() > 50.0
