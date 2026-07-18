import pytest
import sys
from pathlib import Path

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from thesis_project.models import NDGEncoderConfig, NDGMultiViewRelationalGATEncoder, NDGNodeClassifier

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import pandas as pd

from evaluate_ndg_nested_lopo import assert_outer_boundary, project_indices


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
