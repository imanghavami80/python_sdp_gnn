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
import numpy as np

from evaluate_ndg_within_project import parse_args, common_scenario as scenario_name
from ndg_common import prepare_output_dir, validate_cfg_inputs
from extract_promise_cfg import EDGE_TYPE_TO_ID
from thesis_project.training import (
    ProjectGraph,
    binary_metrics,
    select_f1_threshold,
)


def make_config(
    ndg_structural_dim: int = 0,
) -> NDGEncoderConfig:
    return NDGEncoderConfig(
        metrics_dim=20,
        ast_dim=16,
        cfg_dim=16,
        num_edge_types=14,
        ndg_structural_dim=ndg_structural_dim,
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
        "ndg_structural_x": torch.empty(5, 0),
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


def test_late_fusion_keeps_ndg_encoding_independent_of_ast_and_cfg() -> None:
    encoder = NDGMultiViewRelationalGATEncoder(make_config())
    encoder.eval()
    inputs = graph_inputs()
    _, first_attention = encoder(**inputs, return_attention=True)
    inputs["ast_x"] = inputs["ast_x"] * 100.0
    inputs["cfg_x"] = inputs["cfg_x"] * -100.0
    _, second_attention = encoder(**inputs, return_attention=True)

    assert torch.allclose(
        first_attention["ndg_embeddings"], second_attention["ndg_embeddings"], atol=1e-6
    )


def test_ndg_structural_features_use_a_separate_bounded_gate() -> None:
    encoder = NDGMultiViewRelationalGATEncoder(
        make_config(ndg_structural_dim=6)
    )
    encoder.eval()
    inputs = graph_inputs()
    inputs["ndg_structural_x"] = torch.randn(5, 6)

    embeddings, attention = encoder(**inputs, return_attention=True)

    assert embeddings.shape == (5, 24)
    assert attention["ndg_structural_gate"].shape == (5, 1)
    assert torch.all(attention["ndg_structural_gate"] >= 0.0)
    assert torch.all(attention["ndg_structural_gate"] <= 1.0)


def test_metrics_view_is_required() -> None:
    encoder = NDGMultiViewRelationalGATEncoder(make_config())
    inputs = graph_inputs()
    inputs["view_mask"][0, 0] = False
    with pytest.raises(ValueError, match="metrics view"):
        encoder(**inputs)














def test_validation_threshold_maximizes_f1_without_forcing_half() -> None:
    labels = torch.tensor([0, 0, 1, 1]).numpy()
    probabilities = torch.tensor([0.10, 0.20, 0.35, 0.40]).numpy()

    threshold = select_f1_threshold(labels, probabilities)

    assert threshold == pytest.approx(0.35)


def test_g_mean_uses_sensitivity_and_specificity() -> None:
    metrics = binary_metrics(
        torch.tensor([0, 0, 1, 1]).numpy(),
        torch.tensor([0.1, 0.9, 0.8, 0.7]).numpy(),
    )

    assert metrics["g_mean"] == pytest.approx(0.5 ** 0.5)


@pytest.mark.parametrize("structural", [False, True])
def test_all_scenarios_train_and_preserve_late_fusion(structural: bool) -> None:
    config = make_config(ndg_structural_dim=6 if structural else 0)
    model = NDGNodeClassifier(NDGMultiViewRelationalGATEncoder(config), dropout=0.0)
    inputs = graph_inputs()
    inputs["ndg_structural_x"] = torch.randn(5, config.ndg_structural_dim)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(**inputs), torch.ones(5))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    before, info = model.encoder(**inputs, return_attention=True)
    inputs["ast_x"] *= 20
    inputs["cfg_x"] *= -20
    _, changed = model.encoder(**inputs, return_attention=True)
    assert torch.allclose(info["ndg_embeddings"], changed["ndg_embeddings"])
    assert torch.isfinite(before).all()
    options = []
    if structural:
        options.append("--ndg-structural-features")
    args = parse_args(options)
    assert args.ndg_structural_features == structural
    assert scenario_name(args).endswith("on" if structural else "off")


def test_removed_fusion_switch_is_rejected() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--fusion-stage", "early"])


def test_existing_results_are_not_overwritten(tmp_path: Path) -> None:
    artifact = tmp_path / "results.json"
    artifact.write_text("keep")
    with pytest.raises(FileExistsError):
        prepare_output_dir(tmp_path)
    assert artifact.read_text() == "keep"


def test_cfg_construction_and_exact_relations_are_required(tmp_path: Path) -> None:
    index = tmp_path / "graph_index.csv"
    pd.DataFrame({"construction": ["exceptional_control_flow"]}).to_csv(index, index=False)
    validate_cfg_inputs(index, EDGE_TYPE_TO_ID)
    with pytest.raises(ValueError, match="CFG"):
        validate_cfg_inputs(index, {**EDGE_TYPE_TO_ID, "DATA_DEPENDENCE": 99})
    pd.DataFrame({"construction": ["unsupported"]}).to_csv(index, index=False)
    with pytest.raises(ValueError, match="CFG"):
        validate_cfg_inputs(index, EDGE_TYPE_TO_ID)
