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
import evaluate_ndg_nested_lopo as pipeline

from evaluate_ndg_nested_lopo import assert_outer_boundary, choose_validation_project, project_indices
from evaluate_ndg_nested_lopo import parse_args, scenario_name, prepare_output_dir, validate_cfg_inputs
from extract_promise_cfg import EDGE_TYPE_TO_ID
from thesis_project.training import (
    ProjectGraph,
    binary_metrics,
    combine_graphs,
    select_f1_threshold,
    standardize_ndg_structural_features,
    standardize_metrics,
)


def make_config(
    cluster_mode: str = "none",
    cluster_dim: int = 0,
    ndg_structural_dim: int = 0,
) -> NDGEncoderConfig:
    return NDGEncoderConfig(
        metrics_dim=20,
        ast_dim=16,
        cfg_dim=16,
        num_edge_types=14,
        cluster_dim=cluster_dim,
        ndg_structural_dim=ndg_structural_dim,
        hidden_dim=32,
        output_dim=24,
        edge_type_embedding_dim=8,
        num_layers=2,
        heads=4,
        dropout=0.0,
        attention_dropout=0.0,
        cluster_mode=cluster_mode,
    )


def graph_inputs() -> dict[str, torch.Tensor]:
    return {
        "metrics_x": torch.randn(5, 20),
        "cluster_x": torch.empty(5, 0),
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
            loss_weight=torch.ones(count),
            y=torch.zeros(count),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_type=torch.zeros(0, dtype=torch.long),
        )

    train, test = standardize_metrics(graph([[1.0], [3.0], [float("nan")]]), graph([[100.0]]))

    assert torch.isfinite(train.metrics_x).all()
    assert torch.allclose(train.metrics_x.mean(dim=0), torch.zeros(1), atol=1e-6)
    assert test.metrics_x.item() > 50.0


def test_ndg_structural_transform_is_fit_on_training_nodes_only() -> None:
    def graph(values: list[list[float]]) -> ProjectGraph:
        count = len(values)
        return ProjectGraph(
            dataset_name="sample",
            names=[str(i) for i in range(count)],
            source_paths=[str(i) for i in range(count)],
            metrics_x=torch.zeros(count, 1),
            ast_x=torch.zeros(count, 1),
            cfg_x=torch.zeros(count, 1),
            view_mask=torch.ones(count, 3, dtype=torch.bool),
            loss_weight=torch.ones(count),
            y=torch.zeros(count),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_type=torch.zeros(0, dtype=torch.long),
            ndg_structural_x=torch.tensor(values),
        )

    train, test = standardize_ndg_structural_features(
        graph([[1.0], [3.0]]), graph([[100.0]])
    )

    assert train.ndg_structural_x is not None
    assert test.ndg_structural_x is not None
    assert torch.allclose(train.ndg_structural_x.mean(dim=0), torch.zeros(1), atol=1e-6)
    assert test.ndg_structural_x.item() > 50.0


def test_combined_projects_have_equal_total_loss_weight() -> None:
    def graph(count: int, name: str) -> ProjectGraph:
        return ProjectGraph(
            dataset_name=name,
            names=[str(i) for i in range(count)],
            source_paths=[str(i) for i in range(count)],
            metrics_x=torch.zeros(count, 1),
            ast_x=torch.zeros(count, 1),
            cfg_x=torch.zeros(count, 1),
            view_mask=torch.ones(count, 3, dtype=torch.bool),
            loss_weight=torch.ones(count),
            y=torch.zeros(count),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_type=torch.zeros(0, dtype=torch.long),
        )

    combined = combine_graphs([graph(2, "small"), graph(8, "large")])

    assert torch.allclose(combined.loss_weight[:2].sum(), combined.loss_weight[2:].sum())


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


@pytest.mark.parametrize("mode", ["none", "simple"])
@pytest.mark.parametrize("structural", [False, True])
def test_all_scenarios_train_and_preserve_late_fusion(mode: str, structural: bool) -> None:
    config = make_config(mode, cluster_dim=6 if mode != "none" else 0,
                         ndg_structural_dim=6 if structural else 0)
    model = NDGNodeClassifier(NDGMultiViewRelationalGATEncoder(config), dropout=0.0)
    inputs = graph_inputs()
    inputs["cluster_x"] = torch.randn(5, config.cluster_dim)
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
    if mode != "none":
        inputs["cluster_x"] += torch.randn_like(inputs["cluster_x"])
        _, changed = model.encoder(**inputs, return_attention=True)
        assert not torch.allclose(info["ndg_embeddings"], changed["ndg_embeddings"])
    assert torch.isfinite(before).all()
    options = ["--cluster-mode", mode]
    if structural:
        options.append("--ndg-structural-features")
    args = parse_args(options)
    assert args.cluster_mode == mode and args.ndg_structural_features == structural
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


def test_disabled_structural_features_survive_combining_graphs() -> None:
    inputs = graph_inputs()
    graph = ProjectGraph(
        dataset_name="sample", names=list("abcde"), source_paths=list("abcde"),
        y=torch.zeros(5), loss_weight=torch.ones(5), **inputs,
    )
    train, test = standardize_ndg_structural_features(combine_graphs([graph]), graph)
    assert train.ndg_structural_x is None
    assert test.ndg_structural_x.shape == (5, 0)


@pytest.mark.parametrize("mode", ["none", "simple"])
@pytest.mark.parametrize("structural", [False, True])
def test_nested_fold_feature_integration_and_saved_scenario(
    mode: str, structural: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise real feature fitting, selection, retraining and prediction writes.

    Upstream encoders supply small fixed embeddings to keep this integration
    test independent of Java inputs and expensive AST/CFG training.
    """
    rng = np.random.default_rng(42)
    args = parse_args(["--cluster-mode", mode, "--ndg-epochs", "1", "--patience", "1",
                       "--embedding-dim", "16", "--hidden-dim", "32"])
    args.ndg_structural_features = structural
    if mode != "none":
        args.cluster_count = 2
        args.cluster_n_init = 1
    projects = {}
    lookup = {}
    for name in ["a", "b", "c", "test"]:
        names = [f"{name}_{i}" for i in range(12)]
        projects[name] = pipeline.BaseNDGProject(
            dataset_name=name, names=names, source_paths=names,
            metrics_x=rng.normal(size=(12, 20)).astype(np.float32),
            y=np.array([0, 1] * 6, dtype=np.float32),
            edge_index=np.array([list(range(11)), list(range(1, 12))]),
            edge_type=np.zeros(11, dtype=np.int64),
            ndg_structural_x=rng.normal(size=(12, 6 if structural else 0)).astype(np.float32),
        )
        for node in names:
            lookup[(name, node)] = rng.normal(size=16).astype(np.float32)
    config = make_config(ndg_structural_dim=6 if structural else 0)

    def upstream(*unused):
        return lookup, lookup, torch.nn.Linear(1, 1), config, {
            "history": [], "best_info": {"best_epoch": 1},
        }

    monkeypatch.setattr(pipeline, "train_ast_fold", upstream)
    monkeypatch.setattr(pipeline, "train_cfg_fold", upstream)
    row, predictions, embeddings = pipeline.run_outer_fold(
        0, "test", list(projects), projects, pd.DataFrame(), {}, pd.DataFrame(),
        ({}, {}, {}, {}), [], [str(i) for i in range(6)] if structural else [],
        config, args, torch.device("cpu"), tmp_path,
    )
    assert len(predictions) == len(embeddings) == 12
    assert np.isfinite(predictions.probability).all()
    assert row["cluster_mode"] == mode
    assert predictions.ndg_structural_gate.notna().all() if structural else predictions.ndg_structural_gate.isna().all()
    checkpoint = torch.load(tmp_path / "folds/test/ndg_encoder.pt", weights_only=False)
    assert checkpoint["encoder_config"]["cluster_mode"] == mode
    assert checkpoint["encoder_config"]["ndg_structural_dim"] == (6 if structural else 0)
    assert checkpoint["fusion"] == "late" and checkpoint["cfg_construction"] == "exceptional_control_flow"
