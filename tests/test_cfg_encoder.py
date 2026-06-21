import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from thesis_project.models import CFGEdgeAwareGATEncoder, CFGEncoderConfig, normalize_cfg_structural_features


def test_cfg_structural_feature_normalization():
    x = torch.tensor(
        [
            [-1.0, 0.0, 0.0],
            [0.5, 1.5, 1.0],
            [2.0, 1.0, -3.0],
        ],
        dtype=torch.float32,
    )

    normalized = normalize_cfg_structural_features(x)

    assert torch.allclose(normalized[:, 0], torch.tensor([0.0, 0.5, 1.0]))
    assert torch.allclose(normalized[:, 1], torch.tensor([0.0, 1.0, 1.0]))
    assert torch.allclose(normalized[:, 2], torch.tensor([0.0, 1.0, 0.0]))


def test_cfg_edge_aware_gat_encoder_returns_graph_embeddings_and_attention():
    config = CFGEncoderConfig(
        num_node_types=10,
        num_edge_types=5,
        structural_feature_dim=3,
        node_type_embedding_dim=4,
        edge_type_embedding_dim=6,
        hidden_dim=16,
        output_dim=12,
        num_layers=2,
        heads=4,
        dropout=0.0,
        attention_dropout=0.0,
    )
    model = CFGEdgeAwareGATEncoder(config)

    x = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.2, 1.0, 0.0],
            [0.4, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.6, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    node_type_id = torch.tensor([0, 2, 3, 0, 4, 1], dtype=torch.long)
    edge_index = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 2, 5, 1, 4, 5]], dtype=torch.long)
    edge_type = torch.tensor([0, 1, 2, 0, 3, 4], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.long)

    graph_embeddings, attention = model(
        x=x,
        node_type_id=node_type_id,
        edge_index=edge_index,
        edge_type=edge_type,
        batch=batch,
        return_attention=True,
    )

    assert graph_embeddings.shape == (2, config.output_dim)
    assert attention["node_attention"].shape == (x.size(0),)
    assert attention["edge_index"].shape == edge_index.shape
    assert attention["edge_attention"].shape == (edge_index.size(1), config.heads)
    assert torch.isfinite(graph_embeddings).all()
    assert torch.isfinite(attention["node_attention"]).all()
    assert torch.isfinite(attention["edge_attention"]).all()
    assert torch.allclose(attention["node_attention"][batch == 0].sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(attention["node_attention"][batch == 1].sum(), torch.tensor(1.0), atol=1e-6)
