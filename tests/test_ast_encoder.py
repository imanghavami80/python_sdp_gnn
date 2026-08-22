import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from thesis_project.models import ASTEncoderConfig, ASTGINEncoder, normalize_ast_structural_features


def test_ast_structural_feature_normalization():
    x = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [5.0, 2.0, 1.0],
            [10.0, 8.0, 1.0],
        ],
        dtype=torch.float32,
    )

    normalized = normalize_ast_structural_features(x)

    assert torch.allclose(normalized[:, 0], torch.tensor([0.0, 0.5, 1.0]))
    assert torch.allclose(normalized[:, 1], torch.log1p(torch.tensor([0.0, 2.0, 8.0])))
    assert torch.allclose(normalized[:, 2], torch.tensor([0.0, 1.0, 1.0]))


def test_ast_gin_encoder_returns_graph_embeddings():
    config = ASTEncoderConfig(
        num_node_types=8,
        structural_feature_dim=3,
        node_type_embedding_dim=4,
        hidden_dim=16,
        output_dim=12,
        num_layers=2,
        dropout=0.0,
    )
    model = ASTGINEncoder(config)

    x = torch.tensor(
        [
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    node_type_id = torch.tensor([0, 1, 2, 0, 3], dtype=torch.long)
    edge_index = torch.tensor([[0, 0, 3], [1, 2, 4]], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)

    graph_embeddings, attention = model(x, node_type_id, edge_index, batch=batch, return_attention=True)

    assert graph_embeddings.shape == (2, config.output_dim)
    assert attention.shape == (x.size(0),)
    assert torch.isfinite(graph_embeddings).all()
    assert torch.isfinite(attention).all()
    assert torch.allclose(attention[batch == 0].sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(attention[batch == 1].sum(), torch.tensor(1.0), atol=1e-6)
