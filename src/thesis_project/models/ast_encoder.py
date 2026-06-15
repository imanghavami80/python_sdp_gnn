"""AST graph encoder based on GIN layers and attention pooling.

The encoder consumes the AST tensor contract produced by
`scripts/extract_promise_ast.py`:

- x: structural node features with shape [num_nodes, 3]
- node_type_id: exact AST node type ids with shape [num_nodes]
- edge_index: directed AST parent-child edges with shape [2, num_edges]
- batch: graph assignment vector with shape [num_nodes]

It returns one fixed-size embedding per AST graph. These embeddings can later be
joined with the file-level NDG nodes by `graph_id` / fully-qualified class name.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.utils import coalesce
from torch_geometric.utils import softmax


def normalize_ast_structural_features(x: Tensor) -> Tensor:
    """Normalize AST structural node features graph-by-graph.

    The raw AST extractor stores:

    - depth
    - out_degree
    - has_identifier

    Before GNN training, depth and out-degree should not be left on unrelated
    raw numeric scales. This transform keeps the binary identifier flag as-is,
    scales depth by the maximum depth inside the graph, and applies `log1p` to
    out-degree.
    """
    if x.dim() != 2 or x.size(-1) != 3:
        raise ValueError(f"Expected x with shape [num_nodes, 3], received {tuple(x.shape)}")
    if x.size(0) == 0:
        return x.float()

    normalized = x.float().clone()
    max_depth = torch.clamp(normalized[:, 0].max(), min=1.0)
    normalized[:, 0] = normalized[:, 0] / max_depth
    normalized[:, 1] = torch.log1p(torch.clamp(normalized[:, 1], min=0.0))
    normalized[:, 2] = torch.clamp(normalized[:, 2], min=0.0, max=1.0)
    return normalized


@dataclass(frozen=True)
class ASTEncoderConfig:
    """Configuration for :class:`ASTGINEncoder`."""

    num_node_types: int
    structural_feature_dim: int = 3
    node_type_embedding_dim: int = 32
    hidden_dim: int = 128
    output_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.2
    use_batch_norm: bool = True
    bidirectional_edges: bool = True

    def __post_init__(self) -> None:
        if self.num_node_types <= 0:
            raise ValueError("num_node_types must be positive")
        if self.structural_feature_dim <= 0:
            raise ValueError("structural_feature_dim must be positive")
        if self.node_type_embedding_dim <= 0:
            raise ValueError("node_type_embedding_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class AttentionPooling(nn.Module):
    """Graph-level attention pooling for batched node embeddings.

    The gate network assigns one scalar score to each node. Scores are normalized
    with a softmax within each graph in the batch, then used to compute a weighted
    sum of node embeddings. The returned weights are useful for inspecting which
    nodes influenced pooling more, but they should be treated as approximate
    indicators, not exact causal explanations.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_embeddings: Tensor, batch: Tensor) -> tuple[Tensor, Tensor]:
        scores = self.gate(node_embeddings).view(-1)
        attention = softmax(scores, batch)
        pooled = global_add_pool(node_embeddings * attention.unsqueeze(-1), batch)
        return pooled, attention


class ASTGINEncoder(nn.Module):
    """GIN + attention-pooling encoder for AST graphs.

    Node syntax is represented by a trainable node-type embedding. Compact
    structural features are concatenated after the embedding, matching the AST
    extraction design where syntax stays in node features and behavior stays in
    other graph views.
    """

    def __init__(self, config: ASTEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.node_type_embedding = nn.Embedding(config.num_node_types, config.node_type_embedding_dim)

        input_dim = config.node_type_embedding_dim + config.structural_feature_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(config.num_layers):
            mlp = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            if config.use_batch_norm:
                self.norms.append(nn.BatchNorm1d(config.hidden_dim))
            else:
                self.norms.append(nn.Identity())

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)
        self.pool = AttentionPooling(config.hidden_dim, dropout=config.dropout)
        self.output_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.output_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.output_dim, config.output_dim),
        )

    @property
    def output_dim(self) -> int:
        return self.config.output_dim

    def build_node_input(self, x: Tensor, node_type_id: Tensor) -> Tensor:
        """Concatenate trainable node-type embeddings with structural features."""
        if x.dim() != 2:
            raise ValueError(f"x must have shape [num_nodes, num_features], received {tuple(x.shape)}")
        if x.size(-1) != self.config.structural_feature_dim:
            raise ValueError(
                "Unexpected structural feature dimension: "
                f"expected {self.config.structural_feature_dim}, received {x.size(-1)}"
            )
        if node_type_id.dim() != 1:
            raise ValueError(f"node_type_id must have shape [num_nodes], received {tuple(node_type_id.shape)}")
        if node_type_id.size(0) != x.size(0):
            raise ValueError("x and node_type_id must describe the same number of nodes")

        type_embedding = self.node_type_embedding(node_type_id.long())
        return torch.cat([type_embedding, x.float()], dim=-1)

    def encode_nodes(self, x: Tensor, node_type_id: Tensor, edge_index: Tensor) -> Tensor:
        """Return contextual node embeddings before graph-level pooling."""
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"edge_index must have shape [2, num_edges], received {tuple(edge_index.shape)}")

        edge_index = edge_index.long()
        if self.config.bidirectional_edges and edge_index.numel() > 0:
            reverse_edge_index = edge_index.flip(0)
            edge_index = coalesce(torch.cat([edge_index, reverse_edge_index], dim=1), num_nodes=x.size(0))

        h = self.input_projection(self.build_node_input(x, node_type_id))
        for conv, norm in zip(self.convs, self.norms, strict=True):
            residual = h
            h = conv(h, edge_index)
            h = norm(h)
            h = self.activation(h)
            h = self.dropout(h)
            h = h + residual
        return h

    def forward(
        self,
        x: Tensor,
        node_type_id: Tensor,
        edge_index: Tensor,
        batch: Tensor | None = None,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Encode a batch of AST graphs.

        Args:
            x: Structural AST node features, shape [num_nodes, 3].
            node_type_id: AST node type ids, shape [num_nodes].
            edge_index: AST connectivity, shape [2, num_edges].
            batch: Graph id for each node. If omitted, all nodes are treated as
                one graph.
            return_attention: When true, also return one attention weight per node.
        """
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        if batch.dim() != 1 or batch.size(0) != x.size(0):
            raise ValueError("batch must have shape [num_nodes]")

        node_embeddings = self.encode_nodes(x, node_type_id, edge_index)
        pooled, attention = self.pool(node_embeddings, batch.long())
        graph_embeddings = self.output_projection(pooled)
        if return_attention:
            return graph_embeddings, attention
        return graph_embeddings


class ASTGraphClassifier(nn.Module):
    """Supervised graph classifier used to train AST embeddings."""

    def __init__(self, encoder: ASTGINEncoder, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.LayerNorm(encoder.output_dim),
            nn.Dropout(dropout),
            nn.Linear(encoder.output_dim, 1),
        )

    def encode(self, x: Tensor, node_type_id: Tensor, edge_index: Tensor, batch: Tensor | None = None) -> Tensor:
        """Return graph embeddings without applying the classifier head."""
        return self.encoder(x=x, node_type_id=node_type_id, edge_index=edge_index, batch=batch)

    def forward(self, x: Tensor, node_type_id: Tensor, edge_index: Tensor, batch: Tensor | None = None) -> Tensor:
        """Return one binary-classification logit per AST graph."""
        embeddings = self.encode(x=x, node_type_id=node_type_id, edge_index=edge_index, batch=batch)
        return self.classifier(embeddings).view(-1)
