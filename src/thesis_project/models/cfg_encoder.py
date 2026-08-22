"""Edge-aware GAT encoder and attention pooling for file-level CFGs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch_geometric.nn import GATv2Conv

from thesis_project.models.ast_encoder import AttentionPooling

CFG_NUMERIC_FEATURE_DIM = 25
CFG_UNIT_INTERVAL_FEATURE_INDICES = [0, 21, 22, 23, 24]
CFG_BINARY_FEATURE_INDICES = list(range(1, 16)) + [18, 19, 20, 23, 24]
CFG_NON_NEGATIVE_FEATURE_INDICES = [16, 17]


def normalize_cfg_structural_features(x: Tensor) -> Tensor:
    """Clamp CFG numeric node features to their expected ranges."""
    if x.dim() != 2 or x.size(-1) != CFG_NUMERIC_FEATURE_DIM:
        raise ValueError(f"Expected x with shape [num_nodes, {CFG_NUMERIC_FEATURE_DIM}], received {tuple(x.shape)}")
    if x.size(0) == 0:
        return x.float()

    normalized = x.float().clone()
    for index in CFG_UNIT_INTERVAL_FEATURE_INDICES:
        normalized[:, index] = torch.clamp(normalized[:, index], min=0.0, max=1.0)
    for index in CFG_BINARY_FEATURE_INDICES:
        normalized[:, index] = torch.clamp(normalized[:, index], min=0.0, max=1.0)
    for index in CFG_NON_NEGATIVE_FEATURE_INDICES:
        normalized[:, index] = torch.clamp(normalized[:, index], min=0.0)
    return normalized


@dataclass(frozen=True)
class CFGEncoderConfig:
    """Configuration for :class:`CFGEdgeAwareGATEncoder`."""

    num_node_types: int
    num_stmt_kinds: int
    num_invoke_kinds: int
    num_edge_types: int
    structural_feature_dim: int = CFG_NUMERIC_FEATURE_DIM
    node_type_embedding_dim: int = 32
    stmt_kind_embedding_dim: int = 16
    invoke_kind_embedding_dim: int = 8
    edge_type_embedding_dim: int = 16
    hidden_dim: int = 128
    output_dim: int = 128
    num_layers: int = 3
    heads: int = 4
    dropout: float = 0.2
    attention_dropout: float = 0.2
    use_layer_norm: bool = True
    add_self_loops: bool = False

    def __post_init__(self) -> None:
        if self.num_node_types <= 0:
            raise ValueError("num_node_types must be positive")
        if self.num_stmt_kinds <= 0:
            raise ValueError("num_stmt_kinds must be positive")
        if self.num_invoke_kinds <= 0:
            raise ValueError("num_invoke_kinds must be positive")
        if self.num_edge_types <= 0:
            raise ValueError("num_edge_types must be positive")
        if self.structural_feature_dim <= 0:
            raise ValueError("structural_feature_dim must be positive")
        if self.node_type_embedding_dim <= 0:
            raise ValueError("node_type_embedding_dim must be positive")
        if self.stmt_kind_embedding_dim <= 0:
            raise ValueError("stmt_kind_embedding_dim must be positive")
        if self.invoke_kind_embedding_dim <= 0:
            raise ValueError("invoke_kind_embedding_dim must be positive")
        if self.edge_type_embedding_dim <= 0:
            raise ValueError("edge_type_embedding_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.heads <= 0:
            raise ValueError("heads must be positive")
        if self.hidden_dim % self.heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1)")


class CFGEdgeAwareGATEncoder(nn.Module):
    """Return one edge-aware GAT embedding per file-level CFG."""

    def __init__(self, config: CFGEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.node_type_embedding = nn.Embedding(config.num_node_types, config.node_type_embedding_dim)
        self.stmt_kind_embedding = nn.Embedding(config.num_stmt_kinds, config.stmt_kind_embedding_dim)
        self.invoke_kind_embedding = nn.Embedding(config.num_invoke_kinds, config.invoke_kind_embedding_dim)
        self.edge_type_embedding = nn.Embedding(config.num_edge_types, config.edge_type_embedding_dim)

        input_dim = (
            config.node_type_embedding_dim
            + config.stmt_kind_embedding_dim
            + config.invoke_kind_embedding_dim
            + config.structural_feature_dim
        )
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        head_dim = config.hidden_dim // config.heads
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(config.num_layers):
            self.convs.append(
                GATv2Conv(
                    in_channels=config.hidden_dim,
                    out_channels=head_dim,
                    heads=config.heads,
                    concat=True,
                    dropout=config.attention_dropout,
                    edge_dim=config.edge_type_embedding_dim,
                    add_self_loops=config.add_self_loops,
                )
            )
            self.norms.append(nn.LayerNorm(config.hidden_dim) if config.use_layer_norm else nn.Identity())

        self.activation = nn.ELU()
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

    def build_node_input(self, x: Tensor, node_type_id: Tensor, stmt_kind_id: Tensor, invoke_kind_id: Tensor) -> Tensor:
        """Concatenate categorical embeddings with numeric CFG node features."""
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
        if stmt_kind_id.dim() != 1:
            raise ValueError(f"stmt_kind_id must have shape [num_nodes], received {tuple(stmt_kind_id.shape)}")
        if stmt_kind_id.size(0) != x.size(0):
            raise ValueError("x and stmt_kind_id must describe the same number of nodes")
        if invoke_kind_id.dim() != 1:
            raise ValueError(f"invoke_kind_id must have shape [num_nodes], received {tuple(invoke_kind_id.shape)}")
        if invoke_kind_id.size(0) != x.size(0):
            raise ValueError("x and invoke_kind_id must describe the same number of nodes")

        type_embedding = self.node_type_embedding(node_type_id.long())
        stmt_embedding = self.stmt_kind_embedding(stmt_kind_id.long())
        invoke_embedding = self.invoke_kind_embedding(invoke_kind_id.long())
        return torch.cat([type_embedding, stmt_embedding, invoke_embedding, x.float()], dim=-1)

    def build_edge_attr(self, edge_type: Tensor, num_edges: int) -> Tensor:
        """Return trainable edge-type embeddings used by GAT attention."""
        if edge_type.dim() != 1:
            raise ValueError(f"edge_type must have shape [num_edges], received {tuple(edge_type.shape)}")
        if edge_type.size(0) != num_edges:
            raise ValueError("edge_type length must match edge_index edge count")
        return self.edge_type_embedding(edge_type.long())

    def encode_nodes(
        self,
        x: Tensor,
        node_type_id: Tensor,
        stmt_kind_id: Tensor,
        invoke_kind_id: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        return_edge_attention: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Return contextual CFG node embeddings before graph-level pooling."""
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"edge_index must have shape [2, num_edges], received {tuple(edge_index.shape)}")

        edge_index = edge_index.long()
        edge_attr = self.build_edge_attr(edge_type, edge_index.size(1))
        h = self.input_projection(self.build_node_input(x, node_type_id, stmt_kind_id, invoke_kind_id))

        attention_info: dict[str, Tensor] = {}
        for layer_idx, (conv, norm) in enumerate(zip(self.convs, self.norms, strict=True)):
            residual = h
            if return_edge_attention and layer_idx == len(self.convs) - 1:
                h, (att_edge_index, edge_attention) = conv(
                    h,
                    edge_index,
                    edge_attr=edge_attr,
                    return_attention_weights=True,
                )
                attention_info = {
                    "edge_index": att_edge_index.detach(),
                    "edge_attention": edge_attention.detach(),
                }
            else:
                h = conv(h, edge_index, edge_attr=edge_attr)
            h = norm(h)
            h = self.activation(h)
            h = self.dropout(h)
            h = h + residual

        if return_edge_attention:
            return h, attention_info
        return h

    def forward(
        self,
        x: Tensor,
        node_type_id: Tensor,
        stmt_kind_id: Tensor,
        invoke_kind_id: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        batch: Tensor | None = None,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Encode a batch and optionally return node and edge attention."""
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        if batch.dim() != 1 or batch.size(0) != x.size(0):
            raise ValueError("batch must have shape [num_nodes]")

        encoded = self.encode_nodes(
            x=x,
            node_type_id=node_type_id,
            stmt_kind_id=stmt_kind_id,
            invoke_kind_id=invoke_kind_id,
            edge_index=edge_index,
            edge_type=edge_type,
            return_edge_attention=return_attention,
        )
        if return_attention:
            node_embeddings, attention_info = encoded
        else:
            node_embeddings = encoded
            attention_info = {}

        pooled, node_attention = self.pool(node_embeddings, batch.long())
        graph_embeddings = self.output_projection(pooled)
        if return_attention:
            attention_info = dict(attention_info)
            attention_info["node_attention"] = node_attention.detach()
            return graph_embeddings, attention_info
        return graph_embeddings


class CFGGraphClassifier(nn.Module):
    """Supervised graph classifier used to train CFG embeddings."""

    def __init__(self, encoder: CFGEdgeAwareGATEncoder, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.LayerNorm(encoder.output_dim),
            nn.Dropout(dropout),
            nn.Linear(encoder.output_dim, 1),
        )

    def encode(
        self,
        x: Tensor,
        node_type_id: Tensor,
        stmt_kind_id: Tensor,
        invoke_kind_id: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        batch: Tensor | None = None,
    ) -> Tensor:
        """Return graph embeddings without applying the classifier head."""
        return self.encoder(
            x=x,
            node_type_id=node_type_id,
            stmt_kind_id=stmt_kind_id,
            invoke_kind_id=invoke_kind_id,
            edge_index=edge_index,
            edge_type=edge_type,
            batch=batch,
        )

    def forward(
        self,
        x: Tensor,
        node_type_id: Tensor,
        stmt_kind_id: Tensor,
        invoke_kind_id: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        batch: Tensor | None = None,
    ) -> Tensor:
        """Return one binary-classification logit per CFG graph."""
        embeddings = self.encode(
            x=x,
            node_type_id=node_type_id,
            stmt_kind_id=stmt_kind_id,
            invoke_kind_id=invoke_kind_id,
            edge_index=edge_index,
            edge_type=edge_type,
            batch=batch,
        )
        return self.classifier(embeddings).view(-1)
