"""CFG graph encoder based on edge-aware GAT layers and attention pooling.

The encoder consumes the CFG tensor contract produced by
`scripts/extract_promise_cfg.py`:

- x: compact statement-level node features with shape [num_nodes, 3]
- node_type_id: CFG node type ids with shape [num_nodes]
- edge_index: directed CFG edges with shape [2, num_edges]
- edge_type: CFG edge type ids with shape [num_edges]
- batch: graph assignment vector with shape [num_nodes]

It returns one fixed-size embedding per file-level CFG graph. CFG behavior is
represented primarily by typed directed edges, so edge-type embeddings are fed
directly into GATv2 attention.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch_geometric.nn import GATv2Conv

from thesis_project.models.ast_encoder import AttentionPooling


def normalize_cfg_structural_features(x: Tensor) -> Tensor:
    """Clamp CFG structural node features to their expected ranges.

    The CFG extractor stores:

    - normalized source-line position in the file
    - has_source_line flag
    - is_synthetic flag

    These features are already bounded by construction. This transform keeps the
    model robust to accidental numeric drift while preserving their meaning.
    """
    if x.dim() != 2 or x.size(-1) != 3:
        raise ValueError(f"Expected x with shape [num_nodes, 3], received {tuple(x.shape)}")
    if x.size(0) == 0:
        return x.float()

    normalized = x.float().clone()
    normalized[:, 0] = torch.clamp(normalized[:, 0], min=0.0, max=1.0)
    normalized[:, 1] = torch.clamp(normalized[:, 1], min=0.0, max=1.0)
    normalized[:, 2] = torch.clamp(normalized[:, 2], min=0.0, max=1.0)
    return normalized


@dataclass(frozen=True)
class CFGEncoderConfig:
    """Configuration for :class:`CFGEdgeAwareGATEncoder`."""

    num_node_types: int
    num_edge_types: int
    structural_feature_dim: int = 3
    node_type_embedding_dim: int = 32
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
        if self.num_edge_types <= 0:
            raise ValueError("num_edge_types must be positive")
        if self.structural_feature_dim <= 0:
            raise ValueError("structural_feature_dim must be positive")
        if self.node_type_embedding_dim <= 0:
            raise ValueError("node_type_embedding_dim must be positive")
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
    """Relational/edge-aware GAT encoder for CFG graphs.

    Node syntax is represented by trainable CFG node-type embeddings and compact
    structural features. Control-flow behavior is represented by trainable
    edge-type embeddings that are used inside GATv2 attention, allowing edges
    such as `CFG_TRUE`, `CFG_FALSE`, `CFG_RETURN`, and `CFG_EXCEPTION` to receive
    different learned importance.
    """

    def __init__(self, config: CFGEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.node_type_embedding = nn.Embedding(config.num_node_types, config.node_type_embedding_dim)
        self.edge_type_embedding = nn.Embedding(config.num_edge_types, config.edge_type_embedding_dim)

        input_dim = config.node_type_embedding_dim + config.structural_feature_dim
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

    def build_node_input(self, x: Tensor, node_type_id: Tensor) -> Tensor:
        """Concatenate trainable node-type embeddings with CFG node features."""
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
        edge_index: Tensor,
        edge_type: Tensor,
        return_edge_attention: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Return contextual CFG node embeddings before graph-level pooling."""
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"edge_index must have shape [2, num_edges], received {tuple(edge_index.shape)}")

        edge_index = edge_index.long()
        edge_attr = self.build_edge_attr(edge_type, edge_index.size(1))
        h = self.input_projection(self.build_node_input(x, node_type_id))

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
        edge_index: Tensor,
        edge_type: Tensor,
        batch: Tensor | None = None,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Encode a batch of CFG graphs.

        Args:
            x: CFG node features, shape [num_nodes, 3].
            node_type_id: CFG node type ids, shape [num_nodes].
            edge_index: Directed CFG connectivity, shape [2, num_edges].
            edge_type: CFG edge type ids, shape [num_edges].
            batch: Graph id for each node. If omitted, all nodes are treated as
                one graph.
            return_attention: When true, also return node pooling attention and
                last-layer edge attention. These values are useful for
                approximate inspection, not exact causal explanation.
        """
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        if batch.dim() != 1 or batch.size(0) != x.size(0):
            raise ValueError("batch must have shape [num_nodes]")

        encoded = self.encode_nodes(
            x=x,
            node_type_id=node_type_id,
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
        edge_index: Tensor,
        edge_type: Tensor,
        batch: Tensor | None = None,
    ) -> Tensor:
        """Return graph embeddings without applying the classifier head."""
        return self.encoder(
            x=x,
            node_type_id=node_type_id,
            edge_index=edge_index,
            edge_type=edge_type,
            batch=batch,
        )

    def forward(
        self,
        x: Tensor,
        node_type_id: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        batch: Tensor | None = None,
    ) -> Tensor:
        """Return one binary-classification logit per CFG graph."""
        embeddings = self.encode(
            x=x,
            node_type_id=node_type_id,
            edge_index=edge_index,
            edge_type=edge_type,
            batch=batch,
        )
        return self.classifier(embeddings).view(-1)
