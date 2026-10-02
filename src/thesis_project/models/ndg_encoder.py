"""Multi-view relational GAT encoder for file-level dependency graphs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch_geometric.nn import GATv2Conv


@dataclass(frozen=True)
class NDGEncoderConfig:
    """Configuration for :class:`NDGMultiViewRelationalGATEncoder`."""

    metrics_dim: int
    ast_dim: int
    cfg_dim: int
    num_edge_types: int
    ndg_structural_dim: int = 0
    hidden_dim: int = 128
    output_dim: int = 128
    edge_type_embedding_dim: int = 16
    num_layers: int = 2
    heads: int = 4
    dropout: float = 0.25
    attention_dropout: float = 0.15

    def __post_init__(self) -> None:
        for name in ("metrics_dim", "ast_dim", "cfg_dim", "num_edge_types", "hidden_dim", "output_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.ndg_structural_dim < 0:
            raise ValueError("ndg_structural_dim must be non-negative")
        if self.edge_type_embedding_dim <= 0:
            raise ValueError("edge_type_embedding_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.heads <= 0 or self.hidden_dim % self.heads != 0:
            raise ValueError("heads must be positive and divide hidden_dim")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1)")



class ViewProjection(nn.Module):
    """Map one input view into the common node representation space."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class GatedNDGStructuralAugmentation(nn.Module):
    """Add label-free NDG topology descriptors to the learned NDG view."""

    def __init__(self, structural_dim: int, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        if structural_dim <= 0:
            raise ValueError("structural_dim must be positive")
        self.projection = ViewProjection(structural_dim, embedding_dim, dropout)
        gate_hidden_dim = max(embedding_dim // 2, 1)
        self.gate = nn.Sequential(
            nn.LayerNorm(2 * embedding_dim),
            nn.Linear(2 * embedding_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 1),
            nn.Sigmoid(),
        )
        # Start close to the learned NDG-only view. Structural evidence must
        # earn a larger contribution during validation-controlled training.
        nn.init.constant_(self.gate[-2].bias, -2.0)

    def forward(self, ndg_embedding: Tensor, structural_x: Tensor) -> tuple[Tensor, Tensor]:
        structural_state = self.projection(structural_x)
        gate = self.gate(torch.cat([ndg_embedding, structural_state], dim=-1))
        return ndg_embedding + gate * structural_state, gate


class GatedMultiViewFusion(nn.Module):
    """Weight and combine three available file-level representations."""

    NUM_VIEWS = 3

    def __init__(self, metrics_dim: int, ast_dim: int, cfg_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            [
                ViewProjection(metrics_dim, hidden_dim, dropout),
                ViewProjection(ast_dim, hidden_dim, dropout),
                ViewProjection(cfg_dim, hidden_dim, dropout),
            ]
        )
        self.view_embeddings = nn.Parameter(torch.empty(self.NUM_VIEWS, hidden_dim))
        nn.init.normal_(self.view_embeddings, std=0.02)
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 1)),
            nn.Tanh(),
            nn.Linear(max(hidden_dim // 2, 1), 1),
        )
        self.combine = nn.Sequential(
            nn.Linear(self.NUM_VIEWS * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        metrics_x: Tensor,
        ast_x: Tensor,
        cfg_x: Tensor,
        view_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if view_mask.shape != (metrics_x.size(0), self.NUM_VIEWS):
            raise ValueError(
                f"view_mask must have shape {(metrics_x.size(0), self.NUM_VIEWS)}, received {tuple(view_mask.shape)}"
            )
        if not torch.all(view_mask[:, 0]):
            raise ValueError("The primary view must be available for every NDG node")

        inputs = (metrics_x, ast_x, cfg_x)
        projected = torch.stack(
            [projection(view) for projection, view in zip(self.projections, inputs, strict=True)],
            dim=1,
        )
        projected = projected + self.view_embeddings.unsqueeze(0)
        gate_logits = self.gate(projected).squeeze(-1)
        gate_logits = gate_logits.masked_fill(~view_mask.bool(), torch.finfo(gate_logits.dtype).min)
        view_weights = torch.softmax(gate_logits, dim=1)
        weighted_views = projected * view_weights.unsqueeze(-1)
        fused = self.combine(weighted_views.flatten(start_dim=1))
        return fused, view_weights


class NDGMultiViewRelationalGATEncoder(nn.Module):
    """Return one gated relational GAT embedding per NDG file node."""

    def __init__(self, config: NDGEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.metrics_projection = ViewProjection(
            config.metrics_dim, config.hidden_dim, config.dropout
        )
        self.edge_type_embedding = nn.Embedding(config.num_edge_types, config.edge_type_embedding_dim)
        head_dim = config.hidden_dim // config.heads
        self.convs = nn.ModuleList(
            [
                GATv2Conv(
                    in_channels=config.hidden_dim,
                    out_channels=head_dim,
                    heads=config.heads,
                    concat=True,
                    dropout=config.attention_dropout,
                    edge_dim=config.edge_type_embedding_dim,
                    add_self_loops=False,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(config.hidden_dim) for _ in range(config.num_layers)])
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
        self.output_projection = nn.Sequential(
            nn.Linear((config.num_layers + 1) * config.hidden_dim, config.output_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.output_dim),
        )
        self.structural_augmentation = (
            GatedNDGStructuralAugmentation(
                config.ndg_structural_dim,
                config.output_dim,
                config.dropout,
            )
            if config.ndg_structural_dim
            else None
        )
        self.late_fusion = GatedMultiViewFusion(
            metrics_dim=config.output_dim,
            ast_dim=config.ast_dim,
            cfg_dim=config.cfg_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        self.late_output_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.output_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.output_dim),
        )

    @property
    def output_dim(self) -> int:
        return self.config.output_dim

    def forward(
        self,
        metrics_x: Tensor,
        ndg_structural_x: Tensor,
        ast_x: Tensor,
        cfg_x: Tensor,
        view_mask: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Return one contextual embedding per NDG file node."""
        num_nodes = metrics_x.size(0)
        if metrics_x.shape != (num_nodes, self.config.metrics_dim):
            raise ValueError("metrics_x has an unexpected shape")
        if ndg_structural_x.shape != (num_nodes, self.config.ndg_structural_dim):
            raise ValueError("ndg_structural_x has an unexpected shape")
        if ast_x.shape != (num_nodes, self.config.ast_dim):
            raise ValueError("ast_x has an unexpected shape")
        if cfg_x.shape != (num_nodes, self.config.cfg_dim):
            raise ValueError("cfg_x has an unexpected shape")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if edge_type.shape != (edge_index.size(1),):
            raise ValueError("edge_type must have shape [num_edges]")
        if edge_type.numel() and (edge_type.min() < 0 or edge_type.max() >= self.config.num_edge_types):
            raise ValueError("edge_type contains an id outside the configured vocabulary")

        if not torch.all(view_mask[:, 0]):
            raise ValueError("The metrics view must be available for every NDG node")
        metric_state = self.metrics_projection(metrics_x)
        # AST and CFG enter only after independent NDG message passing.
        h = metric_state
        view_weights = metrics_x.new_zeros((num_nodes, 3))
        layer_outputs = [h]
        edge_attr = self.edge_type_embedding(edge_type.long())
        attention: dict[str, Tensor] = {
            "view_weights": view_weights.detach(),
        }

        for layer_index, (conv, norm) in enumerate(zip(self.convs, self.norms, strict=True)):
            residual = h
            if return_attention and layer_index == len(self.convs) - 1:
                updated, (attention_edge_index, edge_attention) = conv(
                    h,
                    edge_index.long(),
                    edge_attr=edge_attr,
                    return_attention_weights=True,
                )
                attention["edge_index"] = attention_edge_index.detach()
                attention["edge_attention"] = edge_attention.detach()
            else:
                updated = conv(h, edge_index.long(), edge_attr=edge_attr)
            h = norm(residual + self.dropout(self.activation(updated)))
            layer_outputs.append(h)

        ndg_embeddings = self.output_projection(torch.cat(layer_outputs, dim=-1))
        if self.structural_augmentation is not None:
            ndg_embeddings, structural_gate = self.structural_augmentation(
                ndg_embeddings, ndg_structural_x
            )
        else:
            structural_gate = metrics_x.new_zeros((num_nodes, 1))
        attention["ndg_structural_gate"] = structural_gate.detach()
        fused, view_weights = self.late_fusion(ndg_embeddings, ast_x, cfg_x, view_mask)
        embeddings = self.late_output_projection(fused)
        attention["view_weights"] = view_weights.detach()
        attention["ndg_embeddings"] = ndg_embeddings.detach()
        if return_attention:
            return embeddings, attention
        return embeddings


class NDGNodeClassifier(nn.Module):
    """Node-level binary defect classifier around the NDG encoder."""

    def __init__(self, encoder: NDGMultiViewRelationalGATEncoder, dropout: float = 0.25) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder.output_dim, 1),
        )

    def encode(self, **kwargs: Tensor) -> Tensor:
        return self.encoder(**kwargs)

    def forward(self, **kwargs: Tensor) -> Tensor:
        return self.classifier(self.encode(**kwargs)).view(-1)
