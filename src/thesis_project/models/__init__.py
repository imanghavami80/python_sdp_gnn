"""Model components for the SDP GNN pipeline."""

from thesis_project.models.ast_encoder import (
    ASTEncoderConfig,
    ASTGINEncoder,
    ASTGraphClassifier,
    AttentionPooling,
    normalize_ast_structural_features,
)
from thesis_project.models.cfg_encoder import (
    CFGEdgeAwareGATEncoder,
    CFGEncoderConfig,
    CFGGraphClassifier,
    normalize_cfg_structural_features,
)
from thesis_project.models.ndg_encoder import (
    GatedClusterMetricEncoder,
    GatedMultiViewFusion,
    NDGEncoderConfig,
    NDGMultiViewRelationalGATEncoder,
    NDGNodeClassifier,
)

__all__ = [
    "ASTEncoderConfig",
    "ASTGINEncoder",
    "ASTGraphClassifier",
    "AttentionPooling",
    "CFGEdgeAwareGATEncoder",
    "CFGEncoderConfig",
    "CFGGraphClassifier",
    "GatedMultiViewFusion",
    "GatedClusterMetricEncoder",
    "NDGEncoderConfig",
    "NDGMultiViewRelationalGATEncoder",
    "NDGNodeClassifier",
    "normalize_ast_structural_features",
    "normalize_cfg_structural_features",
]
