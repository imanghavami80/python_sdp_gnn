import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from thesis_project.models import (
    CFGEdgeAwareGATEncoder,
    CFGEncoderConfig,
    normalize_cfg_structural_features,
)

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from evaluate_cfg_lopo import extract_test_embeddings
from extract_promise_cfg import (
    EDGE_TYPE_TO_ID,
    annotate_cfg_roles,
    choose_ecj_level,
    parse_tsv,
    quote_ecj_argfile_path,
)


def test_cfg_structural_feature_normalization():
    x = torch.zeros((3, 25), dtype=torch.float32)
    x[:, 0] = torch.tensor([-1.0, 0.5, 2.0])
    x[:, 1] = torch.tensor([0.0, 1.5, -3.0])
    x[:, 16] = torch.tensor([-1.0, 0.7, 2.0])
    x[:, 17] = torch.tensor([-2.0, 1.0, 3.0])
    x[:, 21] = torch.tensor([-1.0, 0.5, 2.0])
    x[:, 22] = torch.tensor([-1.0, 0.6, 2.0])
    x[:, 23] = torch.tensor([-1.0, 1.5, 0.0])
    x[:, 24] = torch.tensor([2.0, -1.0, 1.0])

    normalized = normalize_cfg_structural_features(x)

    assert torch.allclose(normalized[:, 0], torch.tensor([0.0, 0.5, 1.0]))
    assert torch.allclose(normalized[:, 1], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.allclose(normalized[:, 16], torch.tensor([0.0, 0.7, 2.0]))
    assert torch.allclose(normalized[:, 17], torch.tensor([0.0, 1.0, 3.0]))
    assert torch.allclose(normalized[:, 21], torch.tensor([0.0, 0.5, 1.0]))
    assert torch.allclose(normalized[:, 22], torch.tensor([0.0, 0.6, 1.0]))
    assert torch.allclose(normalized[:, 23], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.allclose(normalized[:, 24], torch.tensor([1.0, 0.0, 1.0]))


def test_cfg_edge_aware_gat_encoder_returns_graph_embeddings_and_attention():
    config = CFGEncoderConfig(
        num_node_types=10,
        num_stmt_kinds=13,
        num_invoke_kinds=7,
        num_edge_types=11,
        structural_feature_dim=25,
        node_type_embedding_dim=4,
        stmt_kind_embedding_dim=5,
        invoke_kind_embedding_dim=3,
        edge_type_embedding_dim=6,
        hidden_dim=16,
        output_dim=12,
        num_layers=2,
        heads=4,
        dropout=0.0,
        attention_dropout=0.0,
    )
    model = CFGEdgeAwareGATEncoder(config)

    x = torch.zeros((6, 25), dtype=torch.float32)
    x[:, 0] = torch.tensor([0.0, 0.2, 0.4, 0.0, 0.6, 1.0])
    x[:, 1] = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    x[:, 2] = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
    x[:, 3] = torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    x[:, 16] = torch.log1p(torch.tensor([0.0, 1.0, 2.0, 0.0, 1.0, 2.0]))
    x[:, 17] = torch.log1p(torch.tensor([1.0, 2.0, 1.0, 1.0, 1.0, 0.0]))
    x[:, 21] = torch.tensor([0.0, 0.25, 0.5, 0.0, 0.5, 1.0])
    x[:, 22] = torch.tensor([0.6, 0.6, 0.6, 0.4, 0.4, 0.4])
    node_type_id = torch.tensor([0, 2, 3, 0, 4, 1], dtype=torch.long)
    stmt_kind_id = torch.tensor([0, 1, 4, 0, 7, 0], dtype=torch.long)
    invoke_kind_id = torch.tensor([0, 2, 0, 0, 4, 0], dtype=torch.long)
    edge_index = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 2, 5, 1, 4, 5]], dtype=torch.long)
    edge_type = torch.tensor([0, 1, 2, 5, 6, 8], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.long)

    graph_embeddings, attention = model(
        x=x,
        node_type_id=node_type_id,
        stmt_kind_id=stmt_kind_id,
        invoke_kind_id=invoke_kind_id,
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

def test_cfg_embedding_extraction_progress_loop_uses_batch_counter():
    from torch_geometric.data import Data

    class StubModel:
        def eval(self):
            return self

        def encode(self, *args):
            return torch.ones((1, 4), dtype=torch.float32)

    batch = Data(
        x=torch.zeros((1, 25)),
        node_type_id=torch.zeros(1, dtype=torch.long),
        stmt_kind_id=torch.zeros(1, dtype=torch.long),
        invoke_kind_id=torch.zeros(1, dtype=torch.long),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_type=torch.zeros(0, dtype=torch.long),
        batch=torch.zeros(1, dtype=torch.long),
        row_idx=torch.tensor([7]),
    )

    rows, embeddings = extract_test_embeddings(StubModel(), [batch], torch.device("cpu"), output_dim=4)

    assert rows.tolist() == [7]
    assert embeddings.shape == (1, 4)


def test_ecj_argfile_quotes_source_paths_with_spaces():
    quoted = quote_ecj_argfile_path(Path("source tree/Example.java"))

    assert quoted == '"source tree/Example.java"'


def test_ecj_level_ignores_javadoc_tags_but_detects_java5_syntax(tmp_path: Path):
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    (legacy_root / "Legacy.java").write_text(
        "/** @author Example {@link String} */\n"
        "class Legacy { Object enum; void f() { enum = null; } }\n",
        encoding="utf-8",
    )
    assert choose_ecj_level(legacy_root) == "-1.3"

    modern_root = tmp_path / "modern"
    modern_root.mkdir()
    (modern_root / "Modern.java").write_text(
        "class Modern { java.util.List<String> values; }\n",
        encoding="utf-8",
    )
    assert choose_ecj_level(modern_root) == "-1.5"


def test_parse_tsv_preserves_typed_cfg_edges(tmp_path: Path):
    tsv = tmp_path / "cfg.tsv"
    tsv.write_text(
        "METHOD\texample.A\tA.m#0[p0]\tmethod\t0\t1\n"
        "NODE\texample.A\tA.m#0[p0]\t0\tENTRY\t\t1\tNO_STMT\tNO_INVOKE\t"
        + "\t".join(["0"] * 13)
        + "\t__ENTRY__\n"
        "NODE\texample.A\tA.m#0[p0]\t1\tEXIT\t\t1\tNO_STMT\tNO_INVOKE\t"
        + "\t".join(["0"] * 13)
        + "\t__EXIT__\n"
        "NODE\texample.A\tA.m#0[p0]\t2\tSTATEMENT\t1\t0\tASSIGN\tNO_INVOKE\t"
        + "\t".join(["0"] * 13)
        + "\tx = 1\n"
        "NODE\texample.A\tA.m#0[p0]\t3\tRETURN\t2\t0\tRETURN_VALUE\tNO_INVOKE\t"
        + "\t".join(["0"] * 13)
        + "\treturn x\n"
        "EDGE\texample.A\tA.m#0[p0]\t0\t2\tCFG_ENTRY\n"
        "EDGE\texample.A\tA.m#0[p0]\t2\t3\tCFG_FALLTHROUGH\n"
        "EDGE\texample.A\tA.m#0[p0]\t3\t1\tCFG_RETURN\n",
        encoding="utf-8",
    )

    parsed, failures = parse_tsv(tsv)
    graph = parsed["example.A"]

    assert failures == []
    relation_names = {edge["edge_type"] for edge in graph["graph"]["edges"]}
    assert relation_names == {"CFG_ENTRY", "CFG_FALLTHROUGH", "CFG_RETURN"}
    assert set(graph["tensors"]["edge_type"]) == {
        EDGE_TYPE_TO_ID["CFG_ENTRY"],
        EDGE_TYPE_TO_ID["CFG_FALLTHROUGH"],
        EDGE_TYPE_TO_ID["CFG_RETURN"],
    }


def test_cfg_roles_detect_loops_from_graph_cycles_not_source_order():
    nodes = [
        {"id": 0, "method_id": "m", "node_type": "STATEMENT"},
        {"id": 1, "method_id": "m", "node_type": "STATEMENT"},
        {"id": 2, "method_id": "m", "node_type": "STATEMENT"},
        {"id": 3, "method_id": "m", "node_type": "EXIT"},
    ]
    edges = [
        {"source": 0, "target": 2, "edge_type": "CFG_FALLTHROUGH"},
        {"source": 2, "target": 1, "edge_type": "CFG_GOTO"},
        {"source": 1, "target": 2, "edge_type": "CFG_BRANCH_TRUE"},
        {"source": 1, "target": 3, "edge_type": "CFG_BRANCH_FALSE"},
    ]

    annotate_cfg_roles(nodes, edges)

    assert nodes[1]["cfg_role_features"]["is_in_loop"] == 1
    assert nodes[2]["cfg_role_features"]["is_in_loop"] == 1
    assert nodes[2]["cfg_role_features"]["is_loop_header"] == 1
    assert nodes[0]["cfg_role_features"]["is_in_loop"] == 0
