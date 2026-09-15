"""Small Java counterexamples protect graph semantics independently of corpus coverage."""
import sys
from pathlib import Path

import javalang
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from extract_promise_ast import extract_filtered_tree, build_feature_matrix, NODE_TYPE_TO_ID
from extract_promise_ndg import build_type_resolver, DependencyCollector, extract_file_dependencies
from ndg_resolution import ProjectTypes
from thesis_project.features.ast_schema import literal_category, FEATURE_DIM, ROLES


def dependencies(body, extra=None):
    sources = {
        "p.A": "package p; class A { void ping() {} B next() {return null;} }",
        "p.B": "package p; class B { void ping() {} }",
        "p.C": "package p; class C { A item; B other; " + body + " }",
    }
    sources.update(extra or {})
    trees = {name: javalang.parse.parse(code) for name, code in sources.items()}
    factory = lambda tree, known: build_type_resolver(tree, known, {})
    index = ProjectTypes(trees, factory)
    collector = DependencyCollector(set(sources), type_files={k: v.file for k, v in index.types.items()})
    extract_file_dependencies("p.C", trees["p.C"], factory(trees["p.C"], set(index.types)), collector, index)
    return {t for s, t, rel in collector.relations if rel == "METHOD_CALL"}, collector


def test_field_shadowing_starts_at_declaration_and_ends_at_block():
    calls, _ = dependencies("void f() { item.ping(); { B item = null; item.ping(); } item.ping(); }")
    assert calls == {"p.A", "p.B"}
    calls, _ = dependencies("void f(B item) { item.ping(); }")
    assert calls == {"p.B"}


def test_unknown_parameter_and_local_shadow_resolvable_field():
    calls, diagnostics = dependencies("void f(External item) { item.ping(); } void g() { External item=null; item.ping(); }")
    assert calls == set()
    assert any(d.get("member") == "ping" for d in diagnostics.unresolved)


def test_explicit_this_bypasses_shadowing():
    calls, _ = dependencies("void f(B item) { this.item.ping(); }")
    assert calls == {"p.A"}


def test_return_chain_resolves_next_receiver_and_does_not_reuse_first():
    calls, _ = dependencies("void f() { item.next().ping(); }")
    assert calls == {"p.A", "p.B"}
    calls, diagnostics = dependencies("void f() { item.missing().ping(); }")
    assert calls == set()
    assert any(d.get("member") == "ping" for d in diagnostics.unresolved)


def test_chained_fields_use_terminal_field_type():
    calls, _ = dependencies("void f() { item.child.ping(); }", {
        "p.A": "package p; class A { B child; }",
    })
    assert calls == {"p.B"}


def test_for_catch_and_resource_scopes_do_not_escape():
    calls, _ = dependencies("void f() { for(B item=null; false;) {item.ping();} item.ping(); }")
    assert calls == {"p.A", "p.B"}
    calls, _ = dependencies("void f() { try {} catch(External item) {item.ping();} }")
    assert not calls
    calls, _ = dependencies("void f() { try(External item=open()) {item.ping();} item.ping(); }")
    assert calls == {"p.A"}


def test_generic_argument_is_not_receiver_type_and_arrays_are_not_components():
    calls, _ = dependencies("void f(External<B> item, B[] arr) { item.ping(); arr.ping(); }")
    assert not calls


def test_ambiguous_wildcards_and_unimported_types_are_not_guessed():
    tree = javalang.parse.parse("package p; import a.*; import b.*; class C {}")
    resolver = build_type_resolver(tree, {"a.X", "b.X", "z.Only"}, {"Only": {"z.Only"}})
    assert resolver.resolve("X") is None
    assert resolver.resolve("Only") is None
    assert resolver.resolve("a.X.missing") is None
    resolver.known_source_types.add("DefaultPackageType")
    assert resolver.resolve("DefaultPackageType") is None


def test_static_import_and_inherited_methods():
    calls, _ = dependencies("", {
        "p.A": "package p; class A { static void ping() {} }",
        "p.C": "package p; import static p.A.ping; class C {void f(){ping();}}",
    })
    assert calls == {"p.A"}


def test_method_scope_does_not_leak_and_generic_parameters_shadow_types():
    calls, _ = dependencies("void f(B item) {} void g(){item.ping();}")
    assert calls == {"p.A"}
    calls, _ = dependencies("<B> void f(B item){item.ping();}")
    assert calls == set()


def test_enhanced_for_parameter_and_field_after_loop():
    calls, _ = dependencies("void f(){for(B item: values){item.ping();} item.ping();}")
    assert calls == {"p.A", "p.B"}
    calls, _ = dependencies("void f(){synchronized(this){B item=null;} item.ping();}")
    assert calls == {"p.A"}


def test_nested_type_ownership_and_enum_initializers():
    calls, _ = dependencies("void f(){new A.Inner().ping();}", {
        "p.A": "package p; class A {static class Inner {void ping(){}}}",
    })
    assert calls == {"p.A"}
    calls, _ = dependencies("", {
        "p.C": "package p; enum C {X; C(){new B().ping();}}",
    })
    assert calls == {"p.B"}


def test_unknown_return_and_ambiguous_static_imports_do_not_guess():
    calls, diagnostics = dependencies("void f(){item.next().ping();}", {
        "p.A": "package p; class A {External next(){return null;} void ping(){}}",
    })
    assert calls == {"p.A"}
    assert any(d.get("member") == "ping" for d in diagnostics.unresolved)
    calls, _ = dependencies("", {
        "p.C": "package p; import static p.A.*; import static external.Util.*; class C {void f(){ping();}}",
        "p.A": "package p; class A {static void ping(){}}",
    })
    assert not calls
    calls, _ = dependencies("", {
        "p.C": "package p; class C extends A { void f(){ping(); super.ping();} }",
    })
    assert calls == {"p.A"}


def syntax(code):
    nodes, edges = extract_filtered_tree(javalang.parse.parse(code))
    x, types = build_feature_matrix(nodes, edges)
    return nodes, edges, x, types


@pytest.mark.parametrize("a,b", [
    ("return a+b;", "return a-b;"),
    ("return a<b;", "return a>b;"),
    ("return a;", "return ++a;"),
    ("return 0;", "return null;"),
])
def test_ast_distinguishes_operators_and_literals(a, b):
    left = syntax("class C { Object f(){" + a + "} }")[2]
    right = syntax("class C { Object f(){" + b + "} }")[2]
    assert not np.array_equal(left, right)


def test_ast_preserves_roles_and_argument_order():
    nodes, edges, x, _ = syntax("class C {void f(){if(true) a(0,1); else b();}}")
    roles = {n.role for n in nodes}
    assert {"condition", "then_statement", "else_statement", "arguments"} <= roles
    args = [n for n in nodes if n.role == "arguments"]
    assert [n.sibling_index for n in args] == [0, 1]
    assert x.shape[1] == FEATURE_DIM and np.isfinite(x).all()
    assert len(edges) == len(nodes) - 1
    assert all(int(x[n.node_id, 5]) == ROLES.index(n.role) for n in nodes)


@pytest.mark.parametrize("literal,category", [
    ("0xff", "INTEGER"), ("0x1p2", "FLOAT"), ("1L", "ONE"),
    ("1.0", "FLOAT"), ('"x"', "STRING"), ("'x'", "CHAR"), ("false", "BOOLEAN"),
])
def test_literal_categories(literal, category):
    assert literal_category(literal) == category


def test_ast_syntax_reaches_model_and_backward():
    import torch
    from thesis_project.models import ASTEncoderConfig, ASTGINEncoder
    torch.manual_seed(42)
    model = ASTGINEncoder(ASTEncoderConfig(
        num_node_types=len(NODE_TYPE_TO_ID), structural_feature_dim=FEATURE_DIM,
        hidden_dim=16, output_dim=8, num_layers=2, dropout=0.0,
    )).eval()
    outputs = []
    for op in ["+", "-"]:
        _, edges, x, types = syntax(f"class C {{ int f(){{return 1 {op} 0;}} }}")
        outputs.append(model(torch.from_numpy(x), torch.from_numpy(types), torch.tensor(edges).T))
    assert not torch.allclose(outputs[0], outputs[1])
    outputs[0].sum().backward()
    assert all(e.weight.grad is not None for e in model.syntax_embeddings)


def test_structural_features_reject_changed_source_graph(tmp_path):
    import hashlib
    import pandas as pd
    from evaluate_ndg_nested_lopo import validate_structural_provenance

    graph = tmp_path / "graph.json"
    graph.write_text('{"edges": []}', encoding="utf-8")
    ndg_index = tmp_path / "ndg.csv"
    feature_index = tmp_path / "features.csv"
    pd.DataFrame([{"dataset_name": "project", "graph_json": str(graph),
                   "resolution": "lexical_scope"}]).to_csv(ndg_index, index=False)
    pd.DataFrame([{"dataset_name": "project",
                   "source_graph_sha256": hashlib.sha256(graph.read_bytes()).hexdigest()}]).to_csv(feature_index, index=False)
    validate_structural_provenance(ndg_index, feature_index)
    graph.write_text('{"edges": [[0, 1]]}', encoding="utf-8")
    with pytest.raises(ValueError, match="Stale or mismatched"):
        validate_structural_provenance(ndg_index, feature_index)


def test_ndg_loader_requires_scope_aware_inputs(tmp_path):
    import pandas as pd
    from evaluate_ndg_nested_lopo import load_base_ndgs

    index = tmp_path / "ndg.csv"
    row = {key: "unused" for key in (
        "dataset_name", "graph_json", "x_npy", "y_npy", "edge_index_npy", "edge_type_npy"
    )}
    for resolution in (None, "unsupported"):
        candidate = dict(row)
        if resolution is not None:
            candidate["resolution"] = resolution
        pd.DataFrame([candidate]).to_csv(index, index=False)
        with pytest.raises(ValueError, match="Scope-aware NDG inputs required"):
            load_base_ndgs(index, 7)


def test_structural_features_require_source_fingerprint(tmp_path):
    import pandas as pd
    from evaluate_ndg_nested_lopo import validate_structural_provenance

    index = tmp_path / "ndg.csv"
    features = tmp_path / "features.csv"
    pd.DataFrame([{"dataset_name": "project"}]).to_csv(index, index=False)
    pd.DataFrame([{"dataset_name": "project"}]).to_csv(features, index=False)
    with pytest.raises(ValueError, match="Re-extract structural features"):
        validate_structural_provenance(index, features)
