#!/usr/bin/env python3
"""Within-project, transductive AST + CFG + NDG defect prediction."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import ndg_common as common
import experiment_audit as audit
from thesis_project.models import NDGEncoderConfig
from thesis_project.training.ndg import binary_metrics, make_model, model_inputs
from thesis_project.training.within_project import (
    InsufficientClassSupport, split_files, scale_from_training, train_masked,
)

PROTOCOL = "within_project_transductive_grouped_holdout"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run separate within-project AST + CFG + NDG models (transductive file split).")
    parser.add_argument("--ndg-index", type=Path, default=Path("outputs/promise/ndg/graph_index.csv"))
    parser.add_argument("--ndg-edge-vocab", type=Path, default=Path("outputs/promise/ndg/edge_type_vocab.json"))
    parser.add_argument("--ndg-feature-names", type=Path, default=Path("outputs/promise/ndg/feature_names.json"))
    parser.add_argument(
        "--ndg-structural-index",
        type=Path,
        default=Path("outputs/promise/ndg_structural/feature_index.csv"),
    )
    parser.add_argument(
        "--ndg-structural-feature-names",
        type=Path,
        default=Path("outputs/promise/ndg_structural/feature_names.json"),
    )
    parser.add_argument("--ast-index", type=Path, default=Path("outputs/promise/ast/graph_index.csv"))
    parser.add_argument("--ast-node-vocab", type=Path, default=Path("outputs/promise/ast/node_type_vocab.json"))
    parser.add_argument("--cfg-index", type=Path, default=Path("outputs/promise/cfg/graph_index.csv"))
    parser.add_argument("--cfg-node-vocab", type=Path, default=Path("outputs/promise/cfg/node_type_vocab.json"))
    parser.add_argument("--cfg-stmt-vocab", type=Path, default=Path("outputs/promise/cfg/stmt_kind_vocab.json"))
    parser.add_argument("--cfg-invoke-vocab", type=Path, default=Path("outputs/promise/cfg/invoke_kind_vocab.json"))
    parser.add_argument("--cfg-edge-vocab", type=Path, default=Path("outputs/promise/cfg/edge_type_vocab.json"))
    parser.add_argument("--cfg-feature-names", type=Path, default=Path("outputs/promise/cfg/feature_names.json"))
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: outputs/promise/within_project/<scenario>/seed_<seed>. Must be empty.")
    parser.add_argument("--upstream-epochs", type=int, default=50)
    parser.add_argument("--ndg-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--ast-batch-size", type=int, default=32)
    parser.add_argument("--cfg-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--ast-layers", type=int, default=3)
    parser.add_argument("--cfg-layers", type=int, default=3)
    parser.add_argument("--ndg-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--attention-dropout", type=float, default=0.15)
    parser.add_argument(
        "--ndg-structural-features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use label-free handcrafted NDG topology descriptors after message passing.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true", help="Audit every selected project and save the run manifest without training.")
    parser.add_argument("--require-all-projects", action="store_true", help="Fail before training if any selected project is ineligible.")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True,
                        help="Require deterministic PyTorch algorithms; unsupported operations fail explicitly.")
    parser.add_argument(
        "--include-ast-fallbacks",
        action="store_true",
        help="Use coarse fallback ASTs. By default they are masked as an unavailable AST view.",
    )
    parser.add_argument("--project", action="append", default=[], help="Run only this project; repeat to select several.")
    args = parser.parse_args(argv)
    if min(args.upstream_epochs, args.ndg_epochs, args.patience, args.ast_batch_size, args.cfg_batch_size) <= 0:
        parser.error("Epochs, patience and batch sizes must be positive")
    if not 0 < args.validation_fraction < 1 or not 0 < args.test_fraction < 1 or args.validation_fraction + args.test_fraction >= 1:
        parser.error("Validation/test fractions must be positive and sum to less than one")
    return args


def preflight(projects, selected, ast, cfg, args):
    """Assess every project before training; extraction validity and split support differ."""
    rows, splits = [], {}
    for name in selected:
        base = projects[name]
        counts = np.bincount(base.y.astype(int), minlength=2)
        row = dict(project=name, nodes=len(base.y), clean_nodes=int(counts[0]),
                   defective_nodes=int(counts[1]), status='eligible', reason='')
        try:
            split = split_files(base.names, base.source_paths, base.y, args.seed,
                                args.validation_fraction, args.test_fraction)
            for view, index in [('ast', ast), ('cfg', cfg)]:
                _, parts = upstream_indices(index, base, split)
                row[f'{view}_available_nodes'] = sum(map(len, parts.values()))
            for part, indices in split.items():
                cc = np.bincount(base.y[indices].astype(int), minlength=2)
                row.update({f'{part}_nodes': len(indices), f'{part}_clean': int(cc[0]),
                            f'{part}_defective': int(cc[1])})
            splits[name] = split
        except InsufficientClassSupport as exc:
            row.update(status='ineligible', reason=str(exc))
        rows.append(row)
    return pd.DataFrame(rows), splits

def upstream_indices(index, base, split):
    """Align file views to NDG identities, and hide test labels before any loader."""
    index = index[index.dataset_name.astype(str).eq(base.dataset_name) & index.name.astype(str).isin(base.names)].copy().reset_index(drop=True)
    if index.name.duplicated().any():
        raise ValueError('Duplicate file identities in upstream index')
    by_name = dict(zip(base.names, base.y, strict=True))
    if any(float(row.label) != float(by_name[str(row['name'])]) for _, row in index.iterrows()):
        raise ValueError('Upstream and NDG labels disagree')
    parts = {key: np.flatnonzero(index.name.isin([base.names[i] for i in rows]).to_numpy()) for key, rows in split.items()}
    for key in ['train', 'validation']:
        if len(np.unique(index.iloc[parts[key]].label)) < 2:
            raise InsufficientClassSupport(f'Available upstream {key} graphs lack both classes')
    index.loc[parts['test'], 'label'] = 0  # Test labels are not given to upstream code, even during encoding.
    return index, parts


def train_upstream(view, index, parts, vocab, feature_names, args, device):
    pipeline = common.ast_pipeline if view == 'ast' else common.cfg_pipeline
    stage = common.ast_args(args) if view == 'ast' else common.cfg_args(args)
    dataset_cls = pipeline.ASTGraphDataset if view == 'ast' else pipeline.CFGGraphDataset
    if view == 'ast':
        model, config = common.build_ast_model(stage, len(vocab), device)
    else:
        model, config = pipeline.build_model(stage, *vocab, feature_names, device)
    train_loader = pipeline.make_loader(dataset_cls(index, parts['train'], normalize_structural_features=True),
                                        stage.batch_size, shuffle=True, num_workers=stage.num_workers)
    val_loader = pipeline.make_loader(dataset_cls(index, parts['validation'], normalize_structural_features=True),
                                      stage.batch_size, shuffle=False, num_workers=stage.num_workers)
    state, history, best = pipeline.train_model(model, train_loader, val_loader,
                                               index.label.to_numpy(), parts['train'], stage, device)
    model.load_state_dict(state)
    model.eval()
    encode = common.encode_ast if view == 'ast' else common.encode_cfg
    lookup = encode(model, index, np.arange(len(index)), stage, device)
    return lookup, dict(model_state_dict=state, encoder_config=asdict(config), selected_epochs=best['best_epoch'],
                        protocol=PROTOCOL, refit_after_selection=False), history


def run_project(base, split, ast_data, ast_vocab, cfg_data, cfg_vocabs, cfg_features,
                config, args, device, directory):
    started = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    # Check both views before spending time training either encoder.
    ai, ap = upstream_indices(ast_data, base, split)
    ci, cp = upstream_indices(cfg_data, base, split)
    rows = []
    for part, indices in split.items():
        for i in indices:
            rows.append(dict(dataset_name=base.dataset_name, name=base.names[i], source_path=base.source_paths[i],
                             node_index=int(i), split=part, label=int(base.y[i])))
    pd.DataFrame(rows).to_csv(directory/'split.csv', index=False)
    metadata = dict(protocol=PROTOCOL, seed=args.seed, grouping='resolved Java source path',
                    full_unlabeled_graph_visible=True, test_labels_used_for_training=False,
                    refit_after_selection=False, counts={k: len(v) for k, v in split.items()},
                    class_counts={k: np.bincount(base.y[v].astype(int), minlength=2).tolist() for k,v in split.items()})
    (directory/'split.json').write_text(json.dumps(metadata, indent=2))
    common.set_seed(args.seed)
    stage_started = time.perf_counter()
    ast, ast_checkpoint, ah = train_upstream('ast', ai, ap, ast_vocab, [], args, device)
    ast_seconds = time.perf_counter()-stage_started
    common.set_seed(args.seed+1000)
    stage_started = time.perf_counter()
    cfg, cfg_checkpoint, ch = train_upstream('cfg', ci, cp, cfg_vocabs, cfg_features, args, device)
    cfg_seconds = time.perf_counter()-stage_started
    graph = common.assemble_ndgs({base.dataset_name: base}, ast, cfg, args.embedding_dim, [base.dataset_name])[base.dataset_name]
    graph, preprocessing = scale_from_training(graph, split['train'])
    # Expose no test labels to loss/epoch/threshold code.
    hidden_y = graph.y.clone()
    hidden_y[split['test']] = float('nan')
    graph = replace(graph, y=hidden_y).to(device)
    common.set_seed(args.seed+2000)
    model = make_model(config, device)
    stage_started = time.perf_counter()
    epoch, threshold, history = train_masked(model, graph, split['train'], split['validation'], common.ndg_args(args))
    ndg_seconds = time.perf_counter()-stage_started
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(**model_inputs(graph))
        probabilities = torch.sigmoid(model.classifier(embeddings).view(-1)).cpu().numpy()
    test = split['test']
    majority = float(np.mean(base.y[split['train']]) >= .5)
    predictions = pd.DataFrame(dict(dataset_name=base.dataset_name, name=np.array(base.names)[test],
                                    source_path=np.array(base.source_paths)[test], label=base.y[test].astype(int),
                                    probability=probabilities[test], prediction=(probabilities[test] >= threshold).astype(int),
                                    decision_threshold=threshold, baseline_probability=majority,
                                    baseline_prediction=int(majority)))
    predictions.to_csv(directory/'test_node_predictions.csv', index=False)
    val = split['validation']
    pd.DataFrame(dict(name=np.array(base.names)[val], label=base.y[val].astype(int), probability=probabilities[val])).to_csv(directory/'validation_predictions.csv', index=False)
    np.save(directory/'test_node_embeddings.npy', embeddings[test].cpu().numpy())
    for name, h in [('ast', ah), ('cfg', ch), ('ndg', history)]:
        pd.DataFrame(h).to_csv(directory/f'{name}_selection_history.csv', index=False)
    for name, checkpoint in [('ast', ast_checkpoint), ('cfg', cfg_checkpoint)]:
        torch.save(checkpoint, directory/f'{name}_encoder.pt')
    torch.save(dict(protocol=PROTOCOL, encoder_config=asdict(config), model_state_dict=model.cpu().state_dict(),
                    selected_epochs=epoch, decision_threshold=threshold, preprocessing=preprocessing,
                    node_names=base.names, split={k: v.tolist() for k,v in split.items()},
                    model_inputs={k: v.detach().cpu() for k,v in model_inputs(graph).items()},
                    refit_after_selection=False), directory/'ndg_encoder.pt')
    metrics = binary_metrics(base.y[test], probabilities[test], threshold)
    baseline = binary_metrics(base.y[test], np.full(len(test), majority))
    return dict(project=base.dataset_name, train_nodes=len(split['train']), validation_nodes=len(val), test_nodes=len(test),
                ast_selected_epochs=ast_checkpoint['selected_epochs'], cfg_selected_epochs=cfg_checkpoint['selected_epochs'],
                ndg_selected_epochs=epoch, decision_threshold=threshold,
                ast_epochs_run=len(ah), cfg_epochs_run=len(ch), ndg_epochs_run=len(history),
                ast_seconds=ast_seconds, cfg_seconds=cfg_seconds, ndg_seconds=ndg_seconds,
                elapsed_seconds=time.perf_counter()-started,
                **{f'model_{k}': v for k,v in metrics.items()}, **{f'baseline_{k}': v for k,v in baseline.items()}), predictions


def main():
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    args = parse_args()
    if args.deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(args.deterministic)
    torch.backends.cudnn.benchmark = False
    device = common.choose_device(args.device)
    output = common.resolve_path(args.output_dir or Path('outputs/promise/within_project')/common_scenario(args)/f'seed_{args.seed}')
    ndg_vocab = common.load_json(common.resolve_path(args.ndg_edge_vocab))
    metric_names = common.load_json(common.resolve_path(args.ndg_feature_names))
    structural, structural_names = None, []
    if args.ndg_structural_features:
        common.validate_structural_provenance(common.resolve_path(args.ndg_index), common.resolve_path(args.ndg_structural_index))
        structural, structural_names = common.load_ndg_structural_matrices(common.resolve_path(args.ndg_structural_index), common.resolve_path(args.ndg_structural_feature_names))
    projects = common.load_base_ndgs(common.resolve_path(args.ndg_index), len(ndg_vocab), structural)
    selected = args.project or sorted(projects)
    if len(set(selected)) != len(selected) or set(selected)-set(projects):
        raise ValueError('Requested projects must be unique and exist in the NDG index')
    ast, ast_vocab = common.ast_pipeline.load_inputs(common.resolve_path(args.ast_index), common.resolve_path(args.ast_node_vocab))
    args.ast_feature_dim = int(ast.feature_dim.iloc[0])
    if not args.include_ast_fallbacks:
        ast = ast[ast.parser_mode.astype(str) != 'fallback'].reset_index(drop=True)
    cfg, nv, sv, iv, ev, cfg_features, _ = common.cfg_pipeline.load_inputs(*[common.resolve_path(p) for p in
        [args.cfg_index, args.cfg_node_vocab, args.cfg_stmt_vocab, args.cfg_invoke_vocab, args.cfg_edge_vocab, args.cfg_feature_names]])
    common.validate_cfg_inputs(common.resolve_path(args.cfg_index), ev)
    config = NDGEncoderConfig(metrics_dim=len(metric_names), ast_dim=args.embedding_dim, cfg_dim=args.embedding_dim,
                              num_edge_types=2*len(ndg_vocab), ndg_structural_dim=len(structural_names), hidden_dim=args.hidden_dim,
                              output_dim=args.embedding_dim, num_layers=args.ndg_layers, heads=args.heads,
                              dropout=args.dropout, attention_dropout=args.attention_dropout)
    common.prepare_output_dir(output)
    eligibility, splits = preflight(projects, selected, ast, cfg, args)
    eligibility.to_csv(output/'project_eligibility.csv', index=False)
    print(eligibility[['project', 'clean_nodes', 'defective_nodes', 'status', 'reason']].to_string(index=False), flush=True)
    paths = [args.ast_index, args.ast_node_vocab, args.cfg_index, args.cfg_node_vocab, args.cfg_stmt_vocab,
             args.cfg_invoke_vocab, args.cfg_edge_vocab, args.cfg_feature_names, args.ndg_index,
             args.ndg_edge_vocab, args.ndg_feature_names]
    if args.ndg_structural_features:
        paths += [args.ndg_structural_index, args.ndg_structural_feature_names]
    print('Recording environment, source and extracted-input fingerprints...', flush=True)
    manifest = dict(protocol=PROTOCOL, started_at=started_at, environment=audit.environment(device),
                    stage_seeds=dict(split=args.seed, ast=args.seed, cfg=args.seed+1000, ndg=args.seed+2000),
                    arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                    source_sha256=audit.fingerprint_source(common.REPO_ROOT),
                    input_sha256=audit.fingerprint_inputs(paths, selected, common.resolve_path))
    (output/'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    skipped = eligibility.loc[eligibility.status.eq('ineligible'), ['project', 'reason']].to_dict('records')
    if args.require_all_projects and skipped:
        raise ValueError('Some projects cannot support this protocol. See project_eligibility.csv; no models were trained.')
    if args.preflight_only:
        print(f'preflight=finished eligible={len(splits)} ineligible={len(skipped)} report={output / "project_eligibility.csv"}', flush=True)
        return
    rows, frames = [], []
    for name in selected:
        if name not in splits:
            print(f'project={name} status=ineligible; see project_eligibility.csv', flush=True)
            continue
        print(f'project={name} protocol={PROTOCOL} status=started', flush=True)
        base = projects[name]
        row, predictions = run_project(base, splits[name], ast, ast_vocab, cfg, (nv,sv,iv,ev), cfg_features,
                                           config, args, device, output/'projects'/name)
        rows.append(row); frames.append(predictions)
        pd.DataFrame(rows).to_csv(output/'project_metrics.csv', index=False)
        print(f'project={name} status=finished f1={row["model_f1"]:.4f} seconds={row["elapsed_seconds"]:.1f}', flush=True)
    summary = dict(protocol=PROTOCOL, requested_projects=selected, completed_projects=len(rows), skipped_projects=skipped,
                    split_fractions=dict(train=1-args.validation_fraction-args.test_fraction, validation=args.validation_fraction, test=args.test_fraction),
                    transductive=True, test_labels_used_for_training=False, refit_after_selection=False,
                    started_at=started_at, finished_at=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=time.perf_counter()-started)
    if frames:
        predictions = pd.concat(frames, ignore_index=True)
        predictions.to_csv(output/'all_test_node_predictions.csv', index=False)
        summary.update(nodes_evaluated=len(predictions), model_macro_project_metrics=common.aggregate_fold_metrics(pd.DataFrame(rows),'model'),
                       model_pooled_metrics=common.safe_pooled_metrics(predictions,'probability','prediction'),
                       baseline_macro_project_metrics=common.aggregate_fold_metrics(pd.DataFrame(rows),'baseline'),
                       baseline_pooled_metrics=common.safe_pooled_metrics(predictions,'baseline_probability','baseline_prediction'))
    (output/'within_project_summary.json').write_text(json.dumps(summary, indent=2))
    print(f'completed={len(rows)} skipped={len(skipped)} summary={output / "within_project_summary.json"}', flush=True)


def common_scenario(args):
    return f'ndg_structural_{"on" if args.ndg_structural_features else "off"}'


if __name__ == '__main__':
    main()
