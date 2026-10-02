"""Within-project split, leakage boundary, and saved-model integration checks."""
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import evaluate_ndg_within_project as runner
from thesis_project.training.within_project import split_files, scale_from_training, InsufficientClassSupport
from thesis_project.training.ndg import ProjectGraph, make_model
from thesis_project.models import NDGEncoderConfig


def test_split_grouping_reproducibility_and_rare_classes():
    y = np.array([0]*20+[1]*20)
    paths = [f'/tmp/file_{i//2}' for i in range(40)]
    names = [str(i) for i in range(40)]
    a = split_files(names, paths, y, 42)
    b = split_files(names, paths, y, 42)
    assert set(np.concatenate(list(a.values()))) == set(range(40))
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
        assert len(np.unique(y[a[k]])) == 2
        for other in a:
            if k != other:
                assert not set(np.array(paths)[a[k]]) & set(np.array(paths)[a[other]])
    with pytest.raises(InsufficientClassSupport):
        split_files(names, names, np.array([1]*38+[0]*2), 42)


def test_scaling_ignores_heldout_metrics_and_labels():
    x = torch.tensor([[1., float('nan')], [3., float('nan')], [100., 8.]])
    g = ProjectGraph('p', ['a','b','c'], ['a','b','c'], x, torch.zeros(3,2), torch.zeros(3,2),
                     torch.ones(3,3,dtype=torch.bool), torch.tensor([0.,1.,float('nan')]),
                     torch.empty(2,0,dtype=torch.long), torch.empty(0,dtype=torch.long))
    normalized, params = scale_from_training(g, np.array([0,1]))
    assert params['metric_mean'] == [2.,0.]
    assert normalized.metrics_x[:,1].eq(0).all()
    other = replace(g, metrics_x=torch.tensor([[1.,float('nan')],[3.,float('nan')],[-999.,100.]]))
    _, other_params = scale_from_training(other, np.array([0,1]))
    assert params == other_params


@pytest.mark.parametrize('structural', [False, True])
def test_end_to_end_masking_and_checkpoint(tmp_path, monkeypatch, structural):
    rng = np.random.default_rng(42)
    n = 40
    names = [f'file_{i}' for i in range(n)]
    base = runner.common.BaseNDGProject('p', names, names, rng.normal(size=(n,4)).astype('float32'),
                                       np.array([0,1]*20,dtype='float32'),
                                       np.array([np.arange(n-1),np.arange(1,n)]), np.zeros(n-1,dtype='int64'),
                                       rng.normal(size=(n, 6 if structural else 0)).astype('float32'))
    split = split_files(names,names,base.y,42)
    index = pd.DataFrame(dict(dataset_name='p',name=names,label=base.y))
    lookup = {('p',name):rng.normal(size=8).astype('float32') for name in names}
    config = NDGEncoderConfig(metrics_dim=4,ast_dim=8,cfg_dim=8,num_edge_types=1,hidden_dim=8,output_dim=8,heads=2,dropout=0.,ndg_structural_dim=6 if structural else 0)
    args = runner.parse_args(['--ndg-epochs','3','--patience','2','--embedding-dim','8','--hidden-dim','8'])
    calls=[]

    def upstream(view, data, parts, *unused):
        # Loader inputs contain no held-out test labels. Only train/val define supervised fitting.
        assert data.iloc[parts['test']].label.eq(0).all()
        assert not set(parts['train']) & set(parts['test'])
        calls.append(view)
        return lookup, {'selected_epochs':1}, []
    monkeypatch.setattr(runner,'train_upstream',upstream)
    row,pred = runner.run_project(base,split,index,{},index,({},)*4,[],config,args,torch.device('cpu'),tmp_path/'a')
    altered = base.y.copy();altered[split['test']] = 1-altered[split['test']]
    changed_index=index.copy();changed_index.label=altered
    _,changed=runner.run_project(replace(base,y=altered),split,changed_index,{},changed_index,({},)*4,[],config,args,torch.device('cpu'),tmp_path/'b')
    np.testing.assert_array_equal(pred.probability,changed.probability)
    assert calls == ['ast','cfg','ast','cfg']  # no final retraining
    saved=torch.load(tmp_path/'a/ndg_encoder.pt',weights_only=False)
    model=make_model(NDGEncoderConfig(**saved['encoder_config']),torch.device('cpu'))
    model.load_state_dict(saved['model_state_dict']);model.eval()
    with torch.no_grad():
        p=torch.sigmoid(model(**saved['model_inputs'])).numpy()[split['test']]
    np.testing.assert_allclose(p,pred.probability,atol=1e-7)
    assert saved['refit_after_selection'] is False
    assert set(pred.name) == set(np.array(names)[split['test']])
    assert row['ndg_epochs_run'] == len(pd.read_csv(tmp_path/'a/ndg_selection_history.csv'))
    assert row['elapsed_seconds'] >= row['ndg_seconds'] > 0


def test_preflight_reports_rare_projects_without_training():
    projects, indices = {}, []
    for name, labels in [('normal', [0, 1]*10), ('forrest', [0]*30+[1]*2), ('xalan', [0]+[1]*898)]:
        names = [f'{name}_{i}' for i in range(len(labels))]
        projects[name] = SimpleNamespace(names=names, source_paths=names, y=np.array(labels), dataset_name=name)
        indices.append(pd.DataFrame(dict(dataset_name=name, name=names, label=labels)))
    index = pd.concat(indices, ignore_index=True)
    report, splits = runner.preflight(projects, list(projects), index, index, runner.parse_args([]))
    assert set(splits) == {'normal'}
    assert report.status.tolist() == ['eligible', 'ineligible', 'ineligible']
    assert report.clean_nodes.tolist() == [10, 30, 1]
    assert report.defective_nodes.tolist() == [10, 2, 898]
    assert report.iloc[1].reason and report.iloc[2].reason


def test_input_fingerprints_detect_tensor_changes(tmp_path):
    tensor = tmp_path/'x.npy'
    np.save(tensor, np.array([1.]))
    index = tmp_path/'graph_index.csv'
    pd.DataFrame(dict(dataset_name=['p'], x_npy=[str(tensor)])).to_csv(index, index=False)
    before = runner.audit.fingerprint_inputs([index], ['p'], lambda p: p.resolve())
    np.save(tensor, np.array([2.]))
    after = runner.audit.fingerprint_inputs([index], ['p'], lambda p: p.resolve())
    assert before[str(index)] == after[str(index)]
    assert before[str(tensor)] != after[str(tensor)]
