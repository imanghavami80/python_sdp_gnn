"""Grouped within-project splits and transductive, label-masked GNN training."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .ndg import model_inputs, select_f1_threshold


class InsufficientClassSupport(ValueError):
    """A project cannot support three disjoint binary evaluation partitions."""


def split_files(names, source_paths, labels, seed, validation_fraction=.2, test_fraction=.2):
    """Stratify source-file groups, allocating at least one per class per split.

    Mixed-label files are stratified by whether any of their rows are defective.
    Realized node fractions can differ from requested group fractions.
    """
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1 or validation_fraction + test_fraction >= 1:
        raise ValueError('Validation/test fractions must be positive and sum to less than one')
    y = np.asarray(labels)
    if len(names) != len(y) or len(source_paths) != len(y) or len(set(names)) != len(names):
        raise ValueError('Node identifiers must be unique and arrays must align')
    if not np.isin(y, [0, 1]).all():
        raise ValueError('Binary labels required')
    groups = {}
    for i, path in enumerate(source_paths):
        if not str(path).strip():
            raise ValueError('Missing source file identity')
        groups.setdefault(str(Path(path).resolve()), []).append(i)
    strata = [[g for g in sorted(groups) if int(y[groups[g]].max()) == c] for c in [0, 1]]
    if min(map(len, strata)) < 3:
        raise InsufficientClassSupport(f'Need >=3 source-file groups per class; found {[len(s) for s in strata]}')
    result = {k: [] for k in ['train', 'validation', 'test']}
    rng = np.random.default_rng(seed)
    for stratum in strata:
        order = rng.permutation(len(stratum))
        n_test = max(1, int(round(len(stratum)*test_fraction)))
        n_val = max(1, int(round(len(stratum)*validation_fraction)))
        while n_test+n_val >= len(stratum):
            if n_test >= n_val and n_test > 1:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                raise InsufficientClassSupport('Insufficient training groups')
        parts = {'test': order[:n_test], 'validation': order[n_test:n_test+n_val], 'train': order[n_test+n_val:]}
        for split, positions in parts.items():
            for i in positions:
                result[split].extend(groups[stratum[i]])
    for split, rows in result.items():
        result[split] = np.array(sorted(rows), dtype=np.int64)
        if len(np.unique(y[rows])) != 2:
            raise InsufficientClassSupport(f'{split} does not contain both classes')
    return result


def scale_from_training(graph, indices):
    """Persist source-only preprocessing parameters for exact checkpoint reuse."""
    x = graph.metrics_x
    train = x[indices]
    medians = torch.nanmedian(train, dim=0).values
    medians = torch.nan_to_num(medians, nan=0.)
    missing = torch.isnan(train).all(dim=0)
    imputed = torch.where(torch.isnan(x), medians, x)
    mean = imputed[indices].mean(dim=0)
    std = imputed[indices].std(dim=0, unbiased=False)
    std = torch.where(std > 1e-6, std, torch.ones_like(std))
    metrics = (imputed-mean)/std
    metrics[:, missing] = 0  # No training evidence for this coordinate.
    structural = graph.ndg_structural_x
    params = dict(metric_medians=medians.tolist(), metric_mean=mean.tolist(), metric_std=std.tolist(),
                  all_missing_metric_columns=torch.where(missing)[0].tolist())
    if structural is not None and structural.shape[1]:
        sm = structural[indices].mean(dim=0)
        ss = structural[indices].std(dim=0, unbiased=False)
        ss = torch.where(ss > 1e-6, ss, torch.ones_like(ss))
        structural = (structural-sm)/ss
        params.update(structural_mean=sm.tolist(), structural_std=ss.tolist())
    if not torch.isfinite(metrics).all() or (structural is not None and not torch.isfinite(structural).all()):
        raise ValueError('Non-finite preprocessed features')
    return replace(graph, metrics_x=metrics, ndg_structural_x=structural), params


def train_masked(model, graph, train_indices, validation_indices, args):
    """Full graph forward passes; training and validation labels selected explicitly."""
    train = torch.as_tensor(train_indices, device=graph.y.device, dtype=torch.long)
    val = torch.as_tensor(validation_indices, device=graph.y.device, dtype=torch.long)
    if not len(train) or not len(val) or set(train.tolist()) & set(val.tolist()):
        raise ValueError('Nonempty disjoint train/validation indices required')
    y = graph.y[train]
    if not torch.isfinite(y).all() or not torch.isfinite(graph.y[val]).all():
        raise ValueError('Training and validation labels must be finite')
    pos_weight = (1-y).sum()/y.sum().clamp_min(1)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state, best_loss, best_epoch, stale = None, float('inf'), 0, 0
    history = []
    for epoch in range(1, args.epochs+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(**model_inputs(graph))
        loss = criterion(logits[train], graph.y[train])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(**model_inputs(graph))[val], graph.y[val]))
        if not np.isfinite(val_loss):
            raise ValueError('Non-finite validation loss')
        history.append(dict(epoch=epoch, train_loss=float(loss.detach()), validation_loss=val_loss))
        print(f'ndg_epoch={epoch:03d} train_loss={float(loss.detach()):.4f} val_loss={val_loss:.4f}', flush=True)
        if val_loss < best_loss-args.min_delta:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_loss, best_epoch, stale = val_loss, epoch, 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise ValueError('No model selected')
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(**model_inputs(graph))[val]).cpu().numpy()
    threshold = select_f1_threshold(graph.y[val].cpu().numpy(), probabilities)
    return best_epoch, threshold, history
