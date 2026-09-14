"""Shared, auditable outer-CV splits, artifact receipts and grouped explanations.

No study-specific loaders or model hyperparameters belong in this module.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def write_npz(path: Path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(tmp, path)


def make_splits(sample_ids, seed=42, validation_fraction=0.2):
    """Stable ID order, outer folds 1..5, sorted indices for every role."""
    ids = np.asarray(sample_ids, dtype=str)
    if len(ids) < 10 or len(set(ids)) != len(ids):
        raise ValueError('Need at least ten unique sample IDs')
    if ids.tolist() != sorted(ids.tolist()):
        raise ValueError('Caller must align samples in sorted ID order')
    result = []
    for fold, (train, test) in enumerate(KFold(5, shuffle=True, random_state=seed).split(ids), 1):
        fit, val = train_test_split(train, test_size=validation_fraction,
                                    random_state=seed + 1000 + fold)
        result.append(dict(fold=fold, train=train, test=test,
                           fit=np.sort(fit), validation=np.sort(val)))
    validate_splits(result, len(ids))
    return result


def validate_splits(splits, n):
    if len(splits) != 5 or [s['fold'] for s in splits] != list(range(1, 6)):
        raise ValueError('Exactly five numbered outer folds are required')
    tests = []
    for s in splits:
        roles = {r: set(np.asarray(s[r], dtype=int).tolist())
                 for r in ('train', 'test', 'fit', 'validation')}
        if any(len(roles[r]) != len(s[r]) or not roles[r] for r in roles):
            raise ValueError('Empty or duplicate split indices')
        if roles['train'] & roles['test'] or roles['train'] | roles['test'] != set(range(n)):
            raise ValueError('Outer train/test partition is invalid')
        if roles['fit'] & roles['validation'] or roles['fit'] | roles['validation'] != roles['train']:
            raise ValueError('Inner fit/validation partition is invalid')
        tests.extend(s['test'])
    if sorted(tests) != list(range(n)):
        raise ValueError('Each sample must be tested exactly once')


def fit_standardizer(y):
    y = np.asarray(y, dtype=float)
    if not np.isfinite(y).all() or len(y) < 2:
        raise ValueError('Invalid normalization fitting data')
    mu, sd = y.mean(axis=0), y.std(axis=0, ddof=0)
    return mu, np.where(sd > 1e-12, sd, 1.0)


def regression_metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    if y.shape != pred.shape or not np.isfinite(pred).all():
        raise ValueError('Invalid predictions')
    # force_finite=False avoids reporting artificial perfect R2 on constant outcomes.
    return {'r2': float(r2_score(y, pred, force_finite=False)),
            'rmse': float(np.sqrt(mean_squared_error(y, pred))),
            'n': len(y)}


def exact_group_game(predict: Callable, background, evaluation, groups):
    """Multi-output marginal game; preserve missing groups jointly in each row.

    Returns signed local Shapley and Shapley Interaction Index values. The pair
    normalization is |S|!(m-|S|-2)!/(m-1)!, not a half-sized matrix entry.
    Exactness refers to coalition enumeration over this finite background.
    """
    background, evaluation = np.asarray(background), np.asarray(evaluation)
    if background.ndim != 2 or evaluation.ndim != 2 or background.shape[1] != evaluation.shape[1]:
        raise ValueError('Incompatible explanation inputs')
    flat = sorted(int(c) for g in groups for c in g)
    if flat != list(range(evaluation.shape[1])) or any(len(g) == 0 for g in groups):
        raise ValueError('Groups must be a disjoint exhaustive feature partition')
    m, nb, ne = len(groups), len(background), len(evaluation)
    if not nb or not ne or m > 12:
        raise ValueError('Invalid exact-game budget')
    values = []
    for mask in range(1 << m):
        hybrid = np.tile(background, (ne, 1))
        for j, cols in enumerate(groups):
            if mask & (1 << j):
                hybrid[:, cols] = np.repeat(evaluation[:, cols], nb, axis=0)
        pred = np.asarray(predict(hybrid))
        if pred.ndim == 1:
            pred = pred[:, None]
        if pred.ndim != 2 or len(pred) != ne * nb or not np.isfinite(pred).all():
            raise ValueError('Non-finite or malformed coalition predictions')
        values.append(pred.reshape(ne, nb, -1).mean(axis=1))
    v = np.asarray(values)
    phi = np.zeros((ne, m, v.shape[-1]))
    pair_ids, interactions = [], []
    for j in range(m):
        for mask in range(1 << m):
            if mask & (1 << j):
                continue
            k = mask.bit_count()
            w = math.factorial(k) * math.factorial(m-k-1) / math.factorial(m)
            phi[:, j] += w * (v[mask | (1 << j)] - v[mask])
        for l in range(j+1, m):
            inter = np.zeros((ne, v.shape[-1]))
            for mask in range(1 << m):
                if mask & ((1 << j) | (1 << l)):
                    continue
                k = mask.bit_count()
                w = math.factorial(k) * math.factorial(m-k-2) / math.factorial(m-1)
                inter += w * (v[mask | (1 << j) | (1 << l)] - v[mask | (1 << j)]
                              - v[mask | (1 << l)] + v[mask])
            pair_ids.append((j, l))
            interactions.append(inter)
    err = float(np.max(np.abs(phi.sum(axis=1) - (v[-1] - v[0]))))
    if err > 1e-7 * max(1., float(np.max(np.abs(v)))):
        raise ValueError(f'Shapley efficiency failed: {err}')
    return dict(coalitions=v, phi=phi,
                interactions=np.stack(interactions, axis=1) if interactions else np.empty((ne, 0, v.shape[-1])),
                pairs=np.asarray(pair_ids, dtype=int).reshape(-1, 2), efficiency_error=err)


def save_receipt(directory, name, run_fingerprint, files):
    directory = Path(directory)
    artifacts = {f: digest(directory / f) for f in files}
    write_json(directory / name, {'run_fingerprint': run_fingerprint, 'artifacts': artifacts})


def check_receipt(directory, name, run_fingerprint):
    path = Path(directory) / name
    if not path.exists():
        return False
    receipt = json.loads(path.read_text())
    if receipt['run_fingerprint'] != run_fingerprint:
        raise ValueError(f'Configuration mismatch in {path}')
    for relative, expected in receipt['artifacts'].items():
        file = Path(directory) / relative
        if not file.is_file() or digest(file) != expected:
            raise ValueError(f'Artifact integrity failure: {file}')
    return True
