"""Data, panel-selection and model helpers for the Bloom yeast rerun."""
from __future__ import annotations

import csv
import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from kan import KAN
from scipy.stats import t as student_t
from sklearn.ensemble import RandomForestRegressor

from fivefold_common import fit_standardizer
from load_yeast_bloom import (GENO_2013, PHENO_2013, INTERACTIONS_2015_CSV,
                              load_geno_2013, load_pheno_2013, load_interactions_2015)

ROOT = Path(__file__).resolve().parents[1]
TRAITS = ['Neomycin', 'Lactate']


def load_trait(trait):
    geno, _, meta = load_geno_2013('01')
    pheno = load_pheno_2013()
    if not geno.index.is_unique or not geno.columns.is_unique:
        raise ValueError('Duplicate segregant or marker identifiers')
    ycol = pheno[trait]
    ids = sorted(set(geno.index) & set(ycol[ycol.notna()].index))
    y = ycol.loc[ids].to_numpy(float)
    x = geno.loc[ids].to_numpy(np.float32)
    if not np.isfinite(y).all() or not np.isin(x, [0, 1]).all():
        raise ValueError('Invalid aligned yeast data')
    records = []
    for index, row in meta.reset_index(drop=True).iterrows():
        if pd.isna(row['chr']) or pd.isna(row['pos']):
            records.append(dict(marker_index=index, marker_id=row['marker_id'],
                                chromosome=None, position=None))
        else:
            records.append(dict(marker_index=index, marker_id=row['marker_id'],
                                chromosome=normalize_chrom(row['chr']), position=int(row['pos'])))
    return ids, x, y, records


def normalize_chrom(value):
    value = str(value).strip()
    if not value.startswith('chr'):
        value = 'chr' + value
    return value


def marginal_statistics(x, y, min_maf=.05):
    n = len(y)
    xc = x.astype(np.float64) - x.mean(0, dtype=np.float64)
    yc = y.astype(np.float64) - y.mean()
    ssx = np.einsum('ij,ij->j', xc, xc)
    ssy = float(yc @ yc)
    maf = np.minimum(x.mean(0), 1-x.mean(0))
    score = np.full(x.shape[1], -np.inf)
    pvalue = np.ones(x.shape[1])
    ok = (maf >= min_maf) & (ssx > 1e-12) & (ssy > 1e-12)
    r = np.zeros(x.shape[1])
    r[ok] = (xc[:, ok].T @ yc) / np.sqrt(ssx[ok]*ssy)
    r = np.clip(r, -0.9999999, 0.9999999)
    score[ok] = r[ok]**2
    tstat = np.abs(r[ok]) * np.sqrt(n-2) / np.sqrt(1-r[ok]**2)
    pvalue[ok] = 2*student_t.sf(tstat, n-2)
    return score, pvalue, maf


def select_panel(x, y, marker_meta, max_markers=100, min_maf=.05, prune_bp=100000):
    score, pvalue, maf = marginal_statistics(x, y, min_maf)
    index = np.arange(x.shape[1])
    order = np.lexsort((index, -score))
    chosen, positions = [], {}
    for j in order:
        row = marker_meta[int(j)]
        if not np.isfinite(score[j]) or row['chromosome'] is None:
            continue
        prior = positions.setdefault(row['chromosome'], [])
        if any(abs(row['position']-p) < prune_bp for p in prior):
            continue
        chosen.append(int(j))
        prior.append(row['position'])
        if len(chosen) == max_markers:
            break
    if len(chosen) < min(50, max_markers):
        raise ValueError('Unexpectedly small pruned marker panel')
    selection = [dict(**marker_meta[j], training_maf=float(maf[j]),
                      training_r2=float(score[j]), training_pvalue=float(pvalue[j]),
                      neg_log10_p=float(-np.log10(max(pvalue[j], 1e-300))),
                      selection_rank=rank+1)
                 for rank, j in enumerate(chosen)]
    return np.array(chosen, dtype=int), selection


def new_kan(width, seed, hidden=8, grid=5):
    torch.manual_seed(seed)
    return KAN(width=[width, hidden, 1], grid=grid, k=3, seed=seed,
               auto_save=False, symbolic_enabled=False, save_act=False, device='cpu')


def fit_kan(x, y, seed, epochs=200, patience=25, validation=None):
    model = new_kan(x.shape[1], seed)
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor(y[:, None], dtype=torch.float32)
    if validation is not None:
        xv = torch.tensor(validation[0], dtype=torch.float32)
        yv = torch.tensor(validation[1][:, None], dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    rng = np.random.default_rng(seed)
    trace, best, best_epoch, stale = [], float('inf'), 1, 0
    for epoch in range(1, epochs+1):
        model.train()
        order = rng.permutation(len(x))
        total = 0.
        for start in range(0, len(x), 64):
            ix = order[start:start+64]
            optimizer.zero_grad()
            loss = ((model(xt[ix])-yt[ix])**2).mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite KAN loss')
            loss.backward()
            optimizer.step()
            total += len(ix)*float(loss.detach())
        row = dict(epoch=epoch, training_mse=total/len(x))
        model.eval()
        if validation is not None:
            with torch.no_grad():
                val = float(((model(xv)-yv)**2).mean())
            row['validation_mse'] = val
            if val < best-1e-6:
                best, best_epoch, stale = val, epoch, 0
            else:
                stale += 1
        trace.append(row)
        if validation is not None and stale >= patience:
            break
    return model, best_epoch if validation is not None else epochs, trace


def predict_kan(model, x, mu, sd, batch=512):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.tensor(x[i:i+batch], dtype=torch.float32)).numpy().ravel()
                               for i in range(0, len(x), batch)])*sd+mu


def fit_rf(x, y, seed, trees=200):
    model = RandomForestRegressor(n_estimators=trees, max_features='sqrt',
        min_samples_leaf=2, random_state=seed, n_jobs=4)
    model.fit(x, y)
    return model


def map_references(trait, selection, distance_bp=10000):
    refs = load_interactions_2015()
    refs = refs[refs.trait == trait].reset_index(drop=True)
    result = []
    for ref_index, row in refs.iterrows():
        mapped = []
        for which in [1, 2]:
            chrom, pos = normalize_chrom(row[f'Q{which}_chr']), int(row[f'Q{which}_pos'])
            candidates = [(i, abs(feature['position']-pos)) for i, feature in enumerate(selection)
                          if feature['chromosome'] == chrom]
            mapped.append(min(candidates, key=lambda value:value[1]) if candidates else (None, None))
        eligible = all(j is not None and d <= distance_bp for j, d in mapped) and mapped[0][0] != mapped[1][0]
        result.append(dict(reference_index=ref_index, trait=trait,
            q1_chromosome=normalize_chrom(row.Q1_chr), q1_position=int(row.Q1_pos),
            q2_chromosome=normalize_chrom(row.Q2_chr), q2_position=int(row.Q2_pos),
            q1_panel_index=mapped[0][0], q1_distance_bp=mapped[0][1],
            q2_panel_index=mapped[1][0], q2_distance_bp=mapped[1][1],
            eligible=eligible, lod=float(row.LOD), lm_coefficient=float(row.lm_coefficient)))
    return result
