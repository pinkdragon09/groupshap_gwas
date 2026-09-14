#!/usr/bin/env python3
"""Matched Study 43 pilot: separate RF, joint RF and three single-output KANs.

Run with .venv-kan/bin/python. Historical output directories are never written.
The --folds option stages execution only; the protocol always has five folds.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
import torch
from kan import KAN
from sklearn.ensemble import RandomForestRegressor
from threadpoolctl import threadpool_limits

from fivefold_common import (check_receipt, digest, exact_group_game, fingerprint,
    fit_standardizer, make_splits, regression_metrics, save_receipt, validate_splits,
    write_json, write_npz)

ROOT = Path(__file__).resolve().parents[1]
TRAITS = ['delta 13C', 'stomata density', 'stomatasize']
LABELS = ['delta_13C', 'stomata_density', 'stomatasize']
FAMILIES = ['separate_rf', 'joint_rf', 'kan']
GROUPS = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5']
GENO = ROOT / 'data/1001genomes/genotypes/1001_SNP_MATRIX/imputed_snps_binary.hdf5'
PHENO = ROOT / 'data/arapheno/phenotypes/study_43_values.csv'


def load_cohort(genotype_path, phenotype_path):
    frame = pd.read_csv(phenotype_path, dtype={'accession_id': str})
    frame['accession_id'] = frame.accession_id.str.strip()
    if frame.accession_id.isna().any() or (frame.accession_id == '').any():
        raise ValueError('Missing accession identifiers')
    for col in TRAITS:
        frame[col] = pd.to_numeric(frame[col], errors='raise')
    # Identical duplicate records can be collapsed; conflicting replicates need review.
    frame = frame[['accession_id'] + TRAITS].drop_duplicates()
    if frame.accession_id.duplicated().any():
        raise ValueError('Conflicting phenotype rows for one accession; define aggregation first')
    with h5py.File(genotype_path) as f:
        geno_ids = f['accessions'][:].astype(str)
    if len(set(geno_ids)) != len(geno_ids):
        raise ValueError('Duplicate genotype accessions')
    lookup = {a: i for i, a in enumerate(geno_ids)}
    keep = np.isfinite(frame[TRAITS].to_numpy()).all(axis=1) & frame.accession_id.isin(lookup)
    exclusion = []
    for row in frame.loc[~keep].itertuples(index=False, name=None):
        exclusion.append({'accession': row[0], 'reason': 'missing_or_nonfinite_trait'
                          if not np.isfinite(row[1:]).all() else 'no_genotype'})
    cohort = frame.loc[keep].sort_values('accession_id').reset_index(drop=True)
    ids = cohort.accession_id.to_numpy(dtype=str)
    return ids, np.array([lookup[a] for a in ids]), cohort[TRAITS].to_numpy(dtype=float), exclusion


def stable_top(indices, scores, k):
    order = np.lexsort((indices, -scores))[:k]
    return indices[order], scores[order]


def select_variance_panels(genotype_path, genotype_rows, jobs, per_chr=20,
                           chunk_size=20000, progress=True):
    """Stream the binary HDF5 once for all outer/inner fitting panels.

    jobs maps names to cohort-row indices permitted to affect selection. No
    phenotype is used. Matrix multiplication accumulates exact integer counts
    in float64; p(1-p) is the population variance of each binary marker.
    """
    names = list(jobs)
    membership = np.zeros((len(genotype_rows), len(names)), dtype=np.float64)
    for j, name in enumerate(names):
        rows = np.asarray(jobs[name], dtype=int)
        if len(set(rows)) != len(rows) or len(rows) < 2:
            raise ValueError('Invalid feature-selection fitting rows')
        membership[rows, j] = 1.
    sizes = membership.sum(axis=0)
    selected, meta = {}, {}
    t0 = time.perf_counter()
    with h5py.File(genotype_path) as f:
        attrs = f['positions'].attrs
        labels = ['chr' + (x.decode() if isinstance(x, bytes) else str(x)) for x in attrs['chrs']]
        bounds = np.asarray(attrs['chr_regions'], dtype=int)
        if labels != GROUPS or bounds.shape != (5, 2) or bounds[0, 0] != 0 or bounds[-1, 1] != f['snps'].shape[0]:
            raise ValueError('Unexpected chromosome metadata')
        if not np.array_equal(bounds[1:, 0], bounds[:-1, 1]):
            raise ValueError('Noncontiguous chromosome bounds')
        all_best = {name: [] for name in names}
        for chrom, (lo, hi) in zip(labels, bounds):
            best = {name: (np.empty(0, dtype=np.int64), np.empty(0)) for name in names}
            for start in range(lo, hi, chunk_size):
                raw = f['snps'][start:min(start + chunk_size, hi), :][:, genotype_rows]
                if not np.isin(raw, [0, 1]).all():
                    raise ValueError('Pilot expects complete binary 0/1 genotypes')
                counts = raw.astype(np.float64) @ membership
                p = counts / sizes
                scores = p * (1. - p)
                ix = np.arange(start, start + len(raw))
                for j, name in enumerate(names):
                    ci, cs = stable_top(ix, scores[:, j], per_chr)
                    bi, bs = best[name]
                    best[name] = stable_top(np.r_[bi, ci], np.r_[bs, cs], per_chr)
            for name in names:
                ix, score = best[name]
                if len(ix) != per_chr or np.any(score <= 0):
                    raise ValueError('Too few polymorphic training markers')
                all_best[name].extend(zip(ix.tolist(), score.tolist(), [chrom] * len(ix)))
            if progress:
                print(f'Selection {chrom}: {hi:,} SNPs scanned, {time.perf_counter()-t0:.1f}s', flush=True)
        for name in names:
            rows = sorted(all_best[name])
            ix = np.array([r[0] for r in rows], dtype=np.int64)
            selected[name] = f['snps'][ix, :][:, genotype_rows].T.astype(np.float32)
            selected[name + '_indices'] = ix
            selected[name + '_chromosomes'] = np.array([r[2] for r in rows])
            positions = f['positions'][ix]
            meta[name] = [dict(snp_index=r[0], training_variance=r[1], chromosome=r[2],
                               position=int(pos), feature=f'{r[2]}:{int(pos)}')
                          for r, pos in zip(rows, positions)]
    return selected, meta


def prepare(out, config, signature):
    if check_receipt(out, 'PREPARED.json', signature):
        print('Using checksum-verified training-fold panels', flush=True)
        cohort = json.loads((out / 'cohort.json').read_text())
        splits = json.loads((out / 'splits.json').read_text())
        splits = [{k: np.array(v, dtype=int) if k != 'fold' else v for k, v in s.items()}
                  for s in splits]
        with np.load(out / 'panels.npz', allow_pickle=False) as z:
            panels = dict(z)
        validate_splits(splits, len(cohort['sample_ids']))
        return cohort, splits, panels
    ids, genotype_rows, y, excluded = load_cohort(config['genotype'], config['phenotype'])
    if len(ids) != config['expected_n']:
        raise ValueError(f'Expected {config["expected_n"]} complete cases, found {len(ids)}')
    splits = make_splits(ids, config['split_seed'], config['validation_fraction'])
    jobs = {}
    for s in splits:
        jobs[f'outer{s["fold"]}'] = s['train']
        jobs[f'inner{s["fold"]}'] = s['fit']
    print(f'Cohort: {len(ids)} complete cases, {len(excluded)} excluded phenotype records', flush=True)
    panels, selection = select_variance_panels(config['genotype'], genotype_rows, jobs,
                                              config['per_chr'], config['chunk_size'])
    cohort = dict(sample_ids=ids.tolist(), genotype_rows=genotype_rows.tolist(),
                  traits=TRAITS, labels=LABELS, y=y.tolist(), excluded=excluded,
                  ordering='lexicographic accession_id', genotype_encoding='binary 0/1')
    write_json(out / 'cohort.json', cohort)
    write_json(out / 'splits.json', splits)
    write_json(out / 'selected_features.json', selection)
    write_npz(out / 'panels.npz', **panels)
    save_receipt(out, 'PREPARED.json', signature,
                 ['cohort.json', 'splits.json', 'selected_features.json', 'panels.npz'])
    return cohort, splits, panels


def new_kan(width, seed, config, directory):
    # Numeric branch only, fixed grid: same spline architecture as historic Study 43.
    torch.manual_seed(seed)
    return KAN(width=[width, config['kan_hidden'], 1], grid=config['kan_grid'], k=3,
               seed=seed, auto_save=False, symbolic_enabled=False, save_act=False,
               ckpt_path=str(directory), device='cpu')


def fit_kan(x, y, seed, config, directory, validation=None, epochs=None):
    model = new_kan(x.shape[1], seed, config, directory)
    xt, yt = torch.tensor(x, dtype=torch.float32), torch.tensor(y[:, None], dtype=torch.float32)
    if validation is not None:
        xv = torch.tensor(validation[0], dtype=torch.float32)
        yv = torch.tensor(validation[1][:, None], dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['kan_lr'])
    rng = np.random.default_rng(seed)
    trace, best, best_epoch, stale = [], float('inf'), 1, 0
    for epoch in range(1, (epochs or config['epochs']) + 1):
        model.train()
        order = rng.permutation(len(xt))
        losses = []
        # Include the final partial minibatch (the old Study 43 script omitted it).
        for start in range(0, len(xt), config['kan_batch']):
            ix = order[start:start + config['kan_batch']]
            optimizer.zero_grad()
            loss = ((model(xt[ix]) - yt[ix]) ** 2).mean()
            if not torch.isfinite(loss):
                raise ValueError('Non-finite KAN training loss')
            loss.backward()
            optimizer.step()
            losses.append((len(ix), float(loss.detach())))
        row = dict(epoch=epoch, train_mse=sum(n * loss for n, loss in losses) / len(xt))
        model.eval()
        if validation is not None:
            with torch.no_grad():
                score = float(((model(xv) - yv) ** 2).mean())
            if not np.isfinite(score):
                raise ValueError('Non-finite KAN validation loss')
            row['validation_mse'] = score
            if score < best - 1e-6:
                best, best_epoch, stale = score, epoch, 0
            else:
                stale += 1
        trace.append(row)
        if validation is not None and stale >= config['patience']:
            break
    return model, best_epoch if validation is not None else epochs, trace


def kan_predict(models, x, batch=256):
    cols = []
    with torch.no_grad():
        for model in models:
            pieces = [model(torch.tensor(x[i:i+batch], dtype=torch.float32)).numpy().ravel()
                      for i in range(0, len(x), batch)]
            cols.append(np.concatenate(pieces))
    return np.column_stack(cols)


def fit_family(family, x, inner_x, y, split, config, directory):
    tr, it, va = split['train'], split['fit'], split['validation']
    mu, sd = fit_standardizer(y[tr])
    seed = config['model_seed'] + 100 * split['fold']
    artifacts, notes = [], dict(target_mean=mu, target_sd=sd, seed=seed,
                                input_transform='identity: binary genotypes 0/1')
    if family == 'kan':
        inner_mu, inner_sd = fit_standardizer(y[it])
        models, training = [], {}
        for t, label in enumerate(LABELS):
            _, best_epoch, inner_trace = fit_kan(
                inner_x[it], (y[it, t] - inner_mu[t]) / inner_sd[t], seed+t, config,
                directory, validation=(inner_x[va], (y[va, t] - inner_mu[t]) / inner_sd[t]))
            model, _, outer_trace = fit_kan(x[tr], (y[tr, t] - mu[t]) / sd[t],
                                            seed+t, config, directory, epochs=best_epoch)
            model.eval()
            models.append(model)
            training[label] = dict(chosen_epochs=best_epoch, inner_trace=inner_trace,
                                    refit_trace=outer_trace)
            path = directory / (label + '.pt')
            torch.save(model.state_dict(), path)
            artifacts.append(path.name)
            print(f'  KAN {label}: inner chose {best_epoch} epochs; refit on {len(tr)} accessions', flush=True)
        notes.update(inner_target_mean=inner_mu, inner_target_sd=inner_sd,
                     refit='fresh initialization on all outer-training rows for inner-selected epoch count')
        write_json(directory / 'training.json', training)
        artifacts.append('training.json')
        predict_standardized = lambda a: kan_predict(models, a)
    else:
        args = dict(n_estimators=config['rf_trees'], max_features=0.33, min_samples_leaf=5,
                    n_jobs=config['threads'], random_state=seed)
        target = (y[tr] - mu) / sd
        if family == 'joint_rf':
            model = RandomForestRegressor(**args).fit(x[tr], target)
            model.n_jobs = 1
            models = [model]
            predict_standardized = model.predict
        else:
            models = [RandomForestRegressor(**args).fit(x[tr], target[:, t]) for t in range(3)]
            for model in models:
                model.n_jobs = 1
            predict_standardized = lambda a: np.column_stack([m.predict(a) for m in models])
        joblib.dump(models, directory / 'models.joblib')
        artifacts.append('models.joblib')
    write_json(directory / 'preprocessing.json', notes)
    artifacts.append('preprocessing.json')
    return lambda a: predict_standardized(a) * sd + mu, artifacts, mu, sd


def run_fold(out, cohort, split, panels, config, signature):
    fold = split['fold']
    directory = out / f'fold_{fold}'
    directory.mkdir(exist_ok=True)
    ids, y = np.array(cohort['sample_ids']), np.array(cohort['y'])
    x, inner_x = panels[f'outer{fold}'], panels[f'inner{fold}']
    chroms = panels[f'outer{fold}_chromosomes']
    group_cols = [np.flatnonzero(chroms == name) for name in GROUPS]
    tr, te = split['train'], split['test']
    bg = np.sort(np.random.default_rng(config['split_seed'] + 2000 + fold).choice(
        tr, min(config['background'], len(tr)), replace=False))
    shared = dict(fold=fold, outer_train_ids=ids[tr], test_ids=ids[te],
                  inner_fit_ids=ids[split['fit']], inner_validation_ids=ids[split['validation']],
                  background_ids=ids[bg], evaluation_ids=ids[te], background_rows=bg,
                  outer_feature_indices=panels[f'outer{fold}_indices'],
                  inner_feature_indices=panels[f'inner{fold}_indices'], groups=GROUPS,
                  genotype_input_sha256=fingerprint(x), outer_test_count=len(te))
    write_json(directory / 'shared.json', shared)
    for family in FAMILIES:
        d = directory / family
        d.mkdir(exist_ok=True)
        if check_receipt(d, 'COMPLETE.json', signature):
            print(f'Fold {fold} {family}: verified, skipping completed fit', flush=True)
            continue
        print(f'Fold {fold} {family}: fitting', flush=True)
        t0 = time.perf_counter()
        predict, artifacts, mu, sd = fit_family(family, x, inner_x, y, split, config, d)
        pred = predict(x[te])
        fit_seconds = time.perf_counter() - t0
        metrics = {label: regression_metrics(y[te, t], pred[:, t]) for t, label in enumerate(LABELS)}
        baseline = {label: regression_metrics(y[te, t], np.repeat(mu[t], len(te)))
                    for t, label in enumerate(LABELS)}
        t1 = time.perf_counter()
        game = exact_group_game(predict, x[bg], x[te], group_cols)
        if not np.allclose(game['coalitions'][-1], pred, atol=1e-5, rtol=1e-6):
            raise ValueError('Full-coalition predictions differ from held-out predictions')
        phi_joint = (game['phi'] / sd).mean(axis=2)
        inter_joint = (game['interactions'] / sd).mean(axis=2)
        write_npz(d / 'evaluation.npz', sample_ids=ids[te], observed=y[te], predicted=pred,
                  baseline=np.broadcast_to(mu, pred.shape), target_mean=mu, target_sd=sd,
                  phi=game['phi'], interactions=game['interactions'], pairs=game['pairs'],
                  coalitions=game['coalitions'], joint_phi=phi_joint,
                  joint_interactions=inter_joint, background_ids=ids[bg])
        explanation_seconds = time.perf_counter() - t1
        write_json(d / 'metrics.json', dict(fold=fold, family=family, traits=metrics,
            mean_trait_r2=np.mean([m['r2'] for m in metrics.values()]), baseline=baseline,
            fit_seconds=fit_seconds, explanation_seconds=explanation_seconds,
            efficiency_error=game['efficiency_error'], explanation_count=len(te),
            background_count=len(bg), coalition_count=32,
            attribution_units='native phenotype units; joint game in training-standardized units'))
        artifacts.extend(['evaluation.npz', 'metrics.json'])
        save_receipt(d, 'COMPLETE.json', signature, artifacts)
        print(f'Fold {fold} {family}: R2 {[round(m["r2"],4) for m in metrics.values()]}; '
              f'fit {fit_seconds:.1f}s, explain {explanation_seconds:.1f}s', flush=True)
    save_receipt(directory, 'COMPLETE.json', signature,
                 ['shared.json'] + [f'{family}/COMPLETE.json' for family in FAMILIES])


def summarize(out, cohort, splits, config, signature):
    ids, y = np.array(cohort['sample_ids']), np.array(cohort['y'])
    finished = []
    metrics_rows, group_rows, interaction_rows = [], [], []
    records = {family: [] for family in FAMILIES}
    for split in splits:
        fold = split['fold']
        d = out / f'fold_{fold}'
        if not check_receipt(d, 'COMPLETE.json', signature):
            continue
        finished.append(fold)
        shared = json.loads((d / 'shared.json').read_text())
        if shared['test_ids'] != ids[split['test']].tolist():
            raise ValueError('Saved test identity mismatch')
        if set(shared['background_ids']) - set(shared['outer_train_ids']):
            raise ValueError('Background includes held-out observations')
        for family in FAMILIES:
            check_receipt(d / family, 'COMPLETE.json', signature)
            with np.load(d / family / 'evaluation.npz', allow_pickle=False) as z:
                a = dict(z)
            if not np.array_equal(a['sample_ids'], ids[split['test']]):
                raise ValueError('Prediction rows are misaligned')
            if not np.array_equal(a['observed'], y[split['test']]):
                raise ValueError('Saved phenotype mismatch')
            if not np.array_equal(a['background_ids'], np.array(shared['background_ids'])):
                raise ValueError('Models have different explanation backgrounds')
            mu, sd = fit_standardizer(y[split['train']])
            if not np.allclose(a['target_mean'], mu) or not np.allclose(a['target_sd'], sd):
                raise ValueError('Target scaling was not fitted on the outer training set')
            if not np.allclose(a['phi'].sum(axis=1), a['coalitions'][-1]-a['coalitions'][0], atol=1e-5):
                raise ValueError('Saved explanation efficiency failed')
            if not np.allclose(a['joint_phi'], (a['phi']/sd).mean(axis=2), atol=1e-7):
                raise ValueError('Signed joint Shapley linearity failed')
            if not np.allclose(a['joint_interactions'], (a['interactions']/sd).mean(axis=2), atol=1e-7):
                raise ValueError('Signed joint interaction linearity failed')
            records[family].append((fold, a))
            for t, label in enumerate(LABELS):
                metrics_rows.append(dict(fold=fold, family=family, trait=label,
                    **regression_metrics(a['observed'][:, t], a['predicted'][:, t])))
            for label, phi, inter in [(LABELS[t], a['phi'][:, :, t], a['interactions'][:, :, t])
                                     for t in range(3)] + [('signed_joint', a['joint_phi'], a['joint_interactions'])]:
                for j, group in enumerate(GROUPS):
                    group_rows.append(dict(fold=fold, family=family, trait=label, group=group,
                        mean_abs=float(np.abs(phi[:, j]).mean()), signed_mean=float(phi[:, j].mean())))
                for pair, (j, k) in enumerate(a['pairs']):
                    interaction_rows.append(dict(fold=fold, family=family, trait=label,
                        pair=f'{GROUPS[j]} x {GROUPS[k]}', mean_abs=float(np.abs(inter[:, pair]).mean()),
                        signed_mean=float(inter[:, pair].mean())))
    summary = []
    for family, rec in records.items():
        if not rec:
            continue
        all_ids = np.concatenate([a['sample_ids'] for _, a in rec])
        if len(set(all_ids)) != len(all_ids):
            raise ValueError('Duplicate out-of-fold prediction')
        if len(finished) == 5 and sorted(all_ids.tolist()) != ids.tolist():
            raise ValueError('Incomplete out-of-fold coverage')
        for t, label in enumerate(LABELS):
            fold_scores = [regression_metrics(a['observed'][:, t], a['predicted'][:, t]) for _, a in rec]
            obs = np.concatenate([a['observed'][:, t] for _, a in rec])
            preds = np.concatenate([a['predicted'][:, t] for _, a in rec])
            base = np.concatenate([a['baseline'][:, t] for _, a in rec])
            pooled = regression_metrics(obs, preds)
            summary.append(dict(family=family, trait=label, completed_folds=len(rec),
                mean_fold_r2=np.mean([m['r2'] for m in fold_scores]),
                sd_fold_r2=np.std([m['r2'] for m in fold_scores], ddof=1) if len(rec)>1 else None,
                mean_fold_rmse=np.mean([m['rmse'] for m in fold_scores]),
                pooled_oof_r2=pooled['r2'], pooled_oof_rmse=pooled['rmse'],
                pooled_training_mean_baseline_r2=regression_metrics(obs, base)['r2']))
    def aggregate(rows, key):
        result = []
        frame = pd.DataFrame(rows)
        if frame.empty:
            return result
        for names, sub in frame.groupby(['family', 'trait', key], sort=True):
            result.append(dict(family=names[0], trait=names[1], **{key:names[2]},
                mean_abs=sub.mean_abs.mean(), fold_sd=sub.mean_abs.std(ddof=1) if len(sub)>1 else None,
                completed_folds=len(sub)))
        return result
    paired = []
    for fold in finished:
        for trait in LABELS:
            row = {m['family']:m for m in metrics_rows if m['fold']==fold and m['trait']==trait}
            for a, b in [('joint_rf','separate_rf'),('kan','separate_rf'),('kan','joint_rf')]:
                paired.append(dict(fold=fold, trait=trait, comparison=f'{a} minus {b}',
                                   r2_difference=row[a]['r2']-row[b]['r2'],
                                   rmse_difference=row[a]['rmse']-row[b]['rmse']))
    selection = json.loads((out / 'selected_features.json').read_text())
    selected_ids = sorted(set(r['snp_index'] for s in splits for r in selection[f'outer{s["fold"]}']))
    frequency = [dict(snp_index=i, folds_selected=sum(any(r['snp_index']==i for r in selection[f'outer{s["fold"]}']) for s in splits)) for i in selected_ids]
    write_json(out / 'metrics_by_fold.json', metrics_rows)
    write_json(out / 'model_summary.json', summary)
    write_json(out / 'attribution_by_fold.json', group_rows)
    write_json(out / 'attribution_summary.json', aggregate(group_rows, 'group'))
    write_json(out / 'interaction_by_fold.json', interaction_rows)
    write_json(out / 'interaction_summary.json', aggregate(interaction_rows, 'pair'))
    write_json(out / 'paired_differences.json', paired)
    write_json(out / 'selection_frequency.json', frequency)
    lines = ['# Study 43 matched five-fold pilot', '',
        f'Completed folds: {finished}. Primary cohort: {len(ids)} complete cases.', '',
        'Separate RF, joint RF and three single-output KANs share accession splits, training-selected SNP panels and explanation backgrounds.',
        'KAN joint attribution is a signed average of three standardized-output games, not a jointly trained multi-task model.', '',
        '| Model | Trait | Mean fold R2 | Fold SD | Mean RMSE | Pooled OOF R2 |',
        '|---|---|---:|---:|---:|---:|']
    for row in summary:
        sd = 'NA' if row['sd_fold_r2'] is None else f'{row["sd_fold_r2"]:.4f}'
        lines.append(f'| {row["family"]} | {row["trait"]} | {row["mean_fold_r2"]:.4f} | {sd} | {row["mean_fold_rmse"]:.4f} | {row["pooled_oof_r2"]:.4f} |')
    lines += ['', '## Interpretation and limits', '',
        'All attributions and interactions use held-out accessions and 32 exact coalitions over a finite training background. Trait-specific values are in native phenotype units; signed-joint values are in training-standardized units.',
        'Fold SD is descriptive, not a confidence interval. This pilot uses one training seed per fold; it does not measure across-seed uncertainty or establish causal effects.',
        'Five shuffled folds estimate within-panel prediction. They do not establish transfer to new population structure. No population PCs were added to the historical Study 43 input design.',
        'Near-zero or negative predictive R2 limits interpretation of chromosome rankings. A variance filter supplies common polymorphic markers, not an independently validated QTL panel.',
        'The historical 330-accession stomatal analyses are not the matched comparison cohort and remain separate supplementary work.', '',
        '## Files', '',
        '`manifest.json`, `cohort.json`, `splits.json`, `selected_features.json`, and `panels.npz` freeze the run inputs.',
        'Each fold has shared sample/background IDs and each model family has checkpoints, preprocessing, held-out prediction/explanation arrays, timing, and SHA256 completion receipts.',
        '`model_summary.json`, `paired_differences.json`, `attribution_summary.json`, `interaction_summary.json`, and `selection_frequency.json` are the numerical review artifacts.']
    (out / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    write_json(out / 'STATUS.json', dict(completed_folds=finished, required_folds=5,
                                        complete=len(finished)==5, checks_passed=True))
    if len(finished) == 5:
        save_receipt(out, 'COMPLETE.json', signature,
            ['PREPARED.json']+[f'fold_{i}/COMPLETE.json' for i in finished]+
            ['model_summary.json','metrics_by_fold.json','paired_differences.json',
             'attribution_summary.json','interaction_summary.json','selection_frequency.json','REPORT.md'])
    print(f'Validated {len(finished)}/5 complete folds; report: {out / "REPORT.md"}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--folds', type=int, nargs='+', default=[1,2,3,4,5])
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--patience', type=int, default=30)
    p.add_argument('--rf-trees', type=int, default=300)
    p.add_argument('--background', type=int, default=20)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if any(v<1 for v in [args.epochs,args.patience,args.rf_trees,args.background,args.threads]) or not set(args.folds)<=set(range(1,6)):
        p.error('Invalid positive budget or fold index')
    out = args.outdir.resolve()
    if out.exists() and any(out.iterdir()) and not (args.resume or args.verify_only):
        p.error('Output directory is nonempty; use --resume or choose a new directory')
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(args.threads)
        torch.use_deterministic_algorithms(True)
        with threadpool_limits(limits=args.threads):
            if args.verify_only:
                manifest = json.loads((out / 'manifest.json').read_text())
                config, signature = manifest['config'], manifest['fingerprint']
                if not check_receipt(out, 'PREPARED.json', signature):
                    raise ValueError('Run has no completed preparation')
                cohort, splits, _ = prepare(out, config, signature)
                if (out / 'COMPLETE.json').exists():
                    check_receipt(out, 'COMPLETE.json', signature)
                summarize(out, cohort, splits, config, signature)
                return
            config = dict(schema_version=1, study='Study43_complete_cases', expected_n=310,
                genotype=str(GENO), phenotype=str(PHENO), split_seed=42, model_seed=42,
                validation_fraction=0.2, per_chr=20, chunk_size=20000,
                epochs=args.epochs, patience=args.patience, rf_trees=args.rf_trees,
                background=args.background, threads=args.threads,
                kan_hidden=8, kan_grid=5, kan_lr=0.001, kan_batch=64,
                evaluation='all outer-test accessions', grouping=GROUPS,
                feature_ties='descending training variance, ascending global SNP index',
                refit='inner-selected epochs; fresh KAN fit on all outer-training rows',
                rf_max_features=0.33, rf_min_samples_leaf=5, kan_input_transform='identity 0/1')
            print('Hashing inputs and executable source for run identity', flush=True)
            identity = dict(config=config, inputs={str(f):digest(f) for f in [GENO, PHENO]},
                source={str(f.relative_to(ROOT)):digest(f) for f in [Path(__file__).resolve(), Path(__file__).with_name('fivefold_common.py')]},
                versions={package:importlib.metadata.version(package) for package in ['numpy','pandas','h5py','torch','scikit-learn','pykan']},
                python=platform.python_version())
            signature = fingerprint(identity)
            manifest_path = out / 'manifest.json'
            if manifest_path.exists():
                if json.loads(manifest_path.read_text())['fingerprint'] != signature:
                    raise ValueError('Inputs, code, environment or configuration changed; choose a fresh run directory')
            else:
                write_json(manifest_path, dict(**identity, fingerprint=signature,
                    repository_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
            cohort, splits, panels = prepare(out, config, signature)
            if args.prepare_only:
                return
            for split in splits:
                if split['fold'] in args.folds:
                    run_fold(out, cohort, split, panels, config, signature)
            summarize(out, cohort, splits, config, signature)


if __name__ == '__main__':
    main()
