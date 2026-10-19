#!/usr/bin/env python3
"""Study 43: phenotype-blind 10-kb panels, RF/additive ridge, GWAS, block SHAP.

Distance windows are not empirically inferred LD blocks. No population covariates.
New output directory required; completed folds are resumable with identical inputs.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import time
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
from scipy.stats import t as student_t, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from fivefold_common import (digest, fingerprint, make_splits, regression_metrics,
                             write_json, write_npz)

ROOT = Path(__file__).resolve().parents[1]
GENO = ROOT / 'data/1001genomes/genotypes/1001_SNP_MATRIX/imputed_snps_binary.hdf5'
PHENO = ROOT / 'data/arapheno/phenotypes/study_43_values.csv'
TRAITS = ['delta 13C', 'stomata density', 'stomatasize']
LABELS = ['delta_13C', 'stomata_density', 'stomatasize']
SIZES = [1000, 5000, 10000]


def load_cohort():
    frame = pd.read_csv(PHENO, dtype={'accession_id': str})
    frame['accession_id'] = frame.accession_id.str.strip()
    frame = frame[['accession_id'] + TRAITS].drop_duplicates()
    if frame.accession_id.isna().any() or frame.accession_id.duplicated().any():
        raise ValueError('Missing or conflicting accession records')
    y = frame[TRAITS].apply(pd.to_numeric, errors='raise')
    with h5py.File(GENO) as f:
        accessions = f['accessions'][:].astype(str)
    lookup = {a: i for i, a in enumerate(accessions)}
    if len(lookup) != len(accessions):
        raise ValueError('Duplicate genotype identifiers')
    keep = np.isfinite(y.to_numpy()).all(axis=1) & frame.accession_id.isin(lookup)
    excluded = frame.loc[~keep, 'accession_id'].tolist()
    frame = frame.loc[keep].sort_values('accession_id')
    ids = frame.accession_id.to_numpy(dtype=str)
    if len(ids) != 310:
        raise ValueError(f'Expected 310 complete cases; found {len(ids)}')
    return ids, np.array([lookup[a] for a in ids]), frame[TRAITS].to_numpy(float), excluded


def eligible_indices(raw, train, priority, mac=5):
    """Selection depends only on fitting-row genotypes; priority is outcome blind."""
    counts = raw[:, train].sum(axis=1)
    eligible = np.minimum(counts, len(train) - counts) >= mac
    return priority[eligible[priority]][:100]


def prepare_panels(rows, jobs, seed):
    """Twenty random eligible windows/chromosome, 100 priority SNPs/window/job.

    Positions determine random window order before any genotype inspection.
    Jobs independently skip windows with <100 training-MAC-qualified SNPs.
    """
    values = {name: [] for name in jobs}
    indices = {name: [] for name in jobs}
    blocks = {name: [] for name in jobs}
    coords = {name: [] for name in jobs}
    with h5py.File(GENO) as f:
        bounds = f['positions'].attrs['chr_regions']
        for chrom, (lo, hi) in enumerate(bounds, 1):
            pos = f['positions'][lo:hi]
            window = (pos - 1) // 10000
            starts = np.r_[0, np.flatnonzero(np.diff(window)) + 1]
            stops = np.r_[starts[1:], len(pos)]
            # Requiring >=100 raw SNPs is phenotype independent and saves I/O.
            candidates = np.flatnonzero(stops - starts >= 100)
            order = np.random.default_rng(seed + chrom).permutation(candidates)
            counts = {name: 0 for name in jobs}
            scanned = 0
            for b in order:
                start, stop = int(starts[b]), int(stops[b])
                raw = f['snps'][int(lo)+start:int(lo)+stop, :][:, rows]
                if not np.isin(raw, [0, 1]).all():
                    raise ValueError('Expected binary provider-imputed genotypes')
                window_id = int(window[start])
                priority = np.random.default_rng(seed + chrom*1000000 + window_id).permutation(len(raw))
                for name, train in jobs.items():
                    if counts[name] == 20:
                        continue
                    ix = eligible_indices(raw, train, priority)
                    if len(ix) != 100:
                        continue
                    values[name].append(raw[ix].T.astype(np.float32))
                    indices[name].extend((int(lo)+start+ix).tolist())
                    coords[name].extend(pos[start+ix].tolist())
                    blocks[name].extend([f'chr{chrom}:{window_id*10000+1}-{(window_id+1)*10000}']*100)
                    counts[name] += 1
                scanned += 1
                if all(n == 20 for n in counts.values()):
                    break
            if not all(n == 20 for n in counts.values()):
                raise ValueError(f'Insufficient eligible windows on chromosome {chrom}: {counts}')
            print(f'Prepared chr{chrom}: inspected {scanned} windows; 20 selected/job', flush=True)
    panels = {}
    for name in jobs:
        panels[name] = np.concatenate(values[name], axis=1)
        panels[name+'_indices'] = np.array(indices[name])
        panels[name+'_positions'] = np.array(coords[name])
        panels[name+'_blocks'] = np.array(blocks[name])
    return panels


def panel_columns(size):
    return np.concatenate([np.arange(b*100, b*100+size//100) for b in range(100)])


def make_model(family, params, trees, threads, seed):
    if family == 'rf':
        return RandomForestRegressor(n_estimators=trees, min_samples_leaf=5,
            max_features=params, random_state=seed, n_jobs=threads)
    return make_pipeline(StandardScaler(), Ridge(alpha=params, solver='cholesky'))


def gwas_table(x, y, snps, positions, block_labels):
    """Unadjusted additive association; two-sided t test, no structure covariates."""
    xc = x.astype(float) - x.mean(axis=0)
    yc = y - y.mean()
    ssx = (xc*xc).sum(axis=0)
    ssy = float(yc @ yc)
    cov = xc.T @ yc
    r2 = np.divide(cov*cov, ssx*ssy, out=np.zeros_like(cov), where=ssx*ssy > 0)
    t = np.sqrt(np.maximum(0, r2)*(len(y)-2)/np.maximum(1-r2, 1e-15))
    p = 2*student_t.sf(t, df=len(y)-2)
    beta = np.divide(cov, ssx, out=np.zeros_like(cov), where=ssx>0)
    table = pd.DataFrame(dict(snp_index=snps, position=positions, block=block_labels,
                              beta=beta, p_value=p))
    block = table.groupby('block', sort=False).agg(min_p=('p_value','min'), n_snps=('p_value','size'))
    block['within_block_bonferroni_p'] = np.minimum(1, block.min_p*block.n_snps)
    block['across_blocks_bonferroni_p'] = np.minimum(1, block.within_block_bonferroni_p*len(block))
    return table, block.reset_index()


def sampled_group_game(predict, background, evaluation, groups, permutations, seed):
    """Permutation Shapley over empirical marginal backgrounds; antithetic paths.

    Each complete path telescopes, so efficiency alone does not imply convergence.
    Return standard errors across independently sampled forward/reverse pairs.
    """
    if permutations < 4 or permutations % 4:
        raise ValueError('Permutation count must be a positive multiple of four >=4')
    flat = sorted(int(c) for g in groups for c in g)
    if flat != list(range(evaluation.shape[1])) or any(len(g)==0 for g in groups):
        raise ValueError('Groups must partition features')
    rng = np.random.default_rng(seed)
    nb, ne, ng = len(background), len(evaluation), len(groups)
    orders = [rng.permutation(ng) for _ in range(permutations//2)]
    pairs = np.zeros((ne, permutations//2, ng))
    base = float(np.mean(predict(background)))
    target = predict(evaluation)
    for row, x in enumerate(evaluation):
        for pair, order in enumerate(orders):
            for path in (order, order[::-1]):
                hybrids = np.empty((ng+1, nb, len(x)), dtype=np.float32)
                hybrids[0] = background
                for step, group in enumerate(path):
                    hybrids[step+1] = hybrids[step]
                    hybrids[step+1, :, groups[group]] = x[groups[group]][:, None]
                val = predict(hybrids.reshape(-1, len(x))).reshape(ng+1, nb).mean(axis=1)
                pairs[row, pair, path] += np.diff(val)/2
        if (row+1) % 20 == 0:
            print(f'  block SHAP: {row+1}/{ne} held-out accessions', flush=True)
    phi = pairs.mean(axis=1)
    se = pairs.std(axis=1, ddof=1)/np.sqrt(len(orders))
    if not np.allclose(phi.sum(axis=1), target-base, atol=1e-5, rtol=1e-5):
        raise ValueError('Sampled grouped Shapley efficiency failed')
    half = len(orders)//2
    return dict(phi=phi, mc_se=se, first_half=pairs[:,:half].mean(axis=1),
                second_half=pairs[:,half:].mean(axis=1), base=np.array(base), prediction=target)


def explain(out, model, x, train, test, ids, snps, positions, blocks, config, seed):
    import shap
    labels = list(dict.fromkeys(blocks.tolist()))
    groups = [np.flatnonzero(blocks == b) for b in labels]
    bg = np.random.default_rng(seed).choice(train, min(config['background'], len(train)), replace=False)
    game = sampled_group_game(model.predict, x[bg], x[test], groups,
                              config['permutations'], seed+1)
    write_npz(out/'group_shap.npz', **game, test_ids=ids[test], background_ids=ids[bg],
              block_labels=np.array(labels))
    # Standard individual TreeSHAP is a separate path-dependent explanation game.
    individual = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent').shap_values(x[test])
    if not np.isfinite(individual).all():
        raise ValueError('Nonfinite TreeSHAP')
    write_npz(out/'individual_tree_shap.npz', phi=individual, test_ids=ids[test], snp_indices=snps)
    pd.DataFrame(dict(snp_index=snps, position=positions, block=blocks,
        mean_abs_tree_shap=np.abs(individual).mean(axis=0))).to_csv(out/'snp_shap.csv', index=False)
    rows = []
    for j, label in enumerate(labels):
        rows.append(dict(block=label, n_snps=len(groups[j]),
            mean_abs_group_shap=np.abs(game['phi'][:,j]).mean(),
            mean_signed_group_shap=game['phi'][:,j].mean(),
            mean_mc_se=game['mc_se'][:,j].mean(),
            sum_mean_abs_individual_shap=np.abs(individual[:,groups[j]]).mean(axis=0).sum(),
            mean_abs_first_half=np.abs(game['first_half'][:,j]).mean(),
            mean_abs_second_half=np.abs(game['second_half'][:,j]).mean()))
    table = pd.DataFrame(rows)
    table.to_csv(out/'block_shap.csv', index=False)
    first = table.nlargest(10,'mean_abs_first_half').block
    second = table.nlargest(10,'mean_abs_second_half').block
    write_json(out/'explanation_checks.json', dict(
        efficiency_max_abs_error=np.max(np.abs(game['phi'].sum(axis=1)-(game['prediction']-game['base']))),
        split_half_rank_spearman=spearmanr(table.mean_abs_first_half,table.mean_abs_second_half).statistic,
        split_half_top10_overlap=len(set(first)&set(second))/10,
        permutations=config['permutations'], background=len(bg),
        individual_game='tree_path_dependent; distinct from grouped empirical marginal game'))


def run_fold(out, split, panels, ids, y, config):
    fold = split['fold']
    dest = out/f'fold_{fold}'
    dest.mkdir(exist_ok=True)
    done = dest/'COMPLETE.json'
    if done.exists():
        receipt = json.loads(done.read_text())
        if all((dest/p).exists() and digest(dest/p)==sha for p,sha in receipt['artifacts'].items()):
            print(f'Fold {fold}: verified and skipped', flush=True)
            return
        raise ValueError(f'Corrupt fold {fold} artifacts')
    fit, val, train, test = (np.array(split[k]) for k in ['fit','validation','train','test'])
    inner, outer = f'inner{fold}', f'outer{fold}'
    seed = config['seed']+100*fold
    metrics, tuning = [], []
    predictions = dict(test_ids=ids[test], y_true=y[test])
    for t, label in enumerate(LABELS):
        selected = {}
        for family, parameters in [('rf', ['sqrt', 0.1]), ('ridge', [1.,10.,100.,1000.,10000.])]:
            per_size = {}
            for size in SIZES:
                cols = panel_columns(size)
                xi = panels[inner][:,cols]
                candidates = []
                for params in parameters:
                    model = make_model(family, params, config['trees'],config['threads'], seed+t)
                    model.fit(xi[fit], y[fit,t])
                    mse = float(np.mean((model.predict(xi[val])-y[val,t])**2))
                    candidates.append((mse,params))
                    tuning.append(dict(trait=label, family=family, size=size, params=params, validation_mse=mse))
                best = min(candidates, key=lambda r:r[0])
                per_size[size] = best
            chosen = min(SIZES, key=lambda s:per_size[s][0])
            selected[family] = chosen
            for size in SIZES:
                cols = panel_columns(size)
                x = panels[outer][:,cols]
                model = make_model(family,per_size[size][1],config['trees'],config['threads'],seed+t)
                model.fit(x[train],y[train,t])
                pred = model.predict(x[test])
                predictions[f'{label}_{family}_{size}'] = pred
                if size==chosen:
                    predictions[f'{label}_{family}_selected'] = pred
                metrics.append(dict(fold=fold,trait=label,family=family,size=size,
                    selected=size==chosen,params=per_size[size][1],**regression_metrics(y[test,t],pred)))
                print(f'Fold {fold} {label} {family} {size}: R2={metrics[-1]["r2"]:.4f}; selected={size==chosen}',flush=True)
                if family=='rf' and size==chosen:
                    target = dest/label
                    target.mkdir(exist_ok=True)
                    joblib.dump(model,target/'selected_rf.joblib',compress=3)
                    explain(target,model,x,train,test,ids,panels[outer+'_indices'][cols],
                        panels[outer+'_positions'][cols],panels[outer+'_blocks'][cols],config,seed+t+2000)
        for size in SIZES:
            cols = panel_columns(size)
            table, block = gwas_table(panels[outer][train][:,cols],y[train,t],
                panels[outer+'_indices'][cols],panels[outer+'_positions'][cols],panels[outer+'_blocks'][cols])
            table.to_csv(dest/f'{label}_gwas_{size}.csv',index=False)
            block.to_csv(dest/f'{label}_gwas_blocks_{size}.csv',index=False)
        baseline = np.full(len(test),y[train,t].mean())
        predictions[f'{label}_mean'] = baseline
        metrics.append(dict(fold=fold,trait=label,family='mean',size=0,selected=True,params=None,
                            **regression_metrics(y[test,t],baseline)))
    write_npz(dest/'predictions.npz',**predictions)
    write_json(dest/'metrics.json',metrics)
    write_json(dest/'tuning.json',tuning)
    artifacts = {str(p.relative_to(dest)):digest(p) for p in sorted(dest.rglob('*')) if p.is_file() and p!=done}
    write_json(done,dict(fold=fold,artifacts=artifacts))


def summarize(out, ids, y, splits):
    completed = [s for s in splits if (out/f'fold_{s["fold"]}'/'COMPLETE.json').exists()]
    if not completed:
        return
    records = [r for s in completed for r in json.loads((out/f'fold_{s["fold"]}'/'metrics.json').read_text())]
    frame = pd.DataFrame(records)
    frame.to_csv(out/'metrics_by_fold.csv',index=False)
    summary = []
    for label in LABELS:
        ti = LABELS.index(label)
        for family in ['rf','ridge','mean']:
            sizes = SIZES+['selected'] if family!='mean' else ['selected']
            for size in sizes:
                truth, pred, differences = [], [], []
                for s in completed:
                    with np.load(out/f'fold_{s["fold"]}'/'predictions.npz') as a:
                        key = f'{label}_{family}' if family=='mean' else f'{label}_{family}_{size}'
                        truth.extend(a['y_true'][:,ti]); pred.extend(a[key])
                rows = frame[(frame.trait==label)&(frame.family==family)]
                rows = rows[rows.selected] if size=='selected' else rows[rows['size']==size]
                summary.append(dict(trait=label,family=family,size=size,
                    mean_fold_r2=rows.r2.mean(),sd_fold_r2=rows.r2.std(ddof=1),
                    **regression_metrics(np.array(truth),np.array(pred))))
    pd.DataFrame(summary).to_csv(out/'model_summary.csv',index=False)
    # Fixed physical coordinates allow fold comparisons; report coverage explicitly.
    stability = []
    for label in LABELS:
        tables = [(s['fold'],pd.read_csv(out/f'fold_{s["fold"]}'/label/'block_shap.csv')) for s in completed]
        for i,(fa,a) in enumerate(tables):
            for fb,b in tables[i+1:]:
                common = a.merge(b,on='block',suffixes=('_a','_b'))
                topa=set(a.nlargest(10,'mean_abs_group_shap').block)
                topb=set(b.nlargest(10,'mean_abs_group_shap').block)
                stability.append(dict(trait=label,fold_a=fa,fold_b=fb,common_blocks=len(common),
                    rank_spearman=spearmanr(common.mean_abs_group_shap_a,common.mean_abs_group_shap_b).statistic if len(common)>2 else None,
                    top10_overlap=len(topa&topb)/10))
    write_json(out/'block_stability.json',stability)
    selected_frame=pd.DataFrame(summary)
    selected_frame=selected_frame[selected_frame['size']=='selected']
    lines=['# Study 43: 10-kb block pilot','',f'Completed {len(completed)}/5 folds; 310 complete cases.',
        '', 'No population-structure adjustment. Blocks are physical windows, not measured LD.',
        '100 phenotype-blind blocks (20/chromosome); nested 1,000/5,000/10,000-SNP panels.',
        'All tuning and MAC filtering use fitting rows only. Reported selected-panel results use inner validation.',
        '', '| Trait | Model | Pooled held-out R2 | Mean fold R2 |', '|---|---|---:|---:|']
    for r in selected_frame.to_dict('records'):
        lines.append(f'| {r["trait"]} | {r["family"]} | {r["r2"]:.4f} | {r["mean_fold_r2"]:.4f} |')
    lines += ['', 'GWAS uses the same panels, training accessions, and unadjusted additive tests; it is not a genome-wide scan.',
        'Block GWAS p-values use within-block and across-block Bonferroni corrections.',
        'GroupSHAP explains selected RF predictions using empirical marginal backgrounds and permutation paths.',
        'Individual TreeSHAP uses a different, tree-path-dependent game; its block sums are a descriptive comparator.',
        'Inspect explanation_checks.json for Monte Carlo split-half stability; efficiency alone does not establish convergence.',
        'RF attribution is not an association p-value or evidence of causal epistasis. Selection and importance stability are exploratory.',
        'Without population adjustment, structure can contribute to prediction, GWAS associations, and block attribution.',
        'The fixed-seed experiment does not quantify model-seed uncertainty. Fold SD is descriptive.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write_json(out/'STATUS.json',dict(completed_folds=[s['fold'] for s in completed],complete=len(completed)==5))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--folds',type=int,nargs='+',default=[1,2,3,4,5])
    parser.add_argument('--trees',type=int,default=200)
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--permutations',type=int,default=16)
    parser.add_argument('--background',type=int,default=4)
    args=parser.parse_args()
    if args.permutations<4 or args.permutations%4 or min(args.trees,args.threads,args.background)<1 or not set(args.folds)<=set(range(1,6)):
        parser.error('Invalid computation budget or fold index')
    out=args.outdir.resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:
        parser.error('Nonempty output directory: use --resume')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.lock').open('a') as lock, threadpool_limits(limits=args.threads):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config=dict(seed=42,window_bp=10000,blocks_per_chr=20,mac=5,panel_sizes=SIZES,
            trees=args.trees,threads=args.threads,permutations=args.permutations,background=args.background,
            population_structure_adjustment=False,rf_max_features=['sqrt',0.1],rf_min_samples_leaf=5,
            ridge_alphas=[1.,10.,100.,1000.,10000.])
        identity=dict(config=config,source=digest(Path(__file__)),common_source=digest(Path(__file__).with_name('fivefold_common.py')),
            genotype_sha256=digest(GENO),phenotype_sha256=digest(PHENO),
            versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','shap','h5py','pandas']})
        signature=fingerprint(identity)
        manifest=out/'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text())['fingerprint']!=signature:
            raise ValueError('Changed inputs/code/config/environment; choose a fresh output directory')
        write_json(manifest,dict(**identity,fingerprint=signature))
        ids,rows,y,excluded=load_cohort()
        splits=make_splits(ids)
        write_json(out/'cohort_and_splits.json',dict(ids=ids,genotype_rows=rows,traits=TRAITS,y=y,excluded=excluded,splits=splits))
        if (out/'PREPARED.json').exists():
            receipt=json.loads((out/'PREPARED.json').read_text())
            if digest(out/'panels.npz')!=receipt['panels_sha256']:
                raise ValueError('Corrupt panel cache')
            with np.load(out/'panels.npz') as a:
                panels={k:a[k] for k in a.files}
        else:
            jobs={f'{stage}{s["fold"]}':s[role] for s in splits for stage,role in [('inner','fit'),('outer','train')]}
            panels=prepare_panels(rows,jobs,config['seed'])
            write_npz(out/'panels.npz',**panels)
            write_json(out/'PREPARED.json',dict(panels_sha256=digest(out/'panels.npz')))
        if args.prepare_only:
            return
        for s in splits:
            if s['fold'] in args.folds:
                run_fold(out,s,panels,ids,y,config)
                summarize(out,ids,y,splits)
        print(f'Report: {out / "REPORT.md"}',flush=True)


if __name__=='__main__':
    main()
