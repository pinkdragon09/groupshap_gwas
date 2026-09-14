#!/usr/bin/env python3
"""FT10 matched five-fold rerun with inductive selection and held-out gene explanations."""
from __future__ import annotations
import argparse
import csv
import fcntl
import importlib.metadata
import io
import json
import platform
import subprocess
import time
import zipfile
from pathlib import Path
import h5py
import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from fivefold_common import (check_receipt, digest, exact_group_game, fingerprint,
    fit_standardizer, make_splits, regression_metrics, save_receipt,
    validate_splits, write_json, write_npz)
from run_study27_fivefold import (ROOT, GENO, PHENO, MODELS, build_panels, inputs,
    fit_neural, predictor, diagnostics)
from ft10_gene_explanations import (load_genes, assign_genes, canonical_coverage,
    permutation_path_game)
GFF = ROOT / 'data/reference/Araport11_GFF3_genes_transposons.Mar92021.gff.gz'

def load_cohort(genotype, phenotype):
    with zipfile.ZipFile(phenotype) as z:
        rows = list(csv.DictReader(io.StringIO(z.read('1/study_1_values.csv').decode())))
    values, excluded = {}, []
    for row in rows:
        acc, value = row['accession_id'].strip(), row['FT10'].strip()
        if not acc:
            raise ValueError('Missing phenotype accession')
        if not value or not np.isfinite(float(value)):
            excluded.append(dict(accession=acc, reason='missing_or_nonfinite_phenotype'))
            continue
        value = float(value)
        if acc in values and values[acc] != value:
            raise ValueError('Conflicting replicate phenotype: ' + acc)
        values[acc] = value
    with h5py.File(genotype) as f:
        ids = f['accessions'][:].astype(str)
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate genotype accessions')
    lookup = {a: i for i, a in enumerate(ids)}
    for a in sorted(set(values) - set(lookup)):
        excluded.append(dict(accession=a, reason='no_genotype'))
    common = sorted(set(values) & set(lookup))
    return dict(sample_ids=common, genotype_rows=[lookup[a] for a in common],
                y=[values[a] for a in common], excluded=excluded, trait='FT10')


def prepare(out, config, signature):
    if check_receipt(out, 'PREPARED.json', signature):
        cohort = json.loads((out/'cohort.json').read_text())
        splits = json.loads((out/'splits.json').read_text())
        with np.load(out/'panels.npz', allow_pickle=False) as z:
            panels = dict(z)
        validate_splits(splits, len(cohort['y']))
        return cohort, splits, panels
    cohort = load_cohort(config['genotype'], config['phenotype'])
    if len(cohort['y']) != 135:
        raise ValueError('Expected 135 matched FT10 accessions')
    splits = make_splits(cohort['sample_ids'])
    jobs = {f'{stage}{s["fold"]}': s[role] for s in splits
            for stage, role in [('outer', 'train'), ('inner', 'fit')]}
    panels, selection, transforms = build_panels(config['genotype'], cohort['genotype_rows'],
                                                np.array(cohort['y']), jobs)
    for name, value in [('cohort', cohort), ('splits', splits),
                        ('selected_features', selection), ('pca_transforms', transforms)]:
        write_json(out/(name+'.json'), value)
    write_npz(out/'panels.npz', **panels)
    save_receipt(out, 'PREPARED.json', signature, ['cohort.json', 'splits.json',
                 'selected_features.json', 'pca_transforms.json', 'panels.npz'])
    return cohort, splits, panels


def run_fold(out, cohort, s, panels, config, signature):
    fold = s['fold']
    directory = out/f'fold_{fold}'
    directory.mkdir(exist_ok=True)
    if check_receipt(directory, 'COMPLETE.json', signature):
        for seed in config['seeds']:
            for name in MODELS:
                if not check_receipt(directory/f'{name}_{seed}', 'COMPLETE.json', signature):
                    raise ValueError('Missing model receipt')
        print(f'Fold {fold}: verified and skipped', flush=True)
        return
    y, ids = np.array(cohort['y']), np.array(cohort['sample_ids'])
    tr, te, it, va = (np.asarray(s[key]) for key in ['train', 'test', 'fit', 'validation'])
    x, prep = inputs(panels[f'outer{fold}'], panels[f'outer{fold}_pcs'], tr)
    xi, iprep = inputs(panels[f'inner{fold}'], panels[f'inner{fold}_pcs'], it)
    mu, sd = fit_standardizer(y[tr])
    imu, isd = fit_standardizer(y[it])
    chroms = panels[f'outer{fold}_chromosomes']
    labels = sorted(set(chroms)) + ['PCs']
    groups = [np.flatnonzero(chroms == label) for label in labels[:-1]] + [np.arange(200, 203)]
    selection = json.loads((out/'selected_features.json').read_text())[f'outer{fold}']
    genes = load_genes(config['annotation'])
    gene_labels, gene_groups, assignments = assign_genes(selection, genes)
    coverage = canonical_coverage(selection, genes, assignments)
    bg = np.random.default_rng(2042+fold).choice(tr, min(config['background'], len(tr)), replace=False)
    baseline = Ridge(alpha=1).fit(x[tr, 200:], y[tr])
    joblib.dump(baseline, directory/'pc_baseline.joblib')
    write_npz(directory/'baseline.npz', sample_ids=ids[te], observed=y[te],
              pc_only=baseline.predict(x[te, 200:]), training_mean=np.repeat(mu, len(te)))
    write_json(directory/'shared.json', dict(train_ids=ids[tr], test_ids=ids[te],
        fit_ids=ids[it], validation_ids=ids[va], background_ids=ids[bg], groups=labels,
        outer_preprocessing=prep, inner_preprocessing=iprep, target_mean=mu, target_sd=sd,
        inner_target_mean=imu, inner_target_sd=isd, gene_labels=gene_labels,
        gene_groups=gene_groups, gene_assignments=assignments, canonical_coverage=coverage))
    receipts = []
    for seed in config['seeds']:
        for name in MODELS:
            d = directory/f'{name}_{seed}'
            d.mkdir(exist_ok=True)
            receipts.append(f'{name}_{seed}/COMPLETE.json')
            if check_receipt(d, 'COMPLETE.json', signature):
                continue
            t0 = time.perf_counter()
            actual_seed = seed+100*fold
            trace, itrace, epoch = [], [], 0
            if name == 'rf':
                model = RandomForestRegressor(n_estimators=200, max_features='sqrt',
                    min_samples_leaf=2, random_state=actual_seed, n_jobs=config['threads'])
                model.fit(x[tr], (y[tr]-mu)/sd)
                checkpoint = 'model.joblib'
                joblib.dump(model, d/checkpoint)
            else:
                _, epoch, itrace = fit_neural(name, xi[it], (y[it]-imu)/isd, actual_seed, config,
                                                validation=(xi[va], (y[va]-imu)/isd))
                model, _, trace = fit_neural(name, x[tr], (y[tr]-mu)/sd, actual_seed, config, epochs=epoch)
                checkpoint = 'model.pt'
                torch.save(model.state_dict(), d/checkpoint)
            train_seconds = time.perf_counter()-t0
            predict = predictor(model, mu, sd)
            pred = predict(x[te])
            t0 = time.perf_counter()
            game = exact_group_game(predict, x[bg], x[te], groups)
            extra = diagnostics(predict, x[te], y[te], groups, 3042+fold, config['permutations'])
            print(f'Fold {fold} {name} seed {seed}: explaining {len(gene_groups)} gene/PC groups', flush=True)
            gene_game = permutation_path_game(predict, x[bg], x[te], gene_groups,
                                               seed=4042+fold, blocks=config['gene_blocks'])
            # Same five row permutations for every model and group in this fold.
            rng = np.random.default_rng(3042+fold)
            permutations = np.array([rng.permutation(len(te)) for _ in range(5)])
            batches = []
            for cols in gene_groups:
                for order in permutations:
                    z = x[te].copy()
                    z[:,cols] = x[te][order][:,cols]
                    batches.append(z)
            perm_pred = predict(np.concatenate(batches)).reshape(len(gene_groups),5,len(te))
            delta = ((perm_pred-y[te])**2).mean(2)-((pred-y[te])**2).mean()
            write_npz(d/'gene_evaluation.npz', sample_ids=ids[te], background_ids=ids[bg],
                labels=np.array(gene_labels), permutation_delta_mse=delta,
                permutation_rows=permutations, **gene_game)
            write_npz(d/'evaluation.npz', sample_ids=ids[te], observed=y[te], predicted=pred,
                background_ids=ids[bg], labels=np.array(labels), snp_indices=panels[f'outer{fold}_indices'],
                **game, **extra)
            write_json(d/'fit.json', dict(model=name, seed=seed, actual_seed=actual_seed,
                selected_epochs=epoch, inner_trace=itrace, outer_trace=trace,
                training_seconds=train_seconds, explanation_seconds=time.perf_counter()-t0,
                metrics=regression_metrics(y[te], pred)))
            save_receipt(d, 'COMPLETE.json', signature, [checkpoint, 'evaluation.npz', 'gene_evaluation.npz', 'fit.json'])
            print(f'Fold {fold} {name} seed {seed}: R2={regression_metrics(y[te], pred)["r2"]:.4f}, '
                  f'epochs={epoch}, train={train_seconds:.1f}s, explain={time.perf_counter()-t0:.1f}s', flush=True)
    save_receipt(directory, 'COMPLETE.json', signature,
                 ['shared.json', 'baseline.npz', 'pc_baseline.joblib']+receipts)


def summarize(out, cohort, splits, config, signature):
    rows, attribution, pairs, gpi, snps, baselines = [], [], [], [], [], []
    records = {(name, seed): [] for name in MODELS for seed in config['seeds']}
    finished = []
    selection = json.loads((out/'selected_features.json').read_text())
    for s in splits:
        fold, te = s['fold'], s['test']
        d = out/f'fold_{fold}'
        if not check_receipt(d, 'COMPLETE.json', signature):
            continue
        finished.append(fold)
        with np.load(d/'baseline.npz') as base:
            for name in ['pc_only', 'training_mean']:
                baselines.append(dict(fold=fold, model=name, **regression_metrics(base['observed'], base[name])))
        shared = json.loads((d/'shared.json').read_text())
        if set(shared['background_ids']) - set(shared['train_ids']):
            raise ValueError('Invalid background membership')
        for name, seed in records:
            md = d/f'{name}_{seed}'
            if not check_receipt(md, 'COMPLETE.json', signature):
                raise ValueError('Missing model receipt')
            with np.load(md/'evaluation.npz', allow_pickle=False) as z:
                a = dict(z)
            np.testing.assert_array_equal(a['sample_ids'], np.array(cohort['sample_ids'])[te])
            np.testing.assert_array_equal(a['observed'], np.array(cohort['y'])[te])
            np.testing.assert_array_equal(a['background_ids'], shared['background_ids'])
            np.testing.assert_allclose(a['phi'].sum(1), a['coalitions'][-1]-a['coalitions'][0], atol=1e-6)
            records[name, seed].append(a)
            key = dict(fold=fold, model=name, seed=seed)
            rows.append(dict(**key, **regression_metrics(a['observed'], a['predicted'])))
            for j, label in enumerate(a['labels']):
                attribution.append(dict(**key, group=label, mean_abs=float(np.abs(a['phi'][:, j]).mean())))
                gpi.append(dict(**key, group=label, delta_mse=a['group_permutation_delta_mse'][j].tolist()))
            for p, (j, k) in enumerate(a['pairs']):
                pairs.append(dict(**key, pair=f'{a["labels"][j]} x {a["labels"][k]}',
                                  mean_abs=float(np.abs(a['interactions'][:, p]).mean())))
            order = np.argsort(-a['snp_permutation_delta_mse'].mean(1), kind='stable')
            ranks = np.argsort(order)+1
            for j, feature in enumerate(selection[f'outer{fold}']):
                snps.append(dict(**key, **feature, heldout_permutation_rank=int(ranks[j]),
                    delta_mse=a['snp_permutation_delta_mse'][j], ale_levels=a['binary_ale_levels'][j],
                    test_allele_frequency=a['test_allele_frequency'][j]))
    summary, paired = [], []
    for name in MODELS:
        scores = [np.mean([r['r2'] for r in rows if r['model']==name and r['fold']==f]) for f in finished]
        rmses = [np.mean([r['rmse'] for r in rows if r['model']==name and r['fold']==f]) for f in finished]
        if not scores:
            continue
        pooled = []
        for seed in config['seeds']:
            rec = records[name, seed]
            ids = np.concatenate([a['sample_ids'] for a in rec])
            if len(set(ids)) != len(ids) or (len(finished)==5 and sorted(ids.tolist()) != cohort['sample_ids']):
                raise ValueError('Incorrect OOF coverage')
            pooled.append(regression_metrics(np.concatenate([a['observed'] for a in rec]),
                                              np.concatenate([a['predicted'] for a in rec]))['r2'])
        summary.append(dict(model=name, completed_folds=len(finished), seeds=config['seeds'],
            mean_fold_r2=np.mean(scores), sd_fold_r2=np.std(scores, ddof=1) if len(scores)>1 else None,
            mean_fold_rmse=np.mean(rmses), mean_seed_pooled_oof_r2=np.mean(pooled), pooled_oof_r2_by_seed=pooled))
    for f in finished:
        for a, b in [('transformer','rf'), ('kan','rf'), ('kan','transformer')]:
            delta = [next(r['r2'] for r in rows if (r['fold'],r['model'],r['seed'])==(f,a,seed)) -
                     next(r['r2'] for r in rows if (r['fold'],r['model'],r['seed'])==(f,b,seed)) for seed in config['seeds']]
            paired.append(dict(fold=f, comparison=f'{a} minus {b}', mean_seed_r2_difference=np.mean(delta),
                               per_seed_r2_difference=delta))
    features = {}
    for s in splits:
        for feature in selection[f'outer{s["fold"]}']:
            key = feature['feature']
            features.setdefault(key, dict(feature=key, snp_index=feature['snp_index'], folds_selected=[]))['folds_selected'].append(s['fold'])
    artifacts = dict(model_summary=summary, metrics_by_fold_seed=rows, baseline_metrics=baselines,
        attribution_by_fold_seed=attribution, interaction_by_fold_seed=pairs, gpi_by_fold_seed=gpi,
        snp_ranks_ale_by_fold_seed=snps, paired_differences=paired, selection_frequency=list(features.values()))
    for name, value in artifacts.items():
        write_json(out/(name+'.json'), value)
    lines = ['# FT10 inductive five-fold rerun', '', f'Completed folds: {finished}; 135 accessions; seeds {config["seeds"]}.', '',
        '| Model | Mean fold R2 | Fold SD | Mean RMSE | Mean seed pooled OOF R2 |',
        '|---|---:|---:|---:|---:|']
    for r in summary:
        sd = 'NA' if r['sd_fold_r2'] is None else f'{r["sd_fold_r2"]:.4f}'
        lines.append(f'| {r["model"]} | {r["mean_fold_r2"]:.4f} | {sd} | {r["mean_fold_rmse"]:.4f} | {r["mean_seed_pooled_oof_r2"]:.4f} |')
    lines += ['', 'Seeds are averaged within each fold before computing fold mean/SD. Pooled OOF R2 is computed separately per seed and then averaged, not from an ensemble.',
        'Every outer and inner fit has its own training-centered genome-wide PCA and phenotype-dependent 200-SNP selection. PCs are projected into held-out rows. Source genotypes were already imputed by the data provider.',
        'All test accessions are explained against 16 shared training backgrounds. Exact chromosome-plus-PC games enumerate at most 64 coalitions; missing chromosomes have no selected markers and are omitted from that fold game.',
        'GPI and SNP rankings use five shared held-out row permutations and delta-MSE. Negative values are retained. Binary one-interval ALE levels are centered on the test allele frequency; monomorphic test columns cannot support an empirical allele contrast.',
        'Fold SD is descriptive. Small test folds (27 accessions), weak prediction, correlated markers and interventional hybrids limit biological interpretation. These analyses do not establish causal epistasis.',
        'Historical full-cohort-selected and transductive-PC results remain separate.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write_json(out/'STATUS.json', dict(completed_folds=finished, complete=len(finished)==5))
    if len(finished)==5:
        save_receipt(out, 'COMPLETE.json', signature, ['PREPARED.json', 'REPORT.md'] +
                     [name+'.json' for name in artifacts] + [f'fold_{f}/COMPLETE.json' for f in finished])
    print(f'Validated {len(finished)}/5 folds; {out / "REPORT.md"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--folds', nargs='+', type=int, default=[1,2,3,4,5])
    args = parser.parse_args()
    if not args.folds or not set(args.folds) <= set(range(1,6)):
        parser.error('Invalid fold numbers')
    out = args.outdir.resolve()
    if out.exists() and any(out.iterdir()) and not (args.resume or args.verify_only):
        parser.error('Use a fresh output directory or --resume')
    out.mkdir(parents=True, exist_ok=True)
    with (out/'.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        with threadpool_limits(limits=4):
            config = dict(study='FT10', annotation=str(GFF), gene_blocks=8, gene_window_bp=25000, schema_version=1, genotype=str(GENO), phenotype=str(PHENO),
                seeds=[42,43,44], split_seed=42, validation_fraction=0.2, threads=4, epochs=100,
                patience=15, background=16, permutations=5, top_snps=200, n_pcs=3,
                pc_policy='training-centered unscaled binary genome-wide PCA; project held-out rows',
                selection='PC-adjusted marginal association, training MAC>=2; ties by global SNP index',
                rf=dict(trees=200, max_features='sqrt', min_samples_leaf=2),
                transformer=dict(width=32, heads=4, layers=1, dropout=0.1),
                kan=dict(width=[203,4,1], grid=3, spline_degree=3, symbolic=False, fixed_grid=True),
                neural_optimizer=dict(name='AdamW', lr=0.001, weight_decay=0.001, grad_clip=1,
                                      batch='full', refit='fresh fit for inner-selected epochs'))
            if args.verify_only:
                manifest = json.loads((out/'manifest.json').read_text())
                cohort, splits, _ = prepare(out, manifest['config'], manifest['fingerprint'])
                if (out/'COMPLETE.json').exists():
                    check_receipt(out, 'COMPLETE.json', manifest['fingerprint'])
                summarize(out, cohort, splits, manifest['config'], manifest['fingerprint'])
                return
            source_names = ['run_ft10_fivefold.py', 'ft10_gene_explanations.py', 'run_study27_fivefold.py', 'fivefold_common.py',
                            'transformer_mamba_gwas_template.py', 'grouped_shap_utils.py', 'snp_selection.py']
            print('Hashing raw inputs and executable source', flush=True)
            identity = dict(config=config, inputs={str(p):digest(p) for p in [GENO, PHENO, GFF]},
                source={f'scripts/{name}':digest(ROOT/'scripts'/name) for name in source_names},
                versions={p:importlib.metadata.version(p) for p in ['numpy','h5py','torch','scikit-learn','pykan','joblib']},
                python=platform.python_version())
            signature = fingerprint(identity)
            if (out/'manifest.json').exists():
                if json.loads((out/'manifest.json').read_text())['fingerprint'] != signature:
                    raise ValueError('Code, inputs, environment or configuration changed; use a fresh directory')
            else:
                write_json(out/'manifest.json', dict(**identity, fingerprint=signature,
                    repository_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                    started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
            cohort, splits, panels = prepare(out, config, signature)
            if not args.prepare_only:
                for split in splits:
                    if split['fold'] in args.folds:
                        run_fold(out, cohort, split, panels, config, signature)
                summarize(out, cohort, splits, config, signature)


if __name__ == '__main__':
    main()
