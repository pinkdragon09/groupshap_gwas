"""Aggregate verified Study 27 held-out explanations with explicit selection coverage."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from fivefold_common import check_receipt, digest, save_receipt, write_json


def summarize(out):
    manifest = json.loads((out/'manifest.json').read_text())
    signature = manifest['fingerprint']
    if not check_receipt(out, 'COMPLETE.json', signature):
        raise ValueError('Requires a completed five-fold run')
    load = lambda name: json.loads((out/(name+'.json')).read_text())
    labels = ['chr1','chr2','chr3','chr4','chr5','PCs']
    canonical_pairs = [f'{a} x {b}' for a,b in itertools.combinations(labels, 2)]
    config = manifest['config']
    seeds = config['seeds']
    def aggregate(source, key, candidates, value):
        result = []
        for model in ['rf','transformer','kan']:
            for label in candidates:
                matrix, presence = [], []
                for fold in range(1,6):
                    values, found = [], False
                    for seed in seeds:
                        matches = [r for r in source if (r['model'],r['seed'],r['fold'],r[key]) == (model,seed,fold,label)]
                        if len(matches)>1:
                            raise ValueError('Duplicate explanation row')
                        found |= bool(matches)
                        values.append(value(matches[0]) if matches else 0.)
                    matrix.append(values)
                    presence.append(found)
                a = np.asarray(matrix)
                result.append(dict(model=model, **{key:label}, mean=float(a.mean()),
                    fold_sd=float(a.mean(1).std(ddof=1)), fold_means=a.mean(1),
                    seed_sd_within_fold=a.std(1, ddof=1),
                    folds_with_selected_features=[i+1 for i, p in enumerate(presence) if p],
                    absent_group_policy='zero contribution when no selected features belong to the group'))
        return result
    group = aggregate(load('attribution_by_fold_seed'), 'group', labels, lambda r:r['mean_abs'])
    pairs = aggregate(load('interaction_by_fold_seed'), 'pair', canonical_pairs, lambda r:r['mean_abs'])
    gpi = aggregate(load('gpi_by_fold_seed'), 'group', labels, lambda r:np.mean(r['delta_mse']))
    raw = load('snp_ranks_ale_by_fold_seed')
    snps = []
    for model in ['rf','transformer','kan']:
        for feature in sorted(set(r['feature'] for r in raw)):
            rows = [r for r in raw if r['model']==model and r['feature']==feature]
            folds = sorted(set(r['fold'] for r in rows))
            scores, contrasts, supported = [], [], []
            for fold in folds:
                rr = [r for r in rows if r['fold']==fold]
                if len(rr) != len(seeds):
                    raise ValueError('Incomplete seed coverage for a selected marker')
                scores.append(np.mean([np.mean(r['delta_mse']) for r in rr]))
                contrasts.append(np.mean([r['ale_levels'][1]-r['ale_levels'][0] for r in rr]))
                if all(0 < r['test_allele_frequency'] < 1 for r in rr):
                    supported.append(fold)
            snps.append(dict(model=model, feature=feature, snp_index=rows[0]['snp_index'],
                folds_selected=folds, selection_frequency=len(folds)/5,
                conditional_mean_delta_mse=np.mean(scores),
                conditional_fold_sd_delta_mse=np.std(scores,ddof=1) if len(folds)>1 else None,
                conditional_mean_binary_ale_contrast=np.mean(contrasts),
                folds_with_both_test_alleles=supported,
                missing_fold_policy='not selected; excluded from conditional means'))
    baselines = []
    for model in ['pc_only','training_mean']:
        r = [row for row in load('baseline_metrics') if row['model']==model]
        observed, predicted = [], []
        for fold in range(1,6):
            with np.load(out/f'fold_{fold}'/'baseline.npz') as a:
                observed.extend(a['observed'])
                predicted.extend(a[model])
        from fivefold_common import regression_metrics
        baselines.append(dict(model=model, mean_fold_r2=np.mean([a['r2'] for a in r]),
            sd_fold_r2=np.std([a['r2'] for a in r],ddof=1),
            pooled_oof_r2=regression_metrics(observed,predicted)['r2']))
    files = []
    for name, value in [('attribution_summary',group), ('interaction_summary',pairs),
                        ('gpi_summary',gpi), ('snp_selection_ale_summary',snps), ('baseline_summary',baselines)]:
        write_json(out/(name+'.json'), value)
        files.append(name+'.json')
    write_json(out/'summary_provenance.json', dict(source_sha256=digest(Path(__file__)),
        input_completion_sha256=digest(out/'COMPLETE.json'),
        aggregation='seeds averaged within folds; five fold means and sample SDs',
        conditional_marker_statistics='only folds selecting that marker; coverage saved explicitly'))
    save_receipt(out, 'SUMMARY_COMPLETE.json', signature, files+['summary_provenance.json'])
    print(f'Saved fold summaries for {len(snps)} model/locus combinations and both baselines.')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outdir',type=Path,required=True)
    summarize(p.parse_args().outdir.resolve())
