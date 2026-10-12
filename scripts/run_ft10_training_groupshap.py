#!/usr/bin/env python3
"""Explain saved FT10 RF checkpoints on their training rows; compare held-out SHAP."""
import argparse
import csv
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from fivefold_common import digest, exact_group_game, regression_metrics, write_json, write_npz
from ft10_gene_explanations import permutation_path_game
from run_study27_fivefold import inputs, predictor


def read(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('outputs/fivefold_v1/ft10_20260913_inductive'))
    parser.add_argument('--outdir', type=Path, default=Path('outputs/ft10_training_groupshap_20261010'))
    args = parser.parse_args()
    source, out = args.source.resolve(), args.outdir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = read(source/'manifest.json')['config']
    cohort, splits = read(source/'cohort.json'), read(source/'splits.json')
    ids, y = np.array(cohort['sample_ids']), np.array(cohort['y'])
    provenance = {str(source/p): digest(source/p) for p in ['manifest.json', 'cohort.json', 'splits.json', 'panels.npz']}
    for p in [Path(__file__), Path(__file__).with_name('ft10_gene_explanations.py'), Path(__file__).with_name('fivefold_common.py'), Path(__file__).with_name('run_study27_fivefold.py')]:
        provenance[str(p.resolve())] = digest(p)
    rows, metrics, correlations = [], [], []
    with np.load(source/'panels.npz', allow_pickle=False) as panels, threadpool_limits(limits=4):
        for split in splits:
            fold = split['fold']
            tr, te = np.array(split['train']), np.array(split['test'])
            shared_path = source/f'fold_{fold}'/'shared.json'
            shared = read(shared_path)
            provenance[str(shared_path)] = digest(shared_path)
            np.testing.assert_array_equal(ids[tr], shared['train_ids'])
            assert not set(ids[tr]) & set(ids[te])
            bg = np.array([np.flatnonzero(ids == accession)[0] for accession in shared['background_ids']])
            assert set(bg) <= set(tr)
            x, prep = inputs(panels[f'outer{fold}'], panels[f'outer{fold}_pcs'], tr)
            for key in prep:
                np.testing.assert_allclose(prep[key], shared['outer_preprocessing'][key])
            chromosomes = panels[f'outer{fold}_chromosomes']
            chrom_labels = shared['groups']
            chrom_groups = [np.flatnonzero(chromosomes == label) for label in chrom_labels[:-1]] + [np.arange(200, 203)]
            gene_labels = shared['gene_labels']
            gene_groups = [np.array(g, dtype=int) for g in shared['gene_groups']]
            assert sorted(int(c) for g in gene_groups for c in g) == list(range(x.shape[1]))
            for seed in config['seeds']:
                directory = source/f'fold_{fold}'/f'rf_{seed}'
                for name in ['model.joblib', 'gene_evaluation.npz', 'evaluation.npz']:
                    provenance[str(directory/name)] = digest(directory/name)
                model = joblib.load(directory/'model.joblib')
                predict = predictor(model, shared['target_mean'], shared['target_sd'])
                pred = predict(x[tr])
                with np.load(directory/'evaluation.npz', allow_pickle=False) as held:
                    np.testing.assert_allclose(predict(x[te]), held['predicted'], rtol=1e-7, atol=1e-7)
                    held_chrom = held['phi'][..., 0].copy()
                    held_metrics = regression_metrics(y[te], held['predicted'])
                print(f'Fold {fold}, RF seed {seed}: training GroupSHAP ({len(tr)} rows)', flush=True)
                gene = permutation_path_game(predict, x[bg], x[tr], gene_groups, seed=4042+fold, blocks=config['gene_blocks'])
                chrom = exact_group_game(predict, x[bg], x[tr], chrom_groups)
                chrom_phi = chrom['phi'][..., 0]
                with np.load(directory/'gene_evaluation.npz', allow_pickle=False) as held:
                    np.testing.assert_array_equal(held['labels'], gene_labels)
                    np.testing.assert_array_equal(held['background_ids'], ids[bg])
                    np.testing.assert_array_equal(gene['orders'], held['orders'])
                    held_gene = held['phi'].copy()
                assert np.isfinite(gene['phi']).all() and np.isfinite(chrom['phi']).all()
                np.testing.assert_allclose(chrom_phi.sum(1), predict(x[tr])-predict(x[bg]).mean(), atol=1e-6)
                train_metrics = regression_metrics(y[tr], pred)
                metrics.append(dict(fold=fold, seed=seed, train_r2=train_metrics['r2'], test_r2=held_metrics['r2'], train_rmse=train_metrics['rmse'], test_rmse=held_metrics['rmse'], gene_efficiency_error=gene['efficiency_error'], chromosome_efficiency_error=float(np.max(np.abs(chrom_phi.sum(1)-gene['endpoint'])))))
                target = out/f'fold_{fold}'/f'rf_{seed}'
                write_npz(target/'gene_training.npz', sample_ids=ids[tr], background_ids=ids[bg], labels=np.array(gene_labels), observed=y[tr], predicted=pred, **gene)
                write_npz(target/'chromosome_training.npz', sample_ids=ids[tr], background_ids=ids[bg], labels=np.array(chrom_labels), **chrom)
                for level, labels, training_phi, test_phi in [('gene', gene_labels, gene['phi'], held_gene), ('chromosome', chrom_labels, chrom_phi, held_chrom)]:
                    train_abs, test_abs = np.abs(training_phi).mean(0), np.abs(test_phi).mean(0)
                    correlations.append(dict(fold=fold, seed=seed, level=level, rank_correlation=float(spearmanr(train_abs, test_abs).statistic)))
                    for j, label in enumerate(labels):
                        row = dict(fold=fold, seed=seed, level=level, group=label, n_snps=int(len((gene_groups if level == 'gene' else chrom_groups)[j])) if label != 'PCs' else 0, train_mean_abs=float(train_abs[j]), test_mean_abs=float(test_abs[j]), train_mean_signed=float(training_phi[:,j].mean()), test_mean_signed=float(test_phi[:,j].mean()))
                        if level == 'gene':
                            row.update(train_local_mc_se_mean=float(gene['local_mc_se'][:,j].mean()), train_half_budget_mean_abs_change=float(np.abs(gene['block_phi'][:4,:,j].mean(0)-gene['phi'][:,j]).mean()), train_half_background_mean_abs_change=float(np.abs(gene['donor_phi'][:8,:,j].mean(0)-gene['phi'][:,j]).mean()))
                        rows.append(row)
    # Equal weight to represented folds, after averaging seeds within each fold.
    summary = []
    for level in ['gene', 'chromosome']:
        for label in sorted({r['group'] for r in rows if r['level'] == level}):
            selected = [r for r in rows if r['level'] == level and r['group'] == label]
            folds = sorted({r['fold'] for r in selected})
            item = dict(level=level, group=label, represented_folds=len(folds), folds=','.join(map(str, folds)))
            for key in ['train_mean_abs', 'test_mean_abs', 'train_mean_signed', 'test_mean_signed']:
                values = [np.mean([r[key] for r in selected if r['fold'] == f]) for f in folds]
                item[key] = float(np.mean(values))
                item[key+'_fold_sd'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            summary.append(item)
    write_json(out/'per_fold_seed.json', rows)
    write_json(out/'metrics.json', metrics)
    write_json(out/'rank_correlations.json', correlations)
    write_json(out/'group_summary.json', summary)
    write_csv(out/'group_summary.csv', summary)
    write_csv(out/'metrics.csv', metrics)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, data, title in [(axes[0], [r for r in summary if r['level']=='chromosome'], 'Chromosome groups and PCs'), (axes[1], sorted([r for r in summary if r['level']=='gene' and r['group']!='PCs'], key=lambda r:r['train_mean_abs'], reverse=True)[:10], 'Top gene/intergenic groups by training attribution')]:
        pos = np.arange(len(data))
        ax.barh(pos+0.18, [r['train_mean_abs'] for r in data], height=0.36, label='Training (in-sample)')
        ax.barh(pos-0.18, [r['test_mean_abs'] for r in data], height=0.36, label='Held-out')
        ax.set_yticks(pos, [r['group']+f" ({r['represented_folds']}/5 folds)" for r in data], fontsize=9)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Mean absolute grouped Shapley (native FT10 units)')
        ax.legend(fontsize=9)
    fig.suptitle('FT10 saved random forests: training versus held-out attribution')
    fig.savefig(out/'training_vs_heldout_groupshap.png', dpi=180)
    fig.savefig(out/'training_vs_heldout_groupshap.pdf')
    plt.close(fig)
    train_fold = [np.mean([r['train_r2'] for r in metrics if r['fold']==f]) for f in range(1,6)]
    test_fold = [np.mean([r['test_r2'] for r in metrics if r['fold']==f]) for f in range(1,6)]
    pc = next(r for r in summary if r['level']=='gene' and r['group']=='PCs')
    report = ['# FT10 training-data GroupSHAP', '', 'Saved RF checkpoints only; no refitting. All 108 training rows per fold are explained, including the 16 background rows. Five folds and three seeds give 15 fitted models. Training rows recur across folds, so these are in-sample descriptions, not independent validation.', '', 'Gene/intergenic groups use the saved disjoint nearest-centre annotation partition with ±25 kb windows and an explicit PC group. Gene Shapley uses the same 16 training donors, 128 paths and group orders as the held-out analysis. Chromosome-plus-PC Shapley enumerates all coalitions. Values use native FT10 phenotype units.', '', f'Mean training R2: {np.mean(train_fold):.4f}; mean held-out R2: {np.mean(test_fold):.4f}.', f'PC group mean absolute gene-game attribution: training {pc["train_mean_abs"]:.4f}; held-out {pc["test_mean_abs"]:.4f}.', '', '## Largest gene/intergenic groups in training', '', '| Group | Represented folds | Training mean absolute SHAP | Held-out mean absolute SHAP |', '|---|---:|---:|---:|']
    top = sorted([r for r in summary if r['level']=='gene' and r['group']!='PCs'], key=lambda r:r['train_mean_abs'], reverse=True)[:10]
    report += [f'| {r["group"]} | {r["represented_folds"]}/5 | {r["train_mean_abs"]:.4f} | {r["test_mean_abs"]:.4f} |' for r in top]
    report += ['', 'Means average seeds within folds, then average only folds where a group is represented. Missing groups are unrepresented, not zero effects. Top groups are selected by training attribution; the held-out column is a descriptive comparison, not an independent test of that selection. Chromosome and gene games have different partitions, so gene-game magnitudes need not sum to chromosome-game magnitudes.', '', 'Canonical coverage is unchanged: FLC window coverage in fold 4 but never assigned; DOG1 window coverage in folds 4 and 5 but assigned only in fold 4; FRI, FT and VIN3 unrepresented. Attribution is model reliance, not evidence of causal effects or added SNP predictive value.', '', '## Validation', '', 'Every saved RF reproduced held-out predictions. Checked split/background membership, saved preprocessing, group partition, identical donor/path orders, finite attributions and local efficiency. Local arrays, Monte Carlo errors and budget/background sensitivity are saved. Efficiency is not proof of ranking convergence.', f'Maximum gene efficiency error: {max(r["gene_efficiency_error"] for r in metrics):.3g}.', f'Maximum chromosome efficiency error: {max(r["chromosome_efficiency_error"] for r in metrics):.3g}.']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    write_json(out/'provenance.json', dict(source=source, checkpoint_policy='saved RF models, no refitting', evaluation='all outer-training rows, including background donors', backgrounds=16, paths=128, seeds=config['seeds'], input_sha256=provenance, output_sha256={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file() and p.name!='provenance.json'}))
    print('\n'.join(report), flush=True)


if __name__ == '__main__':
    main()
