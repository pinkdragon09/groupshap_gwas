"""Combine FT10 prediction, baseline, gene coverage and uncertainty reports."""
import argparse
import json
from pathlib import Path
import numpy as np
from fivefold_common import check_receipt, digest, save_receipt, write_json
from summarize_study27_fivefold import summarize as summarize_panel
from summarize_ft10_genes import summarize as summarize_genes


def summarize(out):
    manifest=json.loads((out/'manifest.json').read_text())
    signature=manifest['fingerprint']
    if not check_receipt(out,'COMPLETE.json',signature):
        raise ValueError('Full five-fold completion required')
    # The panel summarizer is data-driven and has no cohort-specific computation.
    summarize_panel(out)
    summarize_genes(out)
    load=lambda name:json.loads((out/(name+'.json')).read_text())
    models=load('model_summary')
    pc_rf=json.loads((out/'pc_rf'/'summary.json').read_text())
    pc_rf_rows=json.loads((out/'pc_rf'/'metrics.json').read_text())
    from fivefold_common import fingerprint
    pc_protocol=json.loads((out/'pc_rf'/'manifest.json').read_text())
    if not check_receipt(out/'pc_rf','COMPLETE.json',fingerprint(pc_protocol)):
        raise ValueError('Completed matched PC-only RF baseline required')
    models=models+[pc_rf]
    baselines=load('baseline_summary')
    pc=next(r for r in baselines if r['model']=='pc_only')
    coverage=load('canonical_coverage')
    paired=[]
    base_rows=load('baseline_metrics')
    for r in load('metrics_by_fold_seed'):
        base=next(b for b in base_rows if b['model']=='pc_only' and b['fold']==r['fold'])
        paired.append(dict(model=r['model'],seed=r['seed'],fold=r['fold'],
            r2_difference=r['r2']-base['r2'],rmse_difference=r['rmse']-base['rmse']))
    write_json(out/'paired_vs_pc_baseline.json',paired)
    matched=[]
    for r in load('metrics_by_fold_seed'):
        base=next(b for b in pc_rf_rows if b['fold']==r['fold'] and b['seed']==r['seed'])
        matched.append(dict(model=r['model'],seed=r['seed'],fold=r['fold'],r2_difference=r['r2']-base['r2'],rmse_difference=r['rmse']-base['rmse']))
    write_json(out/'paired_vs_rf_pc_baseline.json',matched)
    lines=['# FT10 five-fold review', '',
        '135 accessions; five matched folds; three seeds per RF/Transformer/KAN; training-only SNP screening and fitted/projected PCs.', '',
        '| Model | Mean fold R2 | Fold SD | Pooled OOF R2* | Mean fold R2 minus PC-only ridge |',
        '|---|---:|---:|---:|---:|']
    for r in models:
        lines.append(f'| {r["model"]} | {r["mean_fold_r2"]:.4f} | {r["sd_fold_r2"]:.4f} | {r["mean_seed_pooled_oof_r2"]:.4f} | {r["mean_fold_r2"]-pc["mean_fold_r2"]:+.4f} |')
    for r in baselines:
        lines.append(f'| {r["model"]} | {r["mean_fold_r2"]:.4f} | {r["sd_fold_r2"]:.4f} | {r["pooled_oof_r2"]:.4f} | {r["mean_fold_r2"]-pc["mean_fold_r2"]:+.4f} |')
    rf=next(r for r in models if r['model']=='rf')
    lines+=['', f'Full RF minus matched PC-only RF: mean fold R2 difference {rf["mean_fold_r2"]-pc_rf["mean_fold_r2"]:+.4f}. See paired_vs_rf_pc_baseline.json for every fold and seed.', '', '*For seeded models, pooled OOF R2 is calculated separately per seed and then averaged. Fold statistics first average seeds within each fold. Fold SD is descriptive, not a confidence interval.', '',
        'The PC-only RF uses identical forest settings and seeds to the full RF; use their paired difference to assess added SNP predictive value. The linear PC-only ridge is a separate reference baseline.', '', '## Canonical locus coverage', '',
        '| Gene | Folds with selected SNPs in ±25 kb window | Folds actually assigned to gene group |',
        '|---|---|---|']
    for symbol in ['FLC','FRI','FT','VIN3','DOG1']:
        rows=[r for r in coverage if r['symbol']==symbol]
        physical=[r['fold'] for r in rows if r['window_selected_columns']]
        assigned=[r['fold'] for r in rows if r['assigned_columns']]
        lines.append(f'| {symbol} | {physical} | {assigned} |')
    lines+=['', 'Nearest-centre assignment is disjoint. A SNP can fall in a canonical window while being assigned to a neighboring gene. Missing groups are unrepresented in the selected panel, not evidence of no biological effect.', '',
        '## Monte Carlo diagnostics', '',
        'Each model explains every held-out accession against 16 training backgrounds using 128 balanced-donor permutation paths. MC standard errors condition on those backgrounds. Half-budget and half-background changes are descriptive sensitivity checks, not formal convergence guarantees.', '',
        '| Model | Median local MC SE averaged within group | Median 64-to-128-path local change | Median 8-to-16-donor local change |',
        '|---|---:|---:|---:|']
    gene=load('gene_by_fold_seed')
    for model in ['rf','transformer','kan']:
        rows=[r for r in gene if r['model']==model and r['group']!='PCs']
        values=[np.median([r[key] for r in rows]) for key in ['mean_local_mc_se','half_budget_mean_abs_change','half_background_mean_abs_change']]
        lines.append(f'| {model} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} |')
    lines+=['', '## Interpretation', '',
        'Assess predictive value against the matched PC-only baseline as well as absolute R2. This run changes SNP screening, PCA representation and split ordering relative to the historical analysis, so differences cannot be attributed to a single change.',
        'Gene rankings are conditional on selected features. Held-out permutation importance is marginal, not a fitted conditional-genotype test. Interventional replacement can break LD; no causal or epistasis discovery is established.',
        'Only chromosome-level interactions are computed; gene-group Shapley values are not gene-gene interaction estimates. Historical results remain unchanged.', '',
        '`gene_summary.json`, `canonical_coverage.json`, `gene_mc_diagnostics.json`, `paired_vs_pc_baseline.json`, and the panel summaries contain the numerical review evidence. Checkpoint binaries remain local.']
    (out/'REVIEW.md').write_text('\n'.join(lines)+'\n')
    sources=[Path(__file__),Path(__file__).with_name('summarize_study27_fivefold.py'),Path(__file__).with_name('summarize_ft10_genes.py')]
    write_json(out/'review_provenance.json',dict(source_sha256={p.name:digest(p) for p in sources}))
    save_receipt(out,'REVIEW_COMPLETE.json',signature,['SUMMARY_COMPLETE.json','GENE_SUMMARY_COMPLETE.json',
                 'paired_vs_pc_baseline.json','paired_vs_rf_pc_baseline.json','pc_rf/COMPLETE.json','REVIEW.md','review_provenance.json'])
    print(out/'REVIEW.md',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outdir',type=Path,required=True)
    summarize(p.parse_args().outdir.resolve())
