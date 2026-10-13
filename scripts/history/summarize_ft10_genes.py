"""Summarize gene-window coverage and held-out MC uncertainty without pooling seeds as folds."""
import argparse
import json
from pathlib import Path
import numpy as np
from fivefold_common import check_receipt, digest, save_receipt, write_json


def summarize(out):
    manifest=json.loads((out/'manifest.json').read_text())
    signature=manifest['fingerprint']
    rows, coverage, diagnostics=[],[],[]
    finished=[]
    for fold in range(1,6):
        d=out/f'fold_{fold}'
        if not check_receipt(d,'COMPLETE.json',signature):
            continue
        finished.append(fold)
        shared=json.loads((d/'shared.json').read_text())
        coverage.extend(dict(fold=fold,**r) for r in shared['canonical_coverage'])
        for model in ['rf','transformer','kan']:
            for seed in manifest['config']['seeds']:
                md=d/f'{model}_{seed}'
                if not check_receipt(md,'COMPLETE.json',signature):
                    raise ValueError('Missing model receipt')
                with np.load(md/'gene_evaluation.npz',allow_pickle=False) as a:
                    np.testing.assert_array_equal(a['labels'],shared['gene_labels'])
                    np.testing.assert_array_equal(a['sample_ids'],shared['test_ids'])
                    np.testing.assert_array_equal(a['background_ids'],shared['background_ids'])
                    np.testing.assert_allclose(a['block_phi'].mean(0),a['phi'],atol=1e-8)
                    np.testing.assert_allclose(a['phi'].sum(1),a['endpoint'],atol=1e-4)
                    key=dict(fold=fold,model=model,seed=seed)
                    order=np.argsort(-np.abs(a['phi']).mean(0),kind='stable')
                    rank=np.argsort(order)+1
                    for j,label in enumerate(a['labels']):
                        rows.append(dict(**key,group=str(label),mean_abs=float(np.abs(a['phi'][:,j]).mean()),
                            signed_mean=float(a['phi'][:,j].mean()),rank=int(rank[j]),
                            mean_local_mc_se=float(a['local_mc_se'][:,j].mean()),
                            half_budget_mean_abs_change=float(np.abs(a['block_phi'][:4,:,j].mean(0)-a['phi'][:,j]).mean()),
                            half_background_mean_abs_change=float(np.abs(a['donor_phi'][:8,:,j].mean(0)-a['phi'][:,j]).mean()),
                            permutation_delta_mse=a['permutation_delta_mse'][j]))
                    diagnostics.append(dict(**key,groups=len(a['labels']),paths=int(np.prod(a['orders'].shape[:2])),
                        max_efficiency_error=float(a['efficiency_error']),
                        half_budget_max_local_change=float(a['half_budget_max_local_change'])))
    summaries=[]
    for model in ['rf','transformer','kan']:
        for group in sorted(set(r['group'] for r in rows)):
            sub=[r for r in rows if r['model']==model and r['group']==group]
            folds=sorted(set(r['fold'] for r in sub))
            means=[np.mean([r['mean_abs'] for r in sub if r['fold']==f]) for f in folds]
            gpi=[np.mean([np.mean(r['permutation_delta_mse']) for r in sub if r['fold']==f]) for f in folds]
            summaries.append(dict(model=model,group=group,folds_present=folds,
                conditional_mean_abs=np.mean(means),conditional_fold_sd=np.std(means,ddof=1) if len(folds)>1 else None,
                conditional_mean_permutation_delta_mse=np.mean(gpi),
                mean_local_mc_se=np.mean([r['mean_local_mc_se'] for r in sub]),
                half_budget_mean_abs_change=np.mean([r['half_budget_mean_abs_change'] for r in sub]),
                half_background_mean_abs_change=np.mean([r['half_background_mean_abs_change'] for r in sub]),
                missing_policy='not represented in selected panel; excluded from conditional average'))
    files=[]
    for name,value in [('gene_by_fold_seed',rows),('gene_summary',summaries),
                        ('canonical_coverage',coverage),('gene_mc_diagnostics',diagnostics)]:
        write_json(out/(name+'.json'),value)
        files.append(name+'.json')
    write_json(out/'gene_summary_provenance.json',dict(source_sha256=digest(Path(__file__)),completed_folds=finished,
        uncertainty='MC SE conditional on the fixed finite background; seed/fold variability kept separate'))
    if len(finished)==5:
        save_receipt(out,'GENE_SUMMARY_COMPLETE.json',signature,files+['gene_summary_provenance.json'])
    print(f'Gene summaries: {len(finished)}/5 folds; {len(summaries)} model/group combinations',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outdir',type=Path,required=True)
    summarize(p.parse_args().outdir.resolve())
