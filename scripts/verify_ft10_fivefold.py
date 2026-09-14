"""FT10 checkpoint, preprocessing and gene-game verification."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from fivefold_common import check_receipt, fit_standardizer, write_json
from run_study27_fivefold import inputs, new_model, predictor
from verify_study27_fivefold import verify as verify_predictive_checkpoints
from ft10_gene_explanations import load_genes, assign_genes, canonical_coverage


def verify(out):
    # This verifier uses saved cohort sizes/configuration and is also valid for FT10.
    verify_predictive_checkpoints(out)
    manifest=json.loads((out/'manifest.json').read_text())
    config, signature=manifest['config'],manifest['fingerprint']
    cohort=json.loads((out/'cohort.json').read_text())
    splits=json.loads((out/'splits.json').read_text())
    selection=json.loads((out/'selected_features.json').read_text())
    genes=load_genes(config['annotation'])
    y=np.array(cohort['y'])
    results=[]
    with np.load(out/'panels.npz',allow_pickle=False) as panels, threadpool_limits(limits=4):
        for s in splits:
            fold=s['fold']
            d=out/f'fold_{fold}'
            if not check_receipt(d,'COMPLETE.json',signature):
                continue
            shared=json.loads((d/'shared.json').read_text())
            labels,groups,assignments=assign_genes(selection[f'outer{fold}'],genes)
            if labels!=shared['gene_labels'] or assignments!=shared['gene_assignments']:
                raise ValueError('Gene partition mismatch')
            if canonical_coverage(selection[f'outer{fold}'],genes,assignments)!=shared['canonical_coverage']:
                raise ValueError('Canonical coverage mismatch')
            x,_=inputs(panels[f'outer{fold}'],panels[f'outer{fold}_pcs'],s['train'])
            mu,sd=fit_standardizer(y[s['train']])
            bg=np.array([cohort['sample_ids'].index(a) for a in shared['background_ids']])
            if set(bg)-set(s['train']):
                raise ValueError('Background leakage')
            for name in ['rf','transformer','kan']:
                for seed in config['seeds']:
                    md=d/f'{name}_{seed}'
                    if name=='rf':
                        model=joblib.load(md/'model.joblib')
                    else:
                        model=new_model(name,seed+100*fold)
                        model.load_state_dict(torch.load(md/'model.pt',weights_only=True,map_location='cpu'))
                        model.eval()
                    predict=predictor(model,mu,sd)
                    endpoint=predict(x[s['test']])-predict(x[bg]).mean()
                    with np.load(md/'gene_evaluation.npz',allow_pickle=False) as a:
                        np.testing.assert_array_equal(a['labels'],labels)
                        np.testing.assert_array_equal(a['sample_ids'],shared['test_ids'])
                        np.testing.assert_array_equal(a['background_ids'],shared['background_ids'])
                        rng=np.random.default_rng(4042+fold)
                        orders=np.array([[rng.permutation(len(groups)) for _ in bg] for _ in range(config['gene_blocks'])])
                        np.testing.assert_array_equal(a['orders'],orders)
                        np.testing.assert_allclose(endpoint,a['endpoint'],rtol=1e-6,atol=1e-5)
                        np.testing.assert_allclose(a['block_phi'].mean(0),a['phi'],atol=1e-8)
                        np.testing.assert_allclose(a['donor_phi'].mean(0),a['phi'],atol=1e-8)
                        np.testing.assert_allclose(a['phi'].sum(1),endpoint,rtol=1e-5,atol=1e-4)
                        np.testing.assert_allclose(a['block_phi'].std(0,ddof=1)/np.sqrt(config['gene_blocks']),a['local_mc_se'],atol=1e-8)
                        # Reproduce one whole gene's five held-out row permutations.
                        batches=[]
                        for order in a['permutation_rows']:
                            z=x[s['test']].copy()
                            z[:,groups[0]]=z[order][:,groups[0]]
                            batches.append(z)
                        pred=predict(np.concatenate(batches)).reshape(5,len(s['test']))
                        expected=((pred-y[s['test']])**2).mean(1)-((predict(x[s['test']])-y[s['test']])**2).mean()
                        np.testing.assert_allclose(expected,a['permutation_delta_mse'][0],rtol=1e-5,atol=1e-4)
                        results.append(dict(fold=fold,model=name,seed=seed,
                            max_efficiency_error=float(np.max(np.abs(a['phi'].sum(1)-endpoint)))))
    write_json(out/'gene_validation.json',dict(passed=True,checked_models=len(results),
        annotations_and_coverage_checked=True,mc_efficiency_and_se_checked=True,
        first_group_permutation_reproduced=True,results=results))
    print(f'Gene explanation checks passed: {len(results)}/45 models',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outdir',type=Path,required=True)
    verify(p.parse_args().outdir.resolve())
