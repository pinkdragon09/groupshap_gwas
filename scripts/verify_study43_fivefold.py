"""Reload pilot checkpoints and independently validate predictions and scaling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from fivefold_common import check_receipt, digest, fit_standardizer, write_json
from run_study43_fivefold import FAMILIES, LABELS, ROOT, kan_predict, new_kan


def verify(out):
    manifest = json.loads((out/'manifest.json').read_text())
    config, signature = manifest['config'], manifest['fingerprint']
    for source, expected in manifest['source'].items():
        if digest(ROOT/source) != expected:
            raise ValueError(f'Executable source changed: {source}')
    if not check_receipt(out,'PREPARED.json',signature):
        raise ValueError('Missing preparation receipt')
    if not check_receipt(out,'COMPLETE.json',signature):
        raise ValueError('Pilot is not complete')
    torch.set_num_threads(config['threads'])
    cohort=json.loads((out/'cohort.json').read_text())
    splits=json.loads((out/'splits.json').read_text())
    selection=json.loads((out/'selected_features.json').read_text())
    y=np.array(cohort['y'])
    results=[]
    with np.load(out/'panels.npz',allow_pickle=False) as panels:
        for s in splits:
            fold=s['fold']
            d=out/f'fold_{fold}'
            check_receipt(d,'COMPLETE.json',signature)
            x=panels[f'outer{fold}']
            for name, train in [(f'outer{fold}',s['train']),(f'inner{fold}',s['fit'])]:
                expected=np.array([r['training_variance'] for r in selection[name]])
                observed=panels[name][train].astype(float).var(axis=0)
                np.testing.assert_allclose(observed,expected,atol=1e-12)
            mu,sd=fit_standardizer(y[s['train']])
            for family in FAMILIES:
                fd=d/family
                check_receipt(fd,'COMPLETE.json',signature)
                prep=json.loads((fd/'preprocessing.json').read_text())
                np.testing.assert_allclose(prep['target_mean'],mu)
                np.testing.assert_allclose(prep['target_sd'],sd)
                if family=='kan':
                    inner_mu,inner_sd=fit_standardizer(y[s['fit']])
                    np.testing.assert_allclose(prep['inner_target_mean'],inner_mu)
                    np.testing.assert_allclose(prep['inner_target_sd'],inner_sd)
                    models=[]
                    for t,label in enumerate(LABELS):
                        m=new_kan(x.shape[1],prep['seed']+t,config,fd)
                        m.load_state_dict(torch.load(fd/(label+'.pt'),map_location='cpu',weights_only=True))
                        m.eval()
                        models.append(m)
                    pred=kan_predict(models,x[s['test']])*sd+mu
                else:
                    models=joblib.load(fd/'models.joblib')
                    z=models[0].predict(x[s['test']]) if family=='joint_rf' else np.column_stack([m.predict(x[s['test']]) for m in models])
                    pred=z*sd+mu
                with np.load(fd/'evaluation.npz',allow_pickle=False) as a:
                    error=float(np.max(np.abs(pred-a['predicted'])))
                    np.testing.assert_allclose(pred,a['predicted'],atol=1e-7,rtol=1e-7)
                    if a['phi'].shape!=(len(s['test']),5,3) or a['interactions'].shape!=(len(s['test']),10,3):
                        raise ValueError('Wrong explanation dimensions')
                results.append(dict(fold=fold,family=family,checkpoint_prediction_max_error=error))
    write_json(out/'checkpoint_validation.json',dict(passed=True,checked_families=len(results),
        selected_marker_training_variance_checked=True,outer_and_inner_scaling_checked=True,
        source_hashes_checked=True,results=results))
    print(f'Passed checkpoint reproduction and preprocessing checks for {len(results)} fold/model families.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outdir',required=True,type=Path)
    args=p.parse_args()
    verify(args.outdir.resolve())
