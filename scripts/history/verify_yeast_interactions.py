#!/usr/bin/env python3
"""Verify every completed yeast RF/KAN interaction ranking."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from fivefold_common import check_receipt, digest, write_json
from run_yeast_interactions import FastBinaryKAN, pair_batch
from summarize_yeast_interactions import map_refs
from yeast_fivefold_common import TRAITS, new_kan


def reproduce(model,x,bg,ev,j,k,mc,seed,pair_index):
    pair_seed=seed+(pair_index+1)*(len(ev)+1)*1009
    batch=pair_batch(x[bg],x[ev],j,k,np.arange(x.shape[1]),mc,pair_seed)
    pred=model.predict(batch).reshape(len(ev),mc,4,len(bg)).mean(3)
    local=(pred[:,:,3]-pred[:,:,1]-pred[:,:,2]+pred[:,:,0]).mean(1)
    return np.array([local.mean(),np.abs(local).mean(),local.std(ddof=0)])


def run(out):
    parent=json.loads((out/'manifest.json').read_text()); config=parent['config']
    inter_manifest=json.loads((out/'interaction_manifest.json').read_text()); signature=inter_manifest['fingerprint']
    rf_manifest=json.loads((out/'rf_interaction_manifest.json').read_text()); rf_signature=rf_manifest['fingerprint']
    canonical=json.loads((out/'interaction_reference_mapping_canonical.json').read_text()); results=[]
    for trait in TRAITS:
        td=out/trait; cohort=json.loads((td/'cohort.json').read_text()); ids=np.array(cohort['sample_ids'])
        selected=json.loads((td/'selected_features.json').read_text()); lookup={v:i for i,v in enumerate(ids)}
        with np.load(td/'panels.npz',allow_pickle=False) as panels:
            for fold in range(1,6):
                shared=json.loads((td/f'fold_{fold}/shared.json').read_text())
                bg=np.array([lookup[v] for v in shared['background_ids']]); ev=np.array([lookup[v] for v in shared['interaction_evaluation_ids']])
                x=panels[f'outer{fold}'].copy(); expected_refs=map_refs(trait,selected[f'outer{fold}'])
                if canonical[f'{trait}_fold_{fold}'] != expected_refs:
                    raise ValueError(f'Canonical mapping mismatch: {trait} fold {fold}')
                inter=td/f'fold_{fold}/interactions'
                if not check_receipt(inter,'kan_COMPLETE.json',signature):
                    raise ValueError(f'KAN receipt failure: {trait} fold {fold}')
                rf_serial=inter/'rf_COMPLETE.json'; rf_parallel=inter/'rf_parallel_COMPLETE.json'
                if rf_serial.exists():
                    if not check_receipt(inter,rf_serial.name,signature):
                        raise ValueError(f'RF receipt failure: {trait} fold {fold}')
                    rf_receipt='serial'
                elif rf_parallel.exists():
                    receipt=json.loads(rf_parallel.read_text()); target=inter/'rf.csv'
                    if (receipt.get('fingerprint') != rf_signature or
                            receipt.get('csv_sha256') != digest(target) or
                            receipt.get('pairs') != x.shape[1]*(x.shape[1]-1)//2):
                        raise ValueError(f'Parallel RF receipt failure: {trait} fold {fold}')
                    rf_receipt='parallel'
                else:
                    raise ValueError(f'Missing RF receipt: {trait} fold {fold}')

                rf=joblib.load(td/f'fold_{fold}/rf_42/model.joblib'); rf.n_jobs=4
                kan=new_kan(x.shape[1],42+100*fold)
                kan.load_state_dict(torch.load(td/f'fold_{fold}/kan_42/model.pt',map_location='cpu',weights_only=True)); kan.eval()
                fast=FastBinaryKAN(kan)
                np.testing.assert_allclose(fast.predict(x),kan(torch.tensor(x)).detach().numpy().ravel(),rtol=1e-5,atol=1e-6)
                for family,model,receipt_type in [('rf',rf,rf_receipt),('kan',fast,'serial')]:
                    frame=pd.read_csv(inter/f'{family}.csv')
                    expected_pairs=x.shape[1]*(x.shape[1]-1)//2
                    if len(frame) != expected_pairs or not np.isfinite(frame[['mean_interaction','mean_abs_interaction','sd_interaction']]).all().all():
                        raise ValueError(f'Invalid ranking: {trait} fold {fold} {family}')
                    ordered=frame.sort_values(['mean_abs_interaction','j_idx','k_idx'],ascending=[False,True,True],kind='mergesort').reset_index(drop=True)
                    pd.testing.assert_frame_equal(frame.reset_index(drop=True),ordered)
                    row=frame.iloc[0]; j,k=int(row.j_idx),int(row.k_idx)
                    pair_index=sum(x.shape[1]-a-1 for a in range(j))+(k-j-1)
                    actual=reproduce(model,x,bg,ev,j,k,config['interaction_mc'],config['interaction_seed']+fold,pair_index)
                    expected=row[['mean_interaction','mean_abs_interaction','sd_interaction']].to_numpy(float)
                    np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-8)
                    results.append(dict(trait=trait,fold=fold,family=family,receipt=receipt_type,
                        verified_pair=[j,k],pairs=len(frame),max_error=float(np.max(np.abs(actual-expected)))))
    write_json(out/'interaction_validation.json',dict(passed=True,checked_rankings=len(results),
        canonical_mapping_checked=True,fast_kan_full_panel_prediction_checked=True,top_pair_recomputed=results))
    print(f'Passed full yeast interaction checks: {len(results)}/20 rankings',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--outdir',type=Path,required=True)
    run(p.parse_args().outdir.resolve())
