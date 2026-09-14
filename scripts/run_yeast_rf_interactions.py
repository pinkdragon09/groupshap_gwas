#!/usr/bin/env python3
"""Parallel-across-pairs RF engine for the matched yeast interaction run."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from fivefold_common import digest, fingerprint, write_json
from run_yeast_interactions import pair_batch, recall_rows
from yeast_fivefold_common import TRAITS


def score_pair(model,bg,evaluation,labels,mc,seed,pair_index,j,k):
    pair_seed=seed+(pair_index+1)*(len(evaluation)+1)*1009
    batch=pair_batch(bg,evaluation,j,k,np.arange(len(labels)),mc,pair_seed)
    pred=model.predict(batch).reshape(len(evaluation),mc,4,len(bg)).mean(3)
    local=(pred[:,:,3]-pred[:,:,1]-pred[:,:,2]+pred[:,:,0]).mean(1)
    return dict(j_idx=j,k_idx=k,marker_j=labels[j],marker_k=labels[k],
        mean_interaction=float(local.mean()),mean_abs_interaction=float(np.abs(local).mean()),
        sd_interaction=float(local.std(ddof=0)))


def run(out,folds,jobs):
    parent=json.loads((out/'manifest.json').read_text()); config=parent['config']
    kan_manifest=json.loads((out/'interaction_manifest.json').read_text())
    identity=dict(parent_fingerprint=parent['fingerprint'],kan_protocol_fingerprint=kan_manifest['fingerprint'],
                  source_sha256=digest(Path(__file__)),engine='loky across pairs; RF n_jobs=1',jobs=jobs)
    signature=fingerprint(identity); write_json(out/'rf_interaction_manifest.json',dict(**identity,fingerprint=signature))
    for trait in TRAITS:
        td=out/trait; cohort=json.loads((td/'cohort.json').read_text()); ids=np.array(cohort['sample_ids'])
        splits=json.loads((td/'splits.json').read_text()); selections=json.loads((td/'selected_features.json').read_text())
        refs=json.loads((td/'reference_mapping.json').read_text())
        with np.load(td/'panels.npz',allow_pickle=False) as panels:
            for split in splits:
                fold=split['fold']
                if fold not in folds: continue
                fd=td/f'fold_{fold}'; inter=fd/'interactions'; inter.mkdir(exist_ok=True)
                target=inter/'rf.csv'; done=inter/'rf_parallel_COMPLETE.json'
                if done.exists() and json.loads(done.read_text())['fingerprint']==signature:
                    print(f'{trait} fold {fold} RF: verified and skipped',flush=True); continue
                shared=json.loads((fd/'shared.json').read_text()); lookup={v:i for i,v in enumerate(ids)}
                bg=np.array([lookup[v] for v in shared['background_ids']]); ev=np.array([lookup[v] for v in shared['interaction_evaluation_ids']])
                x=panels[f'outer{fold}']; labels=[r['marker_id'] for r in selections[f'outer{fold}']]
                model=joblib.load(fd/'rf_42/model.joblib'); model.n_jobs=1
                pairs=[(j,k) for j in range(len(labels)) for k in range(j+1,len(labels))]
                t0=time.perf_counter()
                rows=Parallel(n_jobs=jobs,backend='loky',verbose=0)(delayed(score_pair)(model,x[bg],x[ev],labels,
                    config['interaction_mc'],config['interaction_seed']+fold,pi,j,k) for pi,(j,k) in enumerate(pairs))
                frame=pd.DataFrame(rows).sort_values(['mean_abs_interaction','j_idx','k_idx'],ascending=[False,True,True],kind='mergesort')
                frame.to_csv(target,index=False)
                write_json(done,dict(fingerprint=signature,csv_sha256=digest(target),pairs=len(frame),seconds=time.perf_counter()-t0))
                print(f'{trait} fold {fold} RF: {len(frame)} pairs in {time.perf_counter()-t0:.1f}s',flush=True)
                kan_path=inter/'kan.csv'
                if kan_path.exists():
                    rankings={'rf':frame,'kan':pd.read_csv(kan_path)}
                    write_json(inter/'recall.json',recall_rows(trait,fold,rankings,selections[f'outer{fold}'],refs[f'outer{fold}']))
    metrics=[]
    for trait in TRAITS:
        for fold in range(1,6):
            path=out/trait/f'fold_{fold}/interactions/recall.json'
            if path.exists(): metrics.extend(json.loads(path.read_text()))
    write_json(out/'interaction_recall_by_fold.json',metrics)
    print(f'Interaction recall rows available: {len(metrics)}/120',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--folds',nargs='+',type=int,default=[1,2,3,4,5]); p.add_argument('--jobs',type=int,default=4)
    a=p.parse_args(); run(a.outdir.resolve(),a.folds,a.jobs)
