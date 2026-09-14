#!/usr/bin/env python3
"""Held-out matched pairwise interaction rankings for the yeast five-fold run."""
from __future__ import annotations

import argparse
import fcntl
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from fivefold_common import check_receipt, digest, fingerprint, save_receipt, write_json
from yeast_fivefold_common import ROOT, TRAITS, new_kan


class FastBinaryKAN:
    """Exact predictor for a numeric two-layer KAN on binary inputs."""
    def __init__(self, model):
        self.model = model.eval()
        p = model.width_in[0]
        endpoints = torch.stack([torch.zeros(p), torch.ones(p)])
        with torch.no_grad():
            _, _, edges, _ = model.act_fun[0](endpoints)
        # edges: endpoint x hidden x input
        edges = edges.permute(0, 2, 1)
        self.offset = edges[0].sum(0)
        self.delta = edges[1]-edges[0]

    def predict(self, x, batch=32768):
        values = []
        with torch.no_grad():
            for start in range(0, len(x), batch):
                xt = torch.as_tensor(x[start:start+batch], dtype=torch.float32)
                hidden = self.offset + xt @ self.delta
                hidden = self.model.subnode_scale[0] * hidden + self.model.subnode_bias[0]
                hidden = self.model.node_scale[0] * hidden + self.model.node_bias[0]
                out = self.model.act_fun[1](hidden)[0]
                out = self.model.subnode_scale[1] * out + self.model.subnode_bias[1]
                out = self.model.node_scale[1] * out + self.model.node_bias[1]
                values.append(out.numpy().ravel())
        return np.concatenate(values)


def pair_batch(bg, evaluation, j, k, candidates, mc, seed):
    n_eval, n_bg, p = len(evaluation), len(bg), bg.shape[1]
    block = 4*n_bg
    batch = np.empty((n_eval*mc*block, p), dtype=np.float32)
    rest = candidates[(candidates != j) & (candidates != k)]
    for e, point in enumerate(evaluation):
        rng = np.random.default_rng(seed+e)
        for sample in range(mc):
            size = int(rng.integers(0, len(rest)+1))
            coalition = rng.permutation(rest)[:size]
            base = (e*mc+sample)*block
            batch[base:base+n_bg] = bg
            if size:
                batch[base:base+n_bg, coalition] = point[coalition]
            batch[base+n_bg:base+2*n_bg] = batch[base:base+n_bg]
            batch[base+n_bg:base+2*n_bg, j] = point[j]
            batch[base+2*n_bg:base+3*n_bg] = batch[base:base+n_bg]
            batch[base+2*n_bg:base+3*n_bg, k] = point[k]
            batch[base+3*n_bg:base+4*n_bg] = batch[base+n_bg:base+2*n_bg]
            batch[base+3*n_bg:base+4*n_bg, k] = point[k]
    return batch


def interaction_ranking(predict, bg, evaluation, labels, mc, seed, progress=''):
    p = len(labels); candidates = np.arange(p); rows=[]
    pairs = [(j,k) for j in range(p) for k in range(j+1,p)]
    t0=time.perf_counter()
    for pair_index,(j,k) in enumerate(pairs):
        pair_seed=seed+(pair_index+1)*(len(evaluation)+1)*1009
        batch=pair_batch(bg,evaluation,j,k,candidates,mc,pair_seed)
        pred=predict(batch).reshape(len(evaluation),mc,4,len(bg)).mean(3)
        local=(pred[:,:,3]-pred[:,:,1]-pred[:,:,2]+pred[:,:,0]).mean(1)
        rows.append(dict(j_idx=j,k_idx=k,marker_j=labels[j],marker_k=labels[k],
            mean_interaction=float(local.mean()),mean_abs_interaction=float(np.abs(local).mean()),
            sd_interaction=float(local.std(ddof=0))))
        if (pair_index+1)%1000==0:
            print(f'{progress}: {pair_index+1}/{len(pairs)} pairs in {time.perf_counter()-t0:.1f}s',flush=True)
    frame=pd.DataFrame(rows)
    return frame.sort_values(['mean_abs_interaction','j_idx','k_idx'],ascending=[False,True,True],kind='mergesort').reset_index(drop=True)


def recall_rows(trait,fold,rankings,selection,references,ks=(10,25,50,100)):
    eligible=[r for r in references if r['eligible']]
    mapped=[tuple(sorted((int(r['q1_panel_index']),int(r['q2_panel_index'])))) for r in eligible]
    unique=set(mapped); collisions=len(mapped)-len(unique); total=len(references)
    p=len(selection); pairs=[(j,k) for j in range(p) for k in range(j+1,p)]
    neglog=np.array([r['neg_log10_p'] for r in selection])
    base=pd.DataFrame([dict(j_idx=j,k_idx=k,score=neglog[j]+neglog[k]) for j,k in pairs])
    base=base.sort_values(['score','j_idx','k_idx'],ascending=[False,True,True],kind='mergesort')
    frames={**rankings,'marginal_product':base}
    rows=[]
    for method,frame in frames.items():
        for k in ks:
            top={tuple(sorted((int(r.j_idx),int(r.k_idx)))) for r in frame.head(k).itertuples()}
            tp=len(top & unique); recovered=sum(pair in top for pair in mapped)
            rows.append(dict(trait=trait,fold=fold,method=method,k=k,total_reference_rows=total,
                eligible_reference_rows=len(eligible),unique_eligible_pairs=len(unique),collisions=collisions,
                true_positive_unique_pairs=tp,conditional_recall=tp/len(unique) if unique else None,
                recovered_reference_rows=recovered,unconditional_recovery=recovered/total))
    return rows


def run(out,folds,models):
    parent=json.loads((out/'manifest.json').read_text()); config=parent['config']
    source=Path(__file__).resolve()
    identity=dict(parent_fingerprint=parent['fingerprint'],source_sha256=digest(source),
        protocol=dict(seed_model=42,evaluation=config['evaluation'],background=config['background'],
                      mc=config['interaction_mc'],interaction_seed=config['interaction_seed'],ks=[10,25,50,100]))
    signature=fingerprint(identity); manifest=out/'interaction_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['fingerprint']!=signature:
        raise ValueError('Interaction identity changed; use a fresh prediction output')
    if not manifest.exists(): write_json(manifest,dict(**identity,fingerprint=signature))
    all_metrics=[]
    for trait in TRAITS:
        td=out/trait; cohort=json.loads((td/'cohort.json').read_text()); ids=np.array(cohort['sample_ids'])
        splits=json.loads((td/'splits.json').read_text()); selection=json.loads((td/'selected_features.json').read_text())
        refs=json.loads((td/'reference_mapping.json').read_text())
        with np.load(td/'panels.npz',allow_pickle=False) as panels:
            for s in splits:
                fold=s['fold']
                if fold not in folds: continue
                fd=td/f'fold_{fold}'; shared=json.loads((fd/'shared.json').read_text())
                idx={sample:i for i,sample in enumerate(ids)}
                bg=np.array([idx[a] for a in shared['background_ids']]); ev=np.array([idx[a] for a in shared['interaction_evaluation_ids']])
                x=panels[f'outer{fold}']; labels=[r['marker_id'] for r in selection[f'outer{fold}']]
                inter_dir=fd/'interactions'; inter_dir.mkdir(exist_ok=True)
                rankings={}
                for family in models:
                    target=inter_dir/f'{family}.csv'; receipt=f'{family}_COMPLETE.json'
                    if check_receipt(inter_dir,receipt,signature):
                        rankings[family]=pd.read_csv(target); print(f'{trait} fold {fold} {family}: verified and skipped',flush=True); continue
                    if family=='rf':
                        model=joblib.load(fd/'rf_42/model.joblib'); model.n_jobs=4; predictor=model.predict
                    else:
                        model=new_kan(x.shape[1],42+100*fold)
                        model.load_state_dict(torch.load(fd/'kan_42/model.pt',map_location='cpu',weights_only=True)); model.eval()
                        fast=FastBinaryKAN(model)
                        np.testing.assert_allclose(fast.predict(x[ev]),model(torch.tensor(x[ev])).detach().numpy().ravel(),rtol=1e-5,atol=1e-6)
                        predictor=fast.predict
                    ranking=interaction_ranking(predictor,x[bg],x[ev],labels,config['interaction_mc'],config['interaction_seed']+fold,
                                                f'{trait} fold {fold} {family}')
                    ranking.to_csv(target,index=False); save_receipt(inter_dir,receipt,signature,[target.name]); rankings[family]=ranking
                # Load other completed family when a model subset was requested.
                for family in ['rf','kan']:
                    if family not in rankings and check_receipt(inter_dir,f'{family}_COMPLETE.json',signature):
                        rankings[family]=pd.read_csv(inter_dir/f'{family}.csv')
                if set(rankings)=={'rf','kan'}:
                    rows=recall_rows(trait,fold,rankings,selection[f'outer{fold}'],refs[f'outer{fold}'])
                    write_json(inter_dir/'recall.json',rows); all_metrics.extend(rows)
                    save_receipt(inter_dir,'COMPLETE.json',signature,['rf_COMPLETE.json','kan_COMPLETE.json','recall.json'])
    # Include metrics from previously completed folds.
    all_metrics=[]
    for trait in TRAITS:
        for fold in range(1,6):
            path=out/trait/f'fold_{fold}/interactions/recall.json'
            if path.exists(): all_metrics.extend(json.loads(path.read_text()))
    write_json(out/'interaction_recall_by_fold.json',all_metrics)
    print(f'Interaction recall rows available: {len(all_metrics)}/120',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,required=True)
    parser.add_argument('--folds',nargs='+',type=int,default=[1,2,3,4,5])
    parser.add_argument('--models',nargs='+',choices=['rf','kan'],default=['rf','kan'])
    args=parser.parse_args(); out=args.outdir.resolve()
    with (out/'.interaction.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); torch.set_num_threads(4)
        with threadpool_limits(limits=4): run(out,args.folds,args.models)
