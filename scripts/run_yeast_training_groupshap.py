#!/usr/bin/env python3
"""Explain saved Bloom fold models on training and held-out samples."""
from __future__ import annotations

import argparse
import importlib.metadata
import itertools
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits

from fivefold_common import digest, fingerprint, write_json
from run_yeast_interactions import FastBinaryKAN, interaction_ranking
from run_yeast_rf_interactions import score_pair
from yeast_fivefold_common import ROOT, TRAITS, new_kan
from yeast_grouped_explanations import grouped_shapley_batch, grouped_interactions_batch

SOURCE = ROOT/'outputs/fivefold_v1/yeast_bloom_20260913_matched'
OUTPUT = ROOT/'outputs/yeast_bloom_training_groupshap'


class BinaryKANPredictor:
    """Exact saved binary KAN predictor, with optional CUDA batching."""
    def __init__(self, model, device):
        fast = FastBinaryKAN(model)
        self.device = device
        self.offset, self.delta = fast.offset.to(device), fast.delta.to(device)
        self.model = model.to(device).eval()

    def predict(self, x, batch=32768):
        values = []
        with torch.no_grad():
            for start in range(0, len(x), batch):
                xt = torch.as_tensor(x[start:start+batch], dtype=torch.float32, device=self.device)
                h = self.offset + xt @ self.delta
                h = self.model.subnode_scale[0]*h + self.model.subnode_bias[0]
                h = self.model.node_scale[0]*h + self.model.node_bias[0]
                out = self.model.act_fun[1](h)[0]
                out = self.model.subnode_scale[1]*out + self.model.subnode_bias[1]
                out = self.model.node_scale[1]*out + self.model.node_bias[1]
                values.append(out.cpu().numpy().ravel())
        return np.concatenate(values)


def correlation(a, b):
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(spearmanr(a, b).statistic)


def input_paths(source):
    paths = [source/'manifest.json', source/'interaction_manifest.json']
    for trait in TRAITS:
        td = source/trait
        paths.extend(td/name for name in ['cohort.json','splits.json','selected_features.json','panels.npz'])
        for fold in range(1, 6):
            fd = td/f'fold_{fold}'
            paths.append(fd/'shared.json')
            for method in ['rf','kan']:
                md = fd/f'{method}_42'
                paths.extend([md/('model.joblib' if method=='rf' else 'model.pt'),
                              md/'evaluation.npz',fd/'interactions'/f'{method}.csv'])
    return paths


def summarize(out):
    rows = {name: [] for name in ['group_summary_by_fold','group_comparison_by_fold',
        'group_interactions_by_fold','marker_comparison_by_fold']}
    for trait, fold, method in itertools.product(TRAITS, range(1,6), ['rf','kan']):
        fd = out/trait/f'fold_{fold}'/method
        if not (fd/'COMPLETE.json').exists(): continue
        for name in rows:
            rows[name].extend(json.loads((fd/f'{name}.json').read_text()))
    for name, records in rows.items():
        if records: pd.DataFrame(records).to_csv(out/f'{name}.tsv',sep='\t',index=False)
    return len(rows['group_comparison_by_fold'])


def run(source, out, folds, methods, jobs, permutations, device):
    out.mkdir(parents=True,exist_ok=True)
    parent = json.loads((source/'manifest.json').read_text())
    config = parent['config']
    identity = dict(source=str(source.relative_to(ROOT)), model_seed=42,
        input_sha256={str(p.relative_to(ROOT)):digest(p) for p in input_paths(source)},
        code_sha256={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),
            ROOT/'scripts/yeast_grouped_explanations.py', ROOT/'scripts/run_yeast_interactions.py',
            ROOT/'scripts/run_yeast_rf_interactions.py']},
        groups='chromosome partition of saved outer-fold selected SNPs',
        main_effect_samples='all training and all held-out observations',
        permutations=permutations, background='same saved 10 training observations',
        interaction_samples=20, interaction_mc=config['interaction_mc'],
        interaction_seed=config['interaction_seed'], training_selection_seed=13042,
        shapley_seed=5042, units='fold training-target SD; zero-offset attribution',
        interaction_convention='Shapley interaction index, no TreeSHAP one-half factor',
        marker_training='20 training observations excluding background, matched to historical held-out size',
        packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','torch','scikit-learn','joblib']},
        device=device)
    signature = fingerprint(identity)
    manifest = out/'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text())['fingerprint'] != signature:
            raise ValueError('Analysis identity differs: select a fresh output directory.')
    else: write_json(manifest,dict(**identity,fingerprint=signature))
    for trait in TRAITS:
        td = source/trait
        cohort = json.loads((td/'cohort.json').read_text())
        ids = np.array(cohort['sample_ids']); y = np.array(cohort['y'])
        lookup = {sample:i for i,sample in enumerate(ids)}
        selections = json.loads((td/'selected_features.json').read_text())
        with np.load(td/'panels.npz',allow_pickle=False) as panels:
            for fold in folds:
                fd = td/f'fold_{fold}'; shared = json.loads((fd/'shared.json').read_text())
                indices = {role:np.array([lookup[s] for s in shared[key]]) for role,key in
                    [('train','train_ids'),('heldout','test_ids'),('background','background_ids'),('interaction_heldout','interaction_evaluation_ids')]}
                assert not set(indices['train']) & set(indices['heldout'])
                assert set(indices['background']) <= set(indices['train'])
                assert set(indices['interaction_heldout']) <= set(indices['heldout'])
                eligible = np.setdiff1d(indices['train'],indices['background'])
                indices['interaction_train'] = np.random.default_rng(13042+fold).choice(
                    eligible,len(indices['interaction_heldout']),replace=False)
                x = panels[f'outer{fold}']; selection = selections[f'outer{fold}']
                labels = [s['marker_id'] for s in selection]
                names = sorted({s['chromosome'] for s in selection})
                groups = [[i for i,s in enumerate(selection) if s['chromosome']==name] for name in names]
                cohort_definition = dict(groups=[dict(name=name,indices=g,markers=[labels[i] for i in g]) for name,g in zip(names,groups)],
                    sample_ids={role:ids[ix].tolist() for role,ix in indices.items()},target_sd=shared['target_sd'])
                write_json(out/trait/f'fold_{fold}'/'design.json',cohort_definition)
                for method in methods:
                    target = out/trait/f'fold_{fold}'/method; target.mkdir(parents=True,exist_ok=True)
                    done = target/'COMPLETE.json'
                    if done.exists():
                        receipt = json.loads(done.read_text())
                        assert receipt['fingerprint']==signature
                        for name,sha in receipt['outputs_sha256'].items(): assert digest(target/name)==sha
                        print(f'{trait} fold {fold} {method}: verified and skipped',flush=True)
                        continue
                    started = time.perf_counter()
                    md = fd/f'{method}_42'
                    if method=='rf':
                        model = joblib.load(md/'model.joblib'); model.n_jobs=jobs
                        predict = model.predict
                    else:
                        model = new_kan(x.shape[1],42+100*fold)
                        model.load_state_dict(torch.load(md/'model.pt',map_location='cpu',weights_only=True))
                        fast = BinaryKANPredictor(model,device); predict = fast.predict
                    saved = np.load(md/'evaluation.npz',allow_pickle=False)
                    np.testing.assert_array_equal(saved['sample_ids'],ids[indices['heldout']])
                    np.testing.assert_allclose(predict(x[indices['heldout']])*shared['target_sd']+shared['target_mean'],
                        saved['predicted'],rtol=1e-4,atol=1e-5)
                    print(f'{trait} fold {fold} {method}: explaining {len(indices["train"])} train / {len(indices["heldout"])} held-out samples',flush=True)
                    group_rows=[]; interaction_rows=[]; results={}; halves={}; metrics={}
                    local_arrays={}
                    for role in ['train','heldout']:
                        result = grouped_shapley_batch(predict,x[indices['background']],x[indices[role]],groups,
                            n_permutations=permutations,seed=5042+fold,batch_size=65536,eval_chunk_size=16,return_halves=True)
                        results[role] = result
                        mean_abs = np.abs(result.phi).mean(0)
                        half_abs = np.abs(result.phi_halves).mean(1)
                        halves[role] = correlation(half_abs[0],half_abs[1])
                        metrics[role] = float(r2_score(y[indices[role]],result.predictions*shared['target_sd']+shared['target_mean']))
                        local_arrays[f'{role}_phi']=result.phi
                        local_arrays[f'{role}_phi_halves']=result.phi_halves
                        local_arrays[f'{role}_sample_ids']=ids[indices[role]]
                        for j,name in enumerate(names):
                            group_rows.append(dict(trait=trait,fold=fold,method=method,role=role,group=name,
                                mean_abs_phi=float(mean_abs[j]),mean_phi=float(result.phi[:,j].mean()),
                                sd_phi=float(result.phi[:,j].std(ddof=0)),n_eval=len(indices[role]),n_group_markers=len(groups[j])))
                        interaction = grouped_interactions_batch(predict,x[indices['background']],x[indices[f'interaction_{role}']],groups,
                            n_samples=config['interaction_mc'],seed=config['interaction_seed']+fold,batch_size=65536,eval_chunk_size=20)
                        local_arrays[f'{role}_group_interactions']=interaction.values
                        local_arrays[f'{role}_interaction_sample_ids']=ids[indices[f'interaction_{role}']]
                        for c,(j,k) in enumerate(interaction.pairs):
                            val = interaction.values[:,c]
                            interaction_rows.append(dict(trait=trait,fold=fold,method=method,role=role,
                                group_a=names[j],group_b=names[k],mean_interaction=float(val.mean()),
                                mean_abs_interaction=float(np.abs(val).mean()),n_eval=len(val)))
                    ma = {role:np.abs(results[role].phi).mean(0) for role in results}
                    top = {role:set(np.argsort(-values,kind='stable')[:5]) for role,values in ma.items()}
                    comparison = dict(trait=trait,fold=fold,method=method,n_train=len(indices['train']),n_heldout=len(indices['heldout']),
                        spearman_group_importance=correlation(ma['train'],ma['heldout']),top5_group_overlap=len(top['train']&top['heldout'])/5,
                        mean_abs_phi_train=float(ma['train'].mean()),mean_abs_phi_heldout=float(ma['heldout'].mean()),
                        training_r2=metrics['train'],heldout_r2=metrics['heldout'],
                        efficiency_max_abs_error=max(r.efficiency_max_error for r in results.values()),
                        efficiency_max_abs_error_train=results['train'].efficiency_max_error,
                        efficiency_max_abs_error_heldout=results['heldout'].efficiency_max_error,
                        mc_half_spearman_train=halves['train'],mc_half_spearman_heldout=halves['heldout'])
                    np.savez_compressed(target/'local_explanations.npz',group_names=np.array(names),**local_arrays)
                    print(f'{trait} fold {fold} {method}: group rank rho={comparison["spearman_group_importance"]:.3f}; computing training SNP interactions',flush=True)
                    if method=='rf':
                        model.n_jobs=1
                        pairs=list(itertools.combinations(range(len(labels)),2))
                        marker_rows=Parallel(n_jobs=jobs,backend='loky')(delayed(score_pair)(model,x[indices['background']],
                            x[indices['interaction_train']],labels,config['interaction_mc'],config['interaction_seed']+fold,pi,j,k)
                            for pi,(j,k) in enumerate(pairs))
                        ranking=pd.DataFrame(marker_rows).sort_values(['mean_abs_interaction','j_idx','k_idx'],ascending=[False,True,True],kind='mergesort')
                    else:
                        ranking=interaction_ranking(predict,x[indices['background']],x[indices['interaction_train']],labels,
                            config['interaction_mc'],config['interaction_seed']+fold,f'{trait} fold {fold} training {method}')
                    ranking.to_csv(target/'training_marker_interactions.csv',index=False)
                    historical=pd.read_csv(fd/'interactions'/f'{method}.csv')
                    joined=ranking.merge(historical,on=['j_idx','k_idx'],suffixes=('_train','_heldout'),validate='one_to_one')
                    assert len(joined)==len(ranking)==len(historical)
                    marker_comparison=dict(trait=trait,fold=fold,method=method,
                        spearman_interaction=correlation(joined.mean_abs_interaction_train,joined.mean_abs_interaction_heldout),
                        mean_abs_interaction_train=float(ranking.mean_abs_interaction.mean()),
                        mean_abs_interaction_heldout=float(historical.mean_abs_interaction.mean()),
                        n_eval_train=len(indices['interaction_train']),n_eval_heldout=len(indices['interaction_heldout']))
                    for k in [10,25,50,100]:
                        a=set(map(tuple,ranking.head(k)[['j_idx','k_idx']].to_numpy()))
                        b=set(map(tuple,historical.head(k)[['j_idx','k_idx']].to_numpy()))
                        marker_comparison[f'top{k}_overlap']=len(a&b)/k
                    for name,records in [('group_summary_by_fold',group_rows),('group_comparison_by_fold',[comparison]),
                        ('group_interactions_by_fold',interaction_rows),('marker_comparison_by_fold',[marker_comparison])]:
                        write_json(target/f'{name}.json',records)
                    artifacts=[p for p in target.iterdir() if p.name!='COMPLETE.json']
                    write_json(done,dict(fingerprint=signature,seconds=time.perf_counter()-started,
                        outputs_sha256={p.name:digest(p) for p in artifacts}))
                    completed=summarize(out)
                    print(f'{trait} fold {fold} {method}: complete in {time.perf_counter()-started:.1f}s; {completed}/20 models',flush=True)
    completed=summarize(out)
    assert all(digest(ROOT/p)==sha for p,sha in identity['input_sha256'].items()),'Original input changed'
    if completed==20:
        write_json(out/'COMPLETE.json',dict(status='complete',fingerprint=signature,models=20,original_inputs_unchanged=True,
            output_sha256={p.name:digest(p) for p in out.glob('*.tsv')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=SOURCE);p.add_argument('--outdir',type=Path,default=OUTPUT)
    p.add_argument('--folds',nargs='+',type=int,default=[1,2,3,4,5]);p.add_argument('--models',nargs='+',choices=['rf','kan'],default=['rf','kan'])
    p.add_argument('--jobs',type=int,default=8);p.add_argument('--permutations',type=int,default=32)
    p.add_argument('--device',choices=['cpu','cuda','auto'],default='auto')
    a=p.parse_args();device=('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):run(a.source.resolve(),a.outdir.resolve(),a.folds,a.models,a.jobs,a.permutations,device)
