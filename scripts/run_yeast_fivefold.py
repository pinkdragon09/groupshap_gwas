#!/usr/bin/env python3
"""Matched five-fold Bloom yeast RF/KAN prediction runner."""
from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from fivefold_common import (check_receipt, digest, fingerprint, fit_standardizer,
    make_splits, regression_metrics, save_receipt, validate_splits, write_json, write_npz)
from yeast_fivefold_common import (ROOT, TRAITS, GENO_2013, PHENO_2013,
    INTERACTIONS_2015_CSV, fit_kan, fit_rf, load_trait, map_references,
    new_kan, predict_kan, select_panel)


def prepare_trait(out, trait, signature, config):
    directory = out/trait
    directory.mkdir(exist_ok=True)
    if check_receipt(directory, 'PREPARED.json', signature):
        cohort=json.loads((directory/'cohort.json').read_text())
        splits=json.loads((directory/'splits.json').read_text())
        with np.load(directory/'panels.npz',allow_pickle=False) as z:
            panels=dict(z)
        validate_splits(splits,len(cohort['sample_ids']))
        return cohort,splits,panels
    ids,x,y,meta=load_trait(trait)
    splits=make_splits(ids,config['split_seed'],config['validation_fraction'])
    panels={}; selection={}; references={}
    for s in splits:
        for stage,role in [('outer','train'),('inner','fit')]:
            key=f'{stage}{s["fold"]}'
            ix, selected=select_panel(x[s[role]],y[s[role]],meta,
                config['max_markers'],config['min_maf'],config['prune_bp'])
            panels[key]=x[:,ix]
            panels[key+'_indices']=ix
            selection[key]=selected
            if stage=='outer':
                references[key]=map_references(trait,selected,config['mapping_distance_bp'])
    cohort=dict(sample_ids=ids,y=y,trait=trait,genotype_encoding='BY=0, RM=1',
        complete_markers=x.shape[1],excluded_phenotype_rows=1008-len(ids))
    write_json(directory/'cohort.json',cohort)
    write_json(directory/'splits.json',splits)
    write_json(directory/'selected_features.json',selection)
    write_json(directory/'reference_mapping.json',references)
    write_npz(directory/'panels.npz',**panels)
    save_receipt(directory,'PREPARED.json',signature,
        ['cohort.json','splits.json','selected_features.json','reference_mapping.json','panels.npz'])
    return cohort,splits,panels


def run_fold(out,trait,cohort,s,panels,signature,config):
    fold=s['fold']; directory=out/trait/f'fold_{fold}'
    directory.mkdir(exist_ok=True)
    if check_receipt(directory,'COMPLETE.json',signature):
        print(f'{trait} fold {fold}: verified and skipped',flush=True)
        return
    y=np.array(cohort['y']); ids=np.array(cohort['sample_ids'])
    tr,te,fit,val=(np.asarray(s[k]) for k in ['train','test','fit','validation'])
    x=panels[f'outer{fold}']; xi=panels[f'inner{fold}']
    mu,sd=fit_standardizer(y[tr]); imu,isd=fit_standardizer(y[fit])
    bg=np.random.default_rng(config['background_seed']+fold).choice(tr,config['background'],replace=False)
    ev=np.random.default_rng(config['evaluation_seed']+fold).choice(te,min(config['evaluation'],len(te)),replace=False)
    write_json(directory/'shared.json',dict(train_ids=ids[tr],test_ids=ids[te],fit_ids=ids[fit],
        validation_ids=ids[val],background_ids=ids[bg],interaction_evaluation_ids=ids[ev],
        outer_marker_count=x.shape[1],inner_marker_count=xi.shape[1],target_mean=mu,target_sd=sd,
        inner_target_mean=imu,inner_target_sd=isd))
    write_npz(directory/'baseline.npz',sample_ids=ids[te],observed=y[te],
              training_mean=np.repeat(mu,len(te)))
    receipt_files=['shared.json','baseline.npz']
    for seed in config['seeds']:
        actual=seed+100*fold
        for family in ['rf','kan']:
            model_dir=directory/f'{family}_{seed}'; model_dir.mkdir(exist_ok=True)
            receipt_files.append(f'{family}_{seed}/COMPLETE.json')
            if check_receipt(model_dir,'COMPLETE.json',signature):
                print(f'{trait} fold {fold} {family} seed {seed}: verified and skipped',flush=True)
                continue
            t0=time.perf_counter(); inner_trace=[]; outer_trace=[]; epochs=0
            if family=='rf':
                model=fit_rf(x[tr],(y[tr]-mu)/sd,actual,config['rf_trees'])
                model.n_jobs=1
                checkpoint='model.joblib'; joblib.dump(model,model_dir/checkpoint)
                pred=model.predict(x[te])*sd+mu
            else:
                _,epochs,inner_trace=fit_kan(xi[fit],(y[fit]-imu)/isd,actual,
                    config['epochs'],config['patience'],validation=(xi[val],(y[val]-imu)/isd))
                model,_,outer_trace=fit_kan(x[tr],(y[tr]-mu)/sd,actual,epochs=epochs,
                                             patience=config['patience'])
                checkpoint='model.pt'; torch.save(model.state_dict(),model_dir/checkpoint)
                pred=predict_kan(model,x[te],mu,sd)
            write_npz(model_dir/'evaluation.npz',sample_ids=ids[te],observed=y[te],predicted=pred)
            write_json(model_dir/'fit.json',dict(trait=trait,fold=fold,family=family,seed=seed,
                actual_seed=actual,selected_epochs=epochs,inner_trace=inner_trace,
                outer_trace=outer_trace,seconds=time.perf_counter()-t0,
                metrics=regression_metrics(y[te],pred)))
            save_receipt(model_dir,'COMPLETE.json',signature,[checkpoint,'evaluation.npz','fit.json'])
            print(f'{trait} fold {fold} {family} seed {seed}: '
                  f'R2={regression_metrics(y[te],pred)["r2"]:.4f} epochs={epochs}',flush=True)
    save_receipt(directory,'COMPLETE.json',signature,receipt_files)


def summarize(out,signature,config):
    rows=[]; summary=[]; baseline=[]; completed={}
    for trait in TRAITS:
        directory=out/trait
        cohort=json.loads((directory/'cohort.json').read_text())
        splits=json.loads((directory/'splits.json').read_text())
        y=np.array(cohort['y']); ids=np.array(cohort['sample_ids']); finished=[]
        records={(family,seed):[] for family in ['rf','kan'] for seed in config['seeds']}
        for s in splits:
            fold=s['fold']; d=directory/f'fold_{fold}'
            if not check_receipt(d,'COMPLETE.json',signature): continue
            finished.append(fold)
            with np.load(d/'baseline.npz') as a:
                baseline.append(dict(trait=trait,fold=fold,model='training_mean',
                    **regression_metrics(a['observed'],a['training_mean'])))
            for family,seed in records:
                md=d/f'{family}_{seed}'; check_receipt(md,'COMPLETE.json',signature)
                with np.load(md/'evaluation.npz') as a: record=dict(a)
                np.testing.assert_array_equal(record['sample_ids'],ids[s['test']])
                records[family,seed].append(record)
                rows.append(dict(trait=trait,fold=fold,family=family,seed=seed,
                    **regression_metrics(record['observed'],record['predicted'])))
        completed[trait]=finished
        if not finished:
            continue
        for family in ['rf','kan']:
            fold_r2=[np.mean([r['r2'] for r in rows if r['trait']==trait and r['family']==family and r['fold']==f]) for f in finished]
            fold_rmse=[np.mean([r['rmse'] for r in rows if r['trait']==trait and r['family']==family and r['fold']==f]) for f in finished]
            pooled=[]
            for seed in config['seeds']:
                rec=records[family,seed]
                all_ids=np.concatenate([r['sample_ids'] for r in rec])
                if len(set(all_ids))!=len(all_ids) or (len(finished)==5 and sorted(all_ids.tolist())!=cohort['sample_ids']):
                    raise ValueError('Invalid OOF coverage')
                pooled.append(regression_metrics(np.concatenate([r['observed'] for r in rec]),
                    np.concatenate([r['predicted'] for r in rec]))['r2'])
            summary.append(dict(trait=trait,family=family,completed_folds=len(finished),
                mean_fold_r2=np.mean(fold_r2),sd_fold_r2=np.std(fold_r2,ddof=1) if len(finished)>1 else None,
                mean_fold_rmse=np.mean(fold_rmse),mean_seed_pooled_oof_r2=np.mean(pooled),
                pooled_oof_r2_by_seed=pooled))
    write_json(out/'metrics_by_fold_seed.json',rows); write_json(out/'model_summary.json',summary)
    write_json(out/'baseline_metrics.json',baseline); write_json(out/'STATUS.json',dict(completed=completed))
    lines=['# Yeast Bloom matched five-fold prediction','',
        '| Trait | Model | Mean fold R2 | Fold SD | Mean RMSE | Mean-seed pooled OOF R2 |',
        '|---|---|---:|---:|---:|---:|']
    for r in summary:
        sd='NA' if r['sd_fold_r2'] is None else f'{r["sd_fold_r2"]:.4f}'
        lines.append(f'| {r["trait"]} | {r["family"]} | {r["mean_fold_r2"]:.4f} | {sd} | {r["mean_fold_rmse"]:.4f} | {r["mean_seed_pooled_oof_r2"]:.4f} |')
    lines += ['','RF and KAN use identical fold-selected, distance-pruned panels. Seeds are averaged within folds before fold summaries. Fold SD is descriptive. Interaction results are produced separately from the fixed seed-42 checkpoints.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    if all(len(completed[t])==5 for t in TRAITS):
        save_receipt(out,'COMPLETE.json',signature,['REPORT.md','model_summary.json','metrics_by_fold_seed.json']+
            [f'{t}/PREPARED.json' for t in TRAITS]+[f'{t}/fold_{f}/COMPLETE.json' for t in TRAITS for f in range(1,6)])
    print(f'Prediction status: {completed}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outdir',type=Path,required=True); p.add_argument('--resume',action='store_true')
    p.add_argument('--verify-only',action='store_true'); p.add_argument('--folds',nargs='+',type=int,default=[1,2,3,4,5])
    p.add_argument('--traits',nargs='+',choices=TRAITS,default=TRAITS)
    args=p.parse_args(); out=args.outdir.resolve()
    if not set(args.folds)<=set(range(1,6)): p.error('Invalid fold')
    if out.exists() and any(out.iterdir()) and not (args.resume or args.verify_only): p.error('Use --resume or fresh directory')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); torch.set_num_threads(4)
        with threadpool_limits(limits=4):
            config=dict(schema_version=1,study='Bloom2013_Neomycin_Lactate',traits=TRAITS,
                split_seed=42,validation_fraction=.2,seeds=[42,43,44],min_maf=.05,max_markers=100,
                prune_bp=100000,mapping_distance_bp=10000,rf_trees=200,epochs=200,patience=25,
                background=10,evaluation=20,interaction_mc=10,background_seed=2042,evaluation_seed=3042,
                interaction_seed=4042,threads=4,selection='training marginal Pearson test, MAF>=.05; descending r2/global-index ties; greedy same-chromosome 100kb pruning',
                kan=dict(width='[fold panel size,8,1]',grid=5,k=3,lr=.001,batch=64,target_standardized=True),
                interaction_models='seed 42 outer RF/KAN; same held-out samples, training backgrounds and coalitions')
            source=[Path(__file__).resolve(),Path(__file__).with_name('yeast_fivefold_common.py'),Path(__file__).with_name('fivefold_common.py'),Path(__file__).with_name('load_yeast_bloom.py')]
            identity=dict(config=config,inputs={str(q):digest(q) for q in [GENO_2013,PHENO_2013,INTERACTIONS_2015_CSV]},
                source={str(q.relative_to(ROOT)):digest(q) for q in source},
                versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','torch','scikit-learn','pykan','joblib']},python=platform.python_version())
            signature=fingerprint(identity)
            if (out/'manifest.json').exists():
                if json.loads((out/'manifest.json').read_text())['fingerprint']!=signature: raise ValueError('Run identity changed; use fresh directory')
            elif args.verify_only: raise ValueError('Missing manifest')
            else: write_json(out/'manifest.json',dict(**identity,fingerprint=signature,
                repository_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
            prepared={}
            for trait in TRAITS: prepared[trait]=prepare_trait(out,trait,signature,config)
            if not args.verify_only:
                for trait in args.traits:
                    cohort,splits,panels=prepared[trait]
                    for s in splits:
                        if s['fold'] in args.folds: run_fold(out,trait,cohort,s,panels,signature,config)
            summarize(out,signature,config)


if __name__=='__main__': main()
