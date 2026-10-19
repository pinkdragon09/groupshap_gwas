#!/usr/bin/env python3
"""Candidate-informed Study 43 paired background/augmented analysis.

Published loci were identified in the same phenotype study: not independent
replication. Frozen 10,000-SNP background; three additional physical 10-kb blocks.
"""
import argparse
import fcntl
import importlib.metadata
import json
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from fivefold_common import digest, fingerprint, regression_metrics, validate_splits, write_json, write_npz
from run_study43_10kb import GENO, PHENO, LABELS, load_cohort, explain, gwas_table

ROOT=Path(__file__).resolve().parents[1]
REGIONS=[('AT4G14990',4,8560001,8570000,[8567936,8568437]),
         ('AT2G35970',2,15100001,15110000,[15109013]),
         ('delta13C_lead',2,15090001,15100000,[15094310])]


def varying_candidate_indices(raw,train):
    counts=raw[:,train].sum(axis=1)
    return np.flatnonzero((counts>0)&(counts<len(train)))


def load_candidates(genotype,rows):
    matrices=[]; indices=[]; positions=[]; blocks=[]; names=[]; sentinel=[]
    with h5py.File(genotype) as f:
        bounds=f['positions'].attrs['chr_regions']
        for gene,chrom,start,end,published in REGIONS:
            lo,hi=bounds[chrom-1]
            p=f['positions'][lo:hi]
            a,b=np.searchsorted(p,[start,end+1])
            ix=np.arange(int(lo)+a,int(lo)+b)
            g=f['snps'][int(lo)+a:int(lo)+b,:][:,rows].T.astype(np.float32)
            assert np.isin(g,[0,1]).all()
            matrices.append(g);indices.extend(ix);positions.extend(p[a:b])
            blocks.extend([f'chr{chrom}:{start}-{end}']*len(ix));names.extend([gene]*len(ix))
            for marker in published:
                hits=np.flatnonzero(p[a:b]==marker)
                if len(hits)!=1:
                    raise ValueError(f'Published SNP absent or duplicate: chr{chrom}:{marker}')
                column=g[:,hits[0]]
                sentinel.append(dict(gene=gene,chromosome=chrom,position=marker,
                    global_snp_index=int(ix[hits[0]]),cohort_minor_allele_count=int(min(column.sum(),len(column)-column.sum()))))
    return dict(x=np.concatenate(matrices,axis=1),indices=np.array(indices),positions=np.array(positions),
                blocks=np.array(blocks),genes=np.array(names)),sentinel


def panel(panels,candidates,name,train,augmented):
    x=panels[name]
    metadata={key:panels[name+'_'+key] for key in ['indices','positions','blocks']}
    if augmented:
        ix=varying_candidate_indices(candidates['x'].T,train)
        if np.intersect1d(metadata['indices'],candidates['indices'][ix]).size:
            raise ValueError('Candidate region overlaps background; de-duplicate before fitting')
        x=np.concatenate([x,candidates['x'][:,ix]],axis=1)
        metadata={key:np.r_[metadata[key],candidates[key][ix]] for key in metadata}
    return x,metadata


def model(family,params,config,seed):
    if family=='rf':
        return RandomForestRegressor(n_estimators=config['trees'],max_features=params[0],
            min_samples_leaf=params[1],random_state=seed,n_jobs=config['threads'])
    return make_pipeline(StandardScaler(),Ridge(alpha=params,solver='cholesky'))


def run_fold(out,split,panels,candidates,ids,y,config):
    fold=split['fold']; dest=out/f'fold_{fold}'; dest.mkdir(exist_ok=True)
    complete=dest/'COMPLETE.json'
    if complete.exists():
        receipt=json.loads(complete.read_text())
        for p,sha in receipt['artifacts'].items():
            assert digest(dest/p)==sha
        print(f'Fold {fold} verified and skipped',flush=True)
        return
    fit,val,train,test=[np.array(split[k]) for k in ['fit','validation','train','test']]
    metrics=[];tuning=[]; predictions=dict(test_ids=ids[test],y_true=y[test])
    selection=[]
    for arm,augment in [('background',False),('augmented',True)]:
        xi,mi=panel(panels,candidates,f'inner{fold}',fit,augment)
        x,meta=panel(panels,candidates,f'outer{fold}',train,augment)
        write_npz(dest/f'{arm}_panel.npz',**meta,inner_indices=mi['indices'])
        for t,label in enumerate(LABELS):
            target=dest/arm/label;target.mkdir(parents=True,exist_ok=True)
            for family,grid in [('rf',[(m,l) for m in ['sqrt',0.1] for l in [1,5]]),
                                ('ridge',[1.,10.,100.,1000.,10000.])]:
                candidates_to_fit=[]
                for params in grid:
                    estimator=model(family,params,config,42+fold*100+t)
                    estimator.fit(xi[fit],y[fit,t])
                    loss=float(np.mean((estimator.predict(xi[val])-y[val,t])**2))
                    tuning.append(dict(fold=fold,arm=arm,trait=label,family=family,params=params,validation_mse=loss))
                    candidates_to_fit.append((loss,params))
                loss,params=min(candidates_to_fit,key=lambda v:v[0])
                estimator=model(family,params,config,42+fold*100+t)
                estimator.fit(x[train],y[train,t])
                prediction=estimator.predict(x[test])
                predictions[f'{arm}_{label}_{family}']=prediction
                metrics.append(dict(fold=fold,arm=arm,trait=label,family=family,n_snps=x.shape[1],
                    params=params,**regression_metrics(y[test,t],prediction)))
                print(f'Fold {fold} {arm} {label} {family}: {x.shape[1]} SNPs, R2={metrics[-1]["r2"]:.4f}, params={params}',flush=True)
                if family=='rf' and augment:
                    joblib.dump(estimator,target/'rf.joblib',compress=3)
                    explain(target,estimator,x,train,test,ids,meta['indices'],meta['positions'],meta['blocks'],config,42+fold*100+t+2000)
            snps,block=gwas_table(x[train],y[train,t],meta['indices'],meta['positions'],meta['blocks'])
            snps.to_csv(target/'gwas_snps.csv',index=False)
            block.to_csv(target/'gwas_blocks.csv',index=False)
            if augment:
                attribution=pd.read_csv(target/'block_shap.csv')
                merged=attribution.merge(block,on='block',validate='one_to_one',suffixes=('_shap','_gwas'))
                merged['groupshap_rank']=merged.mean_abs_group_shap.rank(ascending=False,method='min')
                merged['gwas_rank']=merged.min_p.rank(method='min')
                merged['fold']=fold;merged['trait']=label
                candidate_labels=set(candidates['blocks'])
                picked=merged[merged.block.isin(candidate_labels)]
                assert len(picked)==3
                selection.extend(picked.to_dict('records'))
                merged.to_csv(target/'block_comparison.csv',index=False)
    write_json(dest/'metrics.json',metrics);write_json(dest/'tuning.json',tuning)
    write_json(dest/'candidate_ranks.json',selection)
    write_npz(dest/'predictions.npz',**predictions)
    receipt={str(p.relative_to(dest)):digest(p) for p in sorted(dest.rglob('*')) if p.is_file() and p!=complete}
    write_json(complete,dict(fold=fold,artifacts=receipt))


def summarize(out,splits):
    done=[s for s in splits if (out/f'fold_{s["fold"]}'/'COMPLETE.json').exists()]
    if not done:return
    records=[r for s in done for r in json.loads((out/f'fold_{s["fold"]}'/'metrics.json').read_text())]
    frame=pd.DataFrame(records);frame.to_csv(out/'metrics_by_fold.csv',index=False)
    ranks=[r for s in done for r in json.loads((out/f'fold_{s["fold"]}'/'candidate_ranks.json').read_text())]
    pd.DataFrame(ranks).to_csv(out/'candidate_ranks.csv',index=False)
    summary=[]
    for arm in ['background','augmented']:
        for t,label in enumerate(LABELS):
            for family in ['rf','ridge']:
                truth=[];pred=[]
                for s in done:
                    with np.load(out/f'fold_{s["fold"]}'/'predictions.npz') as a:
                        truth.extend(a['y_true'][:,t]);pred.extend(a[f'{arm}_{label}_{family}'])
                rows=frame[(frame.arm==arm)&(frame.trait==label)&(frame.family==family)]
                summary.append(dict(arm=arm,trait=label,family=family,mean_fold_r2=rows.r2.mean(),
                    **regression_metrics(np.array(truth),np.array(pred))))
    result=pd.DataFrame(summary);result.to_csv(out/'model_summary.csv',index=False)
    lines=['# Study 43: published candidate regions','',f'Completed {len(done)}/5 folds.',
        '', 'Candidate-informed analysis: published regions were discovered using this same study. This is not independent replication.',
        'No population adjustment. Frozen 10,000-SNP background plus all fitting-polymorphic SNPs in three 10-kb candidate blocks.',
        'RF and ridge get matched inputs; identical splits and tuning grids in background and augmented arms.',
        'RF tuning permits leaf sizes 1 and 5 to accommodate rare alleles. GWAS uses training rows only.',
        '', '| Trait | Model | Background pooled R2 | Augmented pooled R2 |','|---|---|---:|---:|']
    for label in LABELS:
        for family in ['rf','ridge']:
            a=result[(result.trait==label)&(result.family==family)].set_index('arm')
            lines.append(f'| {label} | {family} | {a.loc["background","r2"]:.4f} | {a.loc["augmented","r2"]:.4f} |')
    lines+=['','See candidate_ranks.csv for separate GroupSHAP and GWAS ranks of the three added blocks.',
        'Rank differences do not establish causality. Added blocks can contain unequal SNP counts; GWAS also reports within/across-block Bonferroni p-values.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write_json(out/'STATUS.json',dict(completed_folds=[s['fold'] for s in done],complete=len(done)==5))


def verify(out,splits,panels,candidates,ids,y):
    for split in splits:
        fold=split['fold'];dest=out/f'fold_{fold}'
        receipt=json.loads((dest/'COMPLETE.json').read_text())
        for p,sha in receipt['artifacts'].items():assert digest(dest/p)==sha,p
        train=np.array(split['train']);test=np.array(split['test'])
        for arm,augment in [('background',False),('augmented',True)]:
            x,meta=panel(panels,candidates,f'outer{fold}',train,augment)
            with np.load(dest/f'{arm}_panel.npz') as a:
                np.testing.assert_array_equal(a['indices'],meta['indices'])
            if augment:
                counts=x[train,10000:].sum(axis=0)
                assert ((counts>0)&(counts<len(train))).all()
                assert len(np.unique(meta['blocks']))==103
        with np.load(dest/'predictions.npz') as a:
            np.testing.assert_array_equal(a['test_ids'],ids[test])
            np.testing.assert_array_equal(a['y_true'],y[test])
            metrics=json.loads((dest/'metrics.json').read_text())
            tuning=json.loads((dest/'tuning.json').read_text())
            for r in metrics:
                options=[v for v in tuning if all(v[k]==r[k] for k in ['arm','trait','family'])]
                best=min(options,key=lambda v:v['validation_mse'])
                assert r['params']==best['params']
                actual=regression_metrics(y[test,LABELS.index(r['trait'])],a[f'{r["arm"]}_{r["trait"]}_{r["family"]}'])
                np.testing.assert_allclose(actual['r2'],r['r2'],atol=1e-12)
            for label in LABELS:
                with np.load(dest/'augmented'/label/'group_shap.npz') as game:
                    np.testing.assert_array_equal(game['test_ids'],ids[test])
                    assert set(game['background_ids'])<=set(ids[train])
                    np.testing.assert_allclose(game['phi'].sum(axis=1),a[f'augmented_{label}_rf']-game['base'],atol=1e-5,rtol=1e-5)
    write_json(out/'VERIFIED.json',dict(checks_passed=True,folds=5,script_sha256=digest(Path(__file__))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,required=True)
    parser.add_argument('--background-run',type=Path,default=ROOT/'outputs/study43_10kb_20261018')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--threads',type=int,default=8)
    args=parser.parse_args()
    out=args.outdir.resolve();parent=args.background_run.resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:parser.error('Use a new directory or --resume')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.lock').open('a') as lock,threadpool_limits(limits=args.threads):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config=dict(seed=42,trees=200,threads=args.threads,background=4,permutations=16,
            population_adjustment=False,candidate_mac=1,background_snps=10000,regions=REGIONS,
            rf_max_features=['sqrt',0.1],rf_min_samples_leaf=[1,5],ridge_alphas=[1.,10.,100.,1000.,10000.])
        identity=dict(config=config,source=digest(Path(__file__)),
            shared_source=digest(Path(__file__).with_name('run_study43_10kb.py')),
            common_source=digest(Path(__file__).with_name('fivefold_common.py')),
            genotype_sha256=digest(GENO),phenotype_sha256=digest(PHENO),
            parent_panels_sha256=digest(parent/'panels.npz'),parent_manifest_sha256=digest(parent/'manifest.json'),
            versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','shap','h5py','pandas']})
        signature=fingerprint(identity);manifest=out/'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text())['fingerprint']!=signature:
            raise ValueError('Input/code/config changed; use fresh output directory')
        write_json(manifest,dict(**identity,fingerprint=signature))
        ids,rows,y,_=load_cohort()
        parent_cohort=json.loads((parent/'cohort_and_splits.json').read_text())
        np.testing.assert_array_equal(ids,parent_cohort['ids']);np.testing.assert_array_equal(y,parent_cohort['y'])
        splits=parent_cohort['splits'];validate_splits(splits,len(ids))
        expected=json.loads((parent/'PREPARED.json').read_text())['panels_sha256']
        assert expected==identity['parent_panels_sha256']
        with np.load(parent/'panels.npz') as a:panels={k:a[k] for k in a.files}
        candidates,sentinel=load_candidates(GENO,rows)
        write_npz(out/'candidate_source.npz',**candidates)
        audit=[]
        for marker in sentinel:
            ix=np.flatnonzero(candidates['indices']==marker['global_snp_index'])[0]
            for split in splits:
                for role in ['fit','train','test']:
                    g=candidates['x'][np.array(split[role]),ix]
                    audit.append(dict(**marker,fold=split['fold'],role=role,minor_allele_count=int(min(g.sum(),len(g)-g.sum()))))
        pd.DataFrame(audit).to_csv(out/'published_snp_audit.csv',index=False)
        write_json(out/'cohort_and_splits.json',parent_cohort)
        write_json(out/'STATUS.json',dict(completed_folds=[s['fold'] for s in splits if (out/f'fold_{s["fold"]}'/'COMPLETE.json').exists()],complete=False))
        print(f'Published SNPs verified; {candidates["x"].shape[1]:,} source SNPs in three candidate windows. Preparation saved.',flush=True)
        if args.prepare_only:return
        for split in splits:
            run_fold(out,split,panels,candidates,ids,y,config)
            summarize(out,splits)
        verify(out,splits,panels,candidates,ids,y)
        print(f'All five folds complete and verified. Report: {out/"REPORT.md"}',flush=True)


if __name__=='__main__':main()
