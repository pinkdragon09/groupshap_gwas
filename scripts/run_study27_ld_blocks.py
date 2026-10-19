#!/usr/bin/env python3
"""Study 27: dense phenotype-blind inputs, training-only LD, RF/KAN/GWAS.

Five matched folds. No population adjustment, consistent with revised Study 43.
Fresh output required; --resume validates unchanged sources/configuration.
"""
import argparse
import fcntl
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from fivefold_common import (digest,fingerprint,fit_standardizer,make_splits,
    regression_metrics,validate_splits,write_json,write_npz)
from run_study27_fivefold import GENO,PHENO,load_cohort
from run_study43_10kb import prepare_panels,gwas_table
from study43_ld_groups import ld_groups,compile_native,exact_rf_group_shap
from yeast_fivefold_common import new_kan
from yeast_ld_groups import BinaryKAN,install_binary_forward
from study27_loci import ANNOTATION,PAPER,reference_loci,candidate_source,augment


def kan_group_shap(fast,background,evaluation,paths=32,seed=42):
    if paths<4 or paths%4:raise ValueError('Use a multiple of four paths')
    base,change=fast.group_deltas(background,evaluation)
    ne,nb,ng,hidden=change.shape
    pairs=np.zeros((paths//2,ne,ng));rng=np.random.default_rng(seed)
    empty=fast.outer(base)
    for pair in range(paths//2):
        order=rng.permutation(ng)
        for path in [order,order[::-1]]:
            states=np.cumsum(change[:,:,path],axis=2)+base[None,:,None]
            values=fast.outer(states)
            before=np.concatenate([np.broadcast_to(empty[None,:,None],(ne,nb,1)),values[:,:,:-1]],axis=2)
            pairs[pair][:,path]+=(values-before).mean(axis=1)/2
    phi=pairs.mean(axis=0);baseline=float(empty.mean());prediction=fast.predict(evaluation)
    if not np.allclose(phi.sum(axis=1),prediction-baseline,atol=1e-8,rtol=1e-8):
        raise ValueError('KAN grouped efficiency failed')
    half=len(pairs)//2
    return dict(phi=phi,mc_se=pairs.std(axis=0,ddof=1)/np.sqrt(len(pairs)),
        first_half=pairs[:half].mean(axis=0),second_half=pairs[half:].mean(axis=0),
        base=np.array(baseline),prediction=prediction)


def train_kan(x,y,seed,epochs,validation=None,patience=25):
    estimator=install_binary_forward(new_kan(x.shape[1],seed))
    optimizer=torch.optim.Adam(estimator.parameters(),lr=.001)
    xx=torch.tensor(x,dtype=torch.float32);yy=torch.tensor(y[:,None],dtype=torch.float32)
    if validation is not None:
        vx=torch.tensor(validation[0],dtype=torch.float32);vy=torch.tensor(validation[1][:,None],dtype=torch.float32)
    rng=np.random.default_rng(seed);best=float('inf');best_epoch=1;stale=0;trace=[]
    for epoch in range(1,epochs+1):
        estimator.train();order=rng.permutation(len(x));total=0.
        for start in range(0,len(x),64):
            ix=order[start:start+64];optimizer.zero_grad()
            loss=(estimator(xx[ix])-yy[ix]).square().mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite KAN loss')
            loss.backward();optimizer.step();total+=len(ix)*float(loss.detach())
        record=dict(epoch=epoch,training_mse=total/len(x));estimator.eval()
        if validation is not None:
            with torch.no_grad():score=float((estimator(vx)-vy).square().mean())
            record['validation_mse']=score
            if score<best-1e-6:best=score;best_epoch=epoch;stale=0
            else:stale+=1
        trace.append(record)
        if validation is not None and stale>=patience:break
    return estimator.eval(),best_epoch if validation is not None else epochs,trace


def prepare(out,signature):
    cohort=load_cohort(GENO,PHENO)
    ids=np.array(cohort['sample_ids']);y=np.array(cohort['y']);rows=np.array(cohort['genotype_rows'])
    if len(ids)!=81:raise ValueError('Expected 81 matched accessions')
    splits=make_splits(ids);validate_splits(splits,len(ids))
    loci=reference_loci()
    write_json(out/'reference_loci.json',loci)
    cache=out/'panels.npz';receipt=out/'PREPARED.json'
    if receipt.exists():
        record=json.loads(receipt.read_text());assert record['fingerprint']==signature
        assert digest(cache)==record['panels_sha256']
        with np.load(cache) as a:panels={k:a[k] for k in a.files}
    else:
        jobs={f'{stage}{s["fold"]}':s[role] for s in splits for stage,role in [('inner','fit'),('outer','train')]}
        panels=prepare_panels(rows,jobs,42)
        candidates=candidate_source(GENO,rows,loci)
        write_npz(out/'candidate_source.npz',**candidates)
        for name,train in jobs.items():
            background=dict(x=panels[name],**{key:panels[name+'_'+key] for key in ['indices','positions','blocks']})
            augmented=augment(background,candidates,np.array(train))
            panels[name]=augmented['x']
            for key in ['indices','positions','blocks']:panels[name+'_'+key]=augmented[key]
        write_npz(cache,**panels)
        write_json(receipt,dict(fingerprint=signature,panels_sha256=digest(cache)))
    write_json(out/'cohort_and_splits.json',dict(**cohort,splits=splits))
    return ids,y,splits,panels


def fold_partition(panels,split):
    name=f'outer{split["fold"]}';x=panels[name]
    chrom=np.array([b.split(':')[0] for b in panels[name+'_blocks']])
    loci=reference_loci();anchors=[]
    for locus in loci:
        if locus['published_snp_position'] is None:continue
        found=np.flatnonzero((chrom==locus['chromosome'])&(panels[name+'_positions']==locus['published_snp_position']))
        anchors.extend(found.tolist())
    groups,mapping=ld_groups(x[np.array(split['train'])],chrom,panels[name+'_positions'],
        panels[name+'_indices'],anchors=anchors,threshold=.8+1e-10,max_span=10000)
    return x,groups,mapping


def run_fold(out,ids,y,split,panels,config,library):
    fold=split['fold'];dest=out/f'fold_{fold}';dest.mkdir(exist_ok=True)
    complete=dest/'COMPLETE.json'
    if complete.exists():
        receipt=json.loads(complete.read_text())
        for p,sha in receipt['artifacts'].items():assert digest(dest/p)==sha,p
        print(f'Root fold {fold}: verified/skipped',flush=True);return
    train,test,fit,val=[np.array(split[k]) for k in ['train','test','fit','validation']]
    x,groups,mapping=fold_partition(panels,split);xi=panels[f'inner{fold}'];name=f'outer{fold}'
    labels=np.array([g['label'] for g in groups]);snp_labels=labels[mapping]
    write_json(dest/'ld_groups.json',groups)
    write_npz(dest/'partition.npz',mapping=mapping,snp_indices=panels[name+'_indices'],positions=panels[name+'_positions'])
    print(f'Root fold {fold}: {len(train)} train/{len(test)} test, {x.shape[1]} SNPs, {len(groups)} LD groups',flush=True)
    bg=np.random.default_rng(2042+fold).choice(train,8,replace=False)
    outer_mean,outer_sd=fit_standardizer(y[train]);inner_mean,inner_sd=fit_standardizer(y[fit])
    tuning=[];metrics=[];predictions=dict(test_ids=ids[test],observed=y[test]);seed=42+100*fold
    models={}
    for family,grid in [('rf',[(m,l) for m in ['sqrt',.1] for l in [1,5]]),('ridge',[1.,10.,100.,1000.,10000.])]:
        options=[]
        for params in grid:
            model=(RandomForestRegressor(n_estimators=200,max_features=params[0],min_samples_leaf=params[1],random_state=seed,n_jobs=config['threads'])
                if family=='rf' else make_pipeline(StandardScaler(),Ridge(alpha=params,solver='cholesky')))
            model.fit(xi[fit],y[fit]);loss=float(np.mean((model.predict(xi[val])-y[val])**2))
            tuning.append(dict(family=family,params=params,validation_mse=loss));options.append((loss,params))
        loss,params=min(options,key=lambda a:a[0])
        model=(RandomForestRegressor(n_estimators=200,max_features=params[0],min_samples_leaf=params[1],random_state=seed,n_jobs=config['threads'])
            if family=='rf' else make_pipeline(StandardScaler(),Ridge(alpha=params,solver='cholesky')))
        model.fit(x[train],y[train]);prediction=model.predict(x[test]);predictions[family]=prediction
        joblib.dump(model,dest/f'{family}.joblib',compress=3);models[family]=model
        metrics.append(dict(fold=fold,family=family,params=params,**regression_metrics(y[test],prediction)))
        print(f'Root fold {fold} {family}: R2={metrics[-1]["r2"]:.4f}; params={params}',flush=True)
    _,epoch,inner_trace=train_kan(xi[fit],(y[fit]-inner_mean)/inner_sd,seed,config['epochs'],
        (xi[val],(y[val]-inner_mean)/inner_sd))
    kan,_,outer_trace=train_kan(x[train],(y[train]-outer_mean)/outer_sd,seed,epoch)
    torch.save(kan.state_dict(),dest/'kan.pt')
    fast=BinaryKAN(kan,mapping)
    with torch.no_grad():direct=kan(torch.tensor(x,dtype=torch.float32)).numpy().ravel()
    accelerated=fast.predict(x)
    if not np.allclose(direct,accelerated,atol=2e-5,rtol=2e-4):raise ValueError('KAN endpoint equivalence failed')
    prediction=accelerated[test]*outer_sd+outer_mean;predictions['kan']=prediction
    metrics.append(dict(fold=fold,family='kan',params=dict(epochs=epoch),**regression_metrics(y[test],prediction)))
    write_json(dest/'kan_fit.json',dict(seed=seed,target_mean=outer_mean,target_sd=outer_sd,epochs=epoch,
        direct_fast_max_error=float(np.max(np.abs(direct-accelerated))),inner_trace=inner_trace,outer_trace=outer_trace))
    predictions['mean']=np.full(len(test),y[train].mean())
    metrics.append(dict(fold=fold,family='mean',params=None,**regression_metrics(y[test],predictions['mean'])))
    print(f'Root fold {fold} kan: R2={metrics[-2]["r2"]:.4f}; epochs={epoch}',flush=True)
    snps,gwas=gwas_table(x[train],y[train],panels[name+'_indices'],panels[name+'_positions'],snp_labels)
    snps.to_csv(dest/'gwas_snps.csv',index=False)
    gwas=gwas.rename(columns={'block':'group'});gwas.to_csv(dest/'gwas_groups.csv',index=False)
    games={}
    rf_game=exact_rf_group_shap(models['rf'],x[bg],x[test],mapping,library,config['threads'])
    games['rf']=rf_game
    kan_game=kan_group_shap(fast,x[bg],x[test],config['paths'],4042+fold)
    for key in ['phi','mc_se','first_half','second_half']:kan_game[key]*=outer_sd
    kan_game['base']=kan_game['base']*outer_sd+outer_mean
    kan_game['prediction']=kan_game['prediction']*outer_sd+outer_mean
    games['kan']=kan_game
    summary=[];reference_results=[]
    for family,game in games.items():
        np.testing.assert_allclose(game['prediction'],predictions[family],atol=1e-6,rtol=1e-6)
        np.testing.assert_allclose(game['phi'].sum(axis=1),predictions[family]-game['base'],atol=1e-6)
        write_npz(dest/f'{family}_group_shap.npz',**game,group_labels=labels,test_ids=ids[test],background_ids=ids[bg])
        table=pd.DataFrame([{k:v for k,v in g.items() if k!='columns'} for g in groups]).rename(columns={'label':'group'})
        table['mean_abs_group_shap']=np.abs(game['phi']).mean(axis=0)
        table['mean_signed_group_shap']=game['phi'].mean(axis=0)
        table=table.merge(gwas,on='group',validate='one_to_one',suffixes=('','_gwas'))
        assert (table.n_snps==table.n_snps_gwas).all()
        table['groupshap_rank']=table.mean_abs_group_shap.rank(ascending=False,method='min')
        table['gwas_raw_rank']=table.min_p.rank(method='min')
        table.to_csv(dest/f'{family}_group_ranks.csv',index=False)
        chromosomes=np.array([str(b).split(':')[0] for b in panels[name+'_blocks']])
        for locus in reference_loci():
            in_gene=np.flatnonzero((chromosomes==locus['chromosome'])&(panels[name+'_positions']>=locus['start'])&
                (panels[name+'_positions']<=locus['end']))
            candidates_ix=in_gene;distance=0;mapping_policy='exact SNP' if locus['published_snp_position'] else 'gene-body SNP groups'
            if not len(candidates_ix) and locus['published_snp_position'] is not None:
                same=np.flatnonzero(chromosomes==locus['chromosome'])
                distances=np.abs(panels[name+'_positions'][same]-locus['start'])
                nearest=int(np.argmin(distances))
                if distances[nearest]<=10000:
                    candidates_ix=np.array([same[nearest]]);distance=int(distances[nearest]);mapping_policy='nearest retained SNP within 10kb'
            gid=np.unique(mapping[candidates_ix])
            eligible=table[table.group_index.isin(gid)]
            reference_results.append(dict(fold=fold,family=family,reference=locus['reference'],category=locus['category'],
                reported_trait=locus['reported_trait'],mapping_policy=mapping_policy,distance_bp=distance,
                mapped_snps=len(candidates_ix),mapped_groups=len(gid),covered=bool(len(eligible)),
                best_groupshap_rank=float(eligible.groupshap_rank.min()) if len(eligible) else None,
                best_gwas_raw_rank=float(eligible.gwas_raw_rank.min()) if len(eligible) else None,
                best_gwas_fwer_p=float(eligible.across_blocks_bonferroni_p.min()) if len(eligible) else None))
        top=set(table.nlargest(10,'mean_abs_group_shap').group);gwas_top=set(table.nsmallest(10,'min_p').group)
        row=dict(fold=fold,family=family,n_groups=len(groups),
            gwas_spearman=spearmanr(table.mean_abs_group_shap,-np.log10(np.maximum(table.min_p,1e-300))).statistic,
            gwas_shared_top10=len(top&gwas_top),efficiency_error=float(np.max(np.abs(game['phi'].sum(axis=1)-(game['prediction']-game['base'])))),
            leading_group=table.nlargest(1,'mean_abs_group_shap').group.iloc[0])
        if family=='kan':
            first=np.abs(game['first_half']).mean(axis=0);second=np.abs(game['second_half']).mean(axis=0)
            row['mc_split_half_spearman']=spearmanr(first,second).statistic
            row['mc_split_half_top10_overlap']=len(set(np.argsort(-first)[:10])&set(np.argsort(-second)[:10]))/10
        summary.append(row)
    rf=np.abs(games['rf']['phi']).mean(axis=0);ks=np.abs(games['kan']['phi']).mean(axis=0)
    write_json(dest/'model_attribution_agreement.json',dict(spearman=spearmanr(rf,ks).statistic,
        shared_top10=len(set(np.argsort(-rf)[:10])&set(np.argsort(-ks)[:10]))))
    write_json(dest/'metrics.json',metrics);write_json(dest/'tuning.json',tuning);write_json(dest/'attribution_summary.json',summary)
    write_json(dest/'reference_recovery.json',reference_results)
    write_npz(dest/'predictions.npz',**predictions)
    receipt={str(p.relative_to(dest)):digest(p) for p in sorted(dest.rglob('*')) if p.is_file() and p!=complete}
    write_json(complete,dict(artifacts=receipt))


def summarize(out,ids,y,splits):
    done=[s for s in splits if (out/f'fold_{s["fold"]}'/'COMPLETE.json').exists()]
    if not done:return
    records=[r for s in done for r in json.loads((out/f'fold_{s["fold"]}'/'metrics.json').read_text())]
    frame=pd.DataFrame(records);frame.to_csv(out/'metrics_by_fold.csv',index=False)
    attributions=[r for s in done for r in json.loads((out/f'fold_{s["fold"]}'/'attribution_summary.json').read_text())]
    pd.DataFrame(attributions).to_csv(out/'attribution_summary.csv',index=False)
    recovery=[r for s in done for r in json.loads((out/f'fold_{s["fold"]}'/'reference_recovery.json').read_text())]
    pd.DataFrame(recovery).to_csv(out/'reference_recovery.csv',index=False)
    summary=[]
    for family in ['rf','kan','ridge','mean']:
        truth=[];pred=[]
        for s in done:
            with np.load(out/f'fold_{s["fold"]}'/'predictions.npz') as a:truth.extend(a['observed']);pred.extend(a[family])
        rows=frame[frame.family==family]
        summary.append(dict(family=family,mean_fold_r2=rows.r2.mean(),sd_fold_r2=rows.r2.std(ddof=1),
            **regression_metrics(np.array(truth),np.array(pred))))
    pd.DataFrame(summary).to_csv(out/'model_summary.csv',index=False)
    lines=['# Study 27: dense LD-group analysis','',f'Completed {len(done)}/5 folds; 81 root-length accessions.',
        '', '10,000 phenotype-blind background SNPs plus reference regions; no population adjustment. Training-only all-pair rho-squared >0.8, group span <=10kb.',
        'RF/KAN retain individual SNP inputs and share LD groups, held-out rows, and eight training backgrounds.',
        'RF group Shapley is exact for the finite background; KAN uses 32 antithetic paths.',
        'GWAS is training-only on the same panel, with per-group minimum p and Bonferroni corrections.',
        '', '| Model | Pooled OOF R2 | Mean fold R2 | Fold SD |','|---|---:|---:|---:|']
    for r in summary:lines.append(f'| {r["family"]} | {r["r2"]:.4f} | {r["mean_fold_r2"]:.4f} | {r["sd_fold_r2"]:.4f} |')
    lines+=['','Ranking summaries are descriptive. Small folds, selected genomic coverage, finite backgrounds and omitted structure limit inference.',
        'The nine Ristova control loci concern LRR or LRD_R, not control total root length. BRX/SHY2 are functional root-growth references, not confirmed natural-variation causes of this exact trait.',
        'This revision evaluates LD-group attribution and matched marginal association. Historical chromosome interaction tables belong to a different analysis.',
        'These results do not establish biological causality or superiority over GWAS.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write_json(out/'STATUS.json',dict(completed_folds=[s['fold'] for s in done],complete=len(done)==5))


def verify(out,ids,y,splits,panels,library):
    audit=[];covered=np.zeros(len(ids),int)
    for s in splits:
        fold=s['fold'];dest=out/f'fold_{fold}';train=np.array(s['train']);test=np.array(s['test']);covered[test]+=1
        receipt=json.loads((dest/'COMPLETE.json').read_text())
        for p,sha in receipt['artifacts'].items():assert digest(dest/p)==sha,p
        for stage,role in [('inner','fit'),('outer','train')]:
            name=f'{stage}{fold}';g=panels[name][np.array(s[role])];count=g.sum(axis=0)
            assert (np.minimum(count[:10000],len(g)-count[:10000])>=5).all() and g.shape[1]>=10000
            assert ((count[10000:]>0)&(count[10000:]<len(g))).all()
        x,groups,mapping=fold_partition(panels,s)
        saved=json.loads((dest/'ld_groups.json').read_text());assert fingerprint(groups)==fingerprint(saved)
        assert sorted(j for g in groups for j in g['columns'])==list(range(x.shape[1]))
        z=x[train].astype(float);z-=z.mean(axis=0);z/=np.sqrt((z*z).sum(axis=0));minimum=1.;pairs=0
        for g in groups:
            cols=g['columns'];assert g['end_bp']-g['start_bp']<=10000
            if len(cols)>1:
                rr=np.square(z[:,cols].T@z[:,cols]);rr=rr[np.triu_indices(len(cols),1)]
                assert (rr>.8).all();minimum=min(minimum,float(rr.min()));pairs+=len(rr)
        with np.load(dest/'predictions.npz') as a:
            np.testing.assert_array_equal(a['test_ids'],ids[test]);np.testing.assert_array_equal(a['observed'],y[test])
            for family in ['rf','ridge']:
                m=joblib.load(dest/f'{family}.joblib');np.testing.assert_allclose(m.predict(x[test]),a[family],atol=1e-9)
            cfg=json.loads((dest/'kan_fit.json').read_text());k=new_kan(x.shape[1],cfg['seed'])
            k.load_state_dict(torch.load(dest/'kan.pt',weights_only=True,map_location='cpu'));k=install_binary_forward(k)
            with torch.no_grad():direct=k(torch.tensor(x[test],dtype=torch.float32)).numpy().ravel()*cfg['target_sd']+cfg['target_mean']
            np.testing.assert_allclose(direct,a['kan'],atol=2e-4,rtol=2e-4)
            for family in ['rf','kan']:
                with np.load(dest/f'{family}_group_shap.npz') as game:
                    assert set(game['background_ids'])<=set(ids[train])
                    np.testing.assert_array_equal(game['test_ids'],ids[test])
                    np.testing.assert_allclose(game['phi'].sum(axis=1),a[family]-game['base'],atol=1e-6)
            snps,_=gwas_table(x[train],y[train],panels[f'outer{fold}_indices'],panels[f'outer{fold}_positions'],np.array([g['label'] for g in groups])[mapping])
            gwas=pd.read_csv(dest/'gwas_snps.csv');np.testing.assert_allclose(snps.p_value,gwas.p_value,atol=1e-14,rtol=1e-9)
        audit.append(dict(fold=fold,n_snps=x.shape[1],n_groups=len(groups),within_group_pairs=pairs,min_pair_r2=minimum))
        print(f'Root fold {fold}: checkpoint, held-out prediction, GWAS and strict LD audit passed',flush=True)
    assert (covered==1).all()
    pd.DataFrame(audit).to_csv(out/'ld_audit.csv',index=False)
    write_json(out/'VERIFIED.json',dict(checks_passed=True,folds=5,n=81,
        checks=['input/source hashes','fold receipts','inner/outer MAC','no phenotype SNP screen','LD partition recomputation',
                'all pairs rho squared >0.8','span <=10kb','saved RF/ridge/KAN predictions','training background membership',
                'GWAS recomputation','Shapley efficiency','exactly one OOF prediction per accession']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args();out=args.outdir.resolve()
    if args.threads<1:parser.error('Positive thread count required')
    if out.exists() and any(out.iterdir()) and not args.resume:parser.error('Fresh output directory or --resume required')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.lock').open('a') as lock,threadpool_limits(limits=args.threads):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(args.threads)
        config=dict(seed=42,n_snps=10000,blocks_per_chr=20,snps_per_window=100,mac=5,ld_r2=.8,ld_numeric_margin=1e-10,
            max_span=10000,population_adjustment=False,reference_loci=reference_loci(),candidate_mac=1,rf_trees=200,rf_features=['sqrt',.1],rf_leaf=[1,5],
            kan_hidden=8,kan_grid=5,epochs=200,patience=25,learning_rate=.001,batch=64,paths=32,background=8,threads=args.threads)
        files=[Path(__file__)]+[Path(__file__).with_name(p) for p in ['run_study27_fivefold.py','run_study43_10kb.py',
            'study43_ld_groups.py','study43_ld_shap.c','yeast_fivefold_common.py','yeast_ld_groups.py','fivefold_common.py','study27_loci.py']]
        identity=dict(config=config,inputs={str(p):digest(p) for p in [GENO,PHENO,ANNOTATION,PAPER]},source={str(p):digest(p) for p in files},
            versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','scikit-learn','torch','pykan','h5py']})
        signature=fingerprint(identity);manifest=out/'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text())['fingerprint']!=signature:raise ValueError('Run identity changed')
        write_json(manifest,dict(**identity,fingerprint=signature))
        ids,y,splits,panels=prepare(out,signature)
        if args.prepare_only:return
        library=compile_native(out/'native')
        for s in splits:run_fold(out,ids,y,s,panels,config,library);summarize(out,ids,y,splits)
        verify(out,ids,y,splits,panels,library)
        for p,sha in identity['source'].items():assert digest(Path(p))==sha
        print(f'All five root-length LD folds complete and verified: {out/"REPORT.md"}',flush=True)


if __name__=='__main__':main()
