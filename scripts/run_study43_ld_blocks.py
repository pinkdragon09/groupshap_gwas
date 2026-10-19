#!/usr/bin/env python3
"""Regroup every fitted candidate-run SNP by training-only local LD and re-explain RF.

Preserves RF inputs, models, predictions, and empirical training backgrounds.
Polls for completed source folds when --wait-for-pid is supplied.
"""
import argparse
import fcntl
import importlib.metadata
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from fivefold_common import digest,fingerprint,write_json,write_npz
from run_study43_10kb import LABELS
from run_study43_candidates import panel
from study43_ld_groups import compile_native,exact_rf_group_shap,ld_groups

ROOT=Path(__file__).resolve().parents[1]


def run_fold(out,parent,background,split,cohort,candidates,published,library,threads):
    fold=split['fold'];source=parent/f'fold_{fold}';dest=out/f'fold_{fold}';dest.mkdir(exist_ok=True)
    receipt_path=source/'COMPLETE.json'
    source_receipt=json.loads(receipt_path.read_text())
    complete=dest/'COMPLETE.json'
    if complete.exists():
        receipt=json.loads(complete.read_text())
        assert receipt['source_receipt_sha256']==digest(receipt_path)
        for p,sha in receipt['artifacts'].items():assert digest(dest/p)==sha,p
        print(f'LD fold {fold}: verified and skipped',flush=True);return
    for p,sha in source_receipt['artifacts'].items():assert digest(source/p)==sha,p
    ids=np.array(cohort['ids']);train=np.array(split['train']);test=np.array(split['test'])
    with np.load(background/'panels.npz') as a:
        name=f'outer{fold}';panels={k:a[k] for k in [name,name+'_indices',name+'_positions',name+'_blocks']}
    x,meta=panel(panels,candidates,name,train,True)
    with np.load(source/'augmented_panel.npz') as a:
        for key in meta:np.testing.assert_array_equal(meta[key],a[key])
    chromosomes=np.array([b.split(':')[0] for b in meta['blocks']])
    anchors=[int(np.flatnonzero(meta['indices']==i)[0]) for i in published.global_snp_index if i in meta['indices']]
    groups,mapping=ld_groups(x[train],chromosomes,meta['positions'],meta['indices'],anchors=anchors)
    write_json(dest/'ld_groups.json',groups)
    write_npz(dest/'ld_mapping.npz',mapping=mapping,**meta)
    labels=np.array([g['label'] for g in groups])
    members=[]
    for j,snp in enumerate(meta['indices']):
        members.append(dict(snp_index=snp,chromosome=chromosomes[j],position=meta['positions'][j],
            group=labels[mapping[j]],group_index=mapping[j]))
    pd.DataFrame(members).to_csv(dest/'snp_membership.csv',index=False)
    # A second independent invariant check validates all intra-group pair correlations.
    z=x[train].astype(float);z-=z.mean(axis=0);z/=np.sqrt((z*z).sum(axis=0))
    for g in groups:
        assert g['end_bp']-g['start_bp']<=10000
        r2=np.square(z[:,g['columns']].T@z[:,g['columns']])
        assert r2.min()>=.8-1e-10
    edges=[];anchor_records=[]
    for marker in published.itertuples(index=False):
        hits=np.flatnonzero(meta['indices']==marker.global_snp_index)
        if not len(hits):
            anchor_records.append(dict(gene=marker.gene,position=marker.position,present_in_training_panel=False));continue
        j=int(hits[0]);g=groups[mapping[j]]
        near=np.flatnonzero((chromosomes==chromosomes[j])&(np.abs(meta['positions']-marker.position)<=10000))
        rr=np.square(z[:,near].T@z[:,j])
        qualifying=near[(rr>=.8-1e-12)&(near!=j)]
        anchor_records.append(dict(gene=marker.gene,position=marker.position,present_in_training_panel=True,
            group=g['label'],group_index=g['group_index'],n_snps=g['n_snps'],min_pair_r2=g['min_pair_r2'],
            qualifying_neighbors=len(qualifying),neighbors_joined=sum(mapping[k]==mapping[j] for k in qualifying)))
        for k,r in zip(near,rr):
            if k!=j and r>=.8-1e-12:
                edges.append(dict(gene=marker.gene,anchor_position=marker.position,neighbor_snp_index=meta['indices'][k],
                    neighbor_position=meta['positions'][k],distance_bp=abs(int(meta['positions'][k])-marker.position),
                    r2=r,same_group=bool(mapping[k]==mapping[j])))
    write_json(dest/'candidate_groups.json',anchor_records)
    write_json(dest/'candidate_strong_ld_edges.json',edges)
    print(f'LD fold {fold}: {len(meta["indices"]):,} SNPs -> {len(groups):,} groups; '
          f'{sum(g["n_snps"]>1 for g in groups):,} multi-SNP groups',flush=True)
    ranks=[];agreement=[]
    for trait in LABELS:
        target=dest/trait;target.mkdir(exist_ok=True)
        model=joblib.load(source/'augmented'/trait/'rf.joblib')
        with np.load(source/'augmented'/trait/'group_shap.npz') as old:
            bg_ids=old['background_ids'];expected=old['prediction']
        lookup={a:i for i,a in enumerate(ids)};bg=np.array([lookup[a] for a in bg_ids])
        assert set(bg)<=set(train)
        game=exact_rf_group_shap(model,x[bg],x[test],mapping,library,threads)
        np.testing.assert_allclose(game['prediction'],expected,atol=1e-10,rtol=1e-10)
        write_npz(target/'group_shap.npz',**game,group_labels=labels,test_ids=ids[test],background_ids=bg_ids)
        gwas=pd.read_csv(source/'augmented'/trait/'gwas_snps.csv')
        membership=pd.DataFrame(members)
        joined=membership.merge(gwas[['snp_index','p_value']],on='snp_index',validate='one_to_one')
        assert len(joined)==x.shape[1]
        bygroup=joined.groupby('group',sort=False).agg(min_p=('p_value','min'),gwas_n_snps=('p_value','size'))
        table=pd.DataFrame([{k:v for k,v in g.items() if k!='columns'} for g in groups]).rename(columns={'label':'group'})
        table['mean_abs_group_shap']=np.abs(game['phi']).mean(axis=0)
        table['mean_signed_group_shap']=game['phi'].mean(axis=0)
        table=table.merge(bygroup,on='group',validate='one_to_one')
        assert (table.n_snps==table.gwas_n_snps).all()
        table['within_group_bonferroni_p']=np.minimum(1.,table.min_p*table.n_snps)
        table['across_groups_bonferroni_p']=np.minimum(1.,table.within_group_bonferroni_p*len(groups))
        table['groupshap_rank']=table.mean_abs_group_shap.rank(ascending=False,method='min')
        table['gwas_raw_rank']=table.min_p.rank(method='min')
        table['gwas_corrected_rank']=table.within_group_bonferroni_p.rank(method='min')
        table.to_csv(target/'group_ranks.csv',index=False)
        for record in anchor_records:
            if not record.get('present_in_training_panel'):continue
            r=table[table.group==record['group']].iloc[0].to_dict()
            ranks.append(dict(**r,gene=record['gene'],published_position=record['position'],fold=fold,trait=trait))
        top_shap=set(table.nlargest(10,'mean_abs_group_shap').group)
        top_gwas=set(table.nsmallest(10,'min_p').group)
        agreement.append(dict(fold=fold,trait=trait,n_groups=len(groups),
            spearman=spearmanr(table.mean_abs_group_shap,-np.log10(np.maximum(table.min_p,1e-300))).statistic,
            top10_overlap=len(top_shap&top_gwas),efficiency_error=float(game['efficiency_error'])))
        print(f'LD fold {fold} {trait}: exact finite-background GroupSHAP complete',flush=True)
    write_json(dest/'candidate_ranks.json',ranks);write_json(dest/'agreement.json',agreement)
    artifacts={str(p.relative_to(dest)):digest(p) for p in sorted(dest.rglob('*')) if p.is_file() and p!=complete}
    write_json(complete,dict(source_receipt_sha256=digest(receipt_path),artifacts=artifacts,checks_passed=True))


def summarize(out):
    done=[i for i in range(1,6) if (out/f'fold_{i}'/'COMPLETE.json').exists()]
    if not done:return
    rank=[r for i in done for r in json.loads((out/f'fold_{i}'/'candidate_ranks.json').read_text())]
    agreement=[r for i in done for r in json.loads((out/f'fold_{i}'/'agreement.json').read_text())]
    pd.DataFrame(rank).to_csv(out/'candidate_ranks.csv',index=False)
    pd.DataFrame(agreement).to_csv(out/'gwas_agreement.csv',index=False)
    lines=['# Study 43: training-only local LD groups','',f'Completed {len(done)}/5 folds.',
        '', 'Every input SNP belongs to one group. Groups have span <=10 kb and every pair has training r2 >=0.8.',
        'Published SNPs seed groups first; remaining groups use deterministic genomic seed order. Weakly linked SNPs remain separate.',
        'A seed neighbor joins only if it is also in strong LD with every existing group member and keeps span <=10 kb.',
        'Uses the same candidate-inclusive RF models, input SNPs, test accessions, and training backgrounds as the source run.',
        'GroupSHAP is exact for this RF and the four empirical backgrounds, using tree-leaf conjunctions; no coalition sampling error.',
        'Four backgrounds remain a finite reference sample; exact computation does not establish background convergence.',
        'GWAS uses the same input SNPs and fitting rows, aggregated into these LD groups with Bonferroni corrections.',
        'Grouping covers the SNPs already in the RF inputs; it does not add unselected neighboring markers.',
        'Candidate regions come from this same phenotype study, so this is not independent replication.',
        'See candidate_ranks.csv, gwas_agreement.csv, and per-fold membership/group/LD-edge files.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write_json(out/'STATUS.json',dict(completed_folds=done,complete=len(done)==5))
    if len(done)==5:write_json(out/'VERIFIED.json',dict(checks_passed=True,completed_folds=done,
        checks=['source artifact hashes','panel identity','training-only all-pair LD and span',
                'disjoint exhaustive mapping','source prediction identity','Shapley efficiency','matched GWAS membership']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,required=True)
    parser.add_argument('--source-run',type=Path,default=ROOT/'outputs/study43_candidates_20261018')
    parser.add_argument('--background-run',type=Path,default=ROOT/'outputs/study43_10kb_20261018')
    parser.add_argument('--wait-for-pid',type=int)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();out=args.outdir.resolve();parent=args.source_run.resolve();background=args.background_run.resolve()
    if args.threads<1:parser.error('Positive thread count required')
    if out.exists() and any(out.iterdir()) and not args.resume:parser.error('Use new output directory or --resume')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.lock').open('a') as lock,threadpool_limits(limits=args.threads):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        sources=[Path(__file__),Path(__file__).with_name('study43_ld_groups.py'),Path(__file__).with_name('study43_ld_shap.c'),
            Path(__file__).with_name('run_study43_candidates.py'),Path(__file__).with_name('fivefold_common.py')]
        identity=dict(threshold=.8,max_span_bp=10000,threads=args.threads,
            rule='anchor-first deterministic greedy complete linkage, noncontiguous membership allowed',
            source={str(p):digest(p) for p in sources},source_run_manifest=digest(parent/'manifest.json'),
            candidate_source=digest(parent/'candidate_source.npz'),background_panels=digest(background/'panels.npz'),
            versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','h5py','pandas']})
        signature=fingerprint(identity);manifest=out/'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text())['fingerprint']!=signature:raise ValueError('Run identity changed')
        write_json(manifest,dict(**identity,fingerprint=signature))
        library=compile_native(out/'native')
        cohort=json.loads((parent/'cohort_and_splits.json').read_text())
        with np.load(parent/'candidate_source.npz') as a:candidates={k:a[k] for k in a.files}
        published=pd.read_csv(parent/'published_snp_audit.csv').drop_duplicates('global_snp_index')
        write_json(out/'STATUS.json',dict(completed_folds=[],complete=False))
        for split in cohort['splits']:
            ready=parent/f'fold_{split["fold"]}'/'COMPLETE.json'
            announced=False
            while not ready.exists():
                if not args.wait_for_pid:raise RuntimeError(f'Source fold not yet complete: {split["fold"]}')
                try:os.kill(args.wait_for_pid,0)
                except ProcessLookupError:raise RuntimeError('Source process stopped before producing the required fold') from None
                if not announced:print(f'Waiting for source fold {split["fold"]}',flush=True);announced=True
                time.sleep(15)
            run_fold(out,parent,background,split,cohort,candidates,published,library,args.threads)
            summarize(out)
        print(f'All five LD-group folds complete and verified: {out/"REPORT.md"}',flush=True)


if __name__=='__main__':main()
