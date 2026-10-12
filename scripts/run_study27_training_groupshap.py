#!/usr/bin/env python3
"""Matched in-sample/held-out chromosome GroupSHAP for fixed-panel Study 27.

Historical fits have no retained checkpoints. Reconstruct RF/KAN with their
unchanged settings, require saved-prediction parity, and retain checkpoints here.
Never overwrite the historical result directories. No phenotype reselection.
"""
import argparse
import importlib.metadata
import inspect
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

import run_phase5 as loader
from study27_kan_historical import fit_kan
from fivefold_common import digest, exact_group_game, regression_metrics, write_json, write_npz

ROOT = Path(__file__).resolve().parents[1]


def compare(a, b, k=3):
    """Compare matched group or pair scores; deterministic stable tie ordering."""
    rho = float(spearmanr(a, b).statistic)
    ta = set(np.argsort(-a, kind='stable')[:k])
    tb = set(np.argsort(-b, kind='stable')[:k])
    return rho, len(ta & tb) / k


def check_reference(path, records, observed, expected, atol):
    if not np.isfinite(observed).all():
        raise ValueError(f'Nonfinite values: {path}')
    np.testing.assert_allclose(observed, expected, rtol=0, atol=atol,
                               err_msg=f'Saved result not reproduced: {path}')
    error = float(np.max(np.abs(np.asarray(observed)-np.asarray(expected))))
    records.append(dict(reference=str(path), max_absolute_error=error, tolerance=atol))


def report(out):
    metrics = pd.read_csv(out/'metrics.tsv', sep='\t')
    ranks = pd.read_csv(out/'rank_comparison.tsv', sep='\t')
    groups = pd.read_csv(out/'group_by_fold.tsv', sep='\t')
    pairs = pd.read_csv(out/'interaction_by_fold.tsv', sep='\t')
    group_summary = groups.groupby(['model','role','group'], sort=True).agg(
        mean_abs=('mean_abs','mean'), fold_sd=('mean_abs','std'),
        mean_signed=('mean_signed','mean'), folds=('fold','nunique')).reset_index()
    pair_summary = pairs.groupby(['model','role','group_a','group_b'], sort=True).agg(
        mean_abs=('mean_abs','mean'), fold_sd=('mean_abs','std'),
        mean_signed=('mean_signed','mean'), folds=('fold','nunique')).reset_index()
    group_summary.to_csv(out/'group_summary.tsv', sep='\t', index=False)
    pair_summary.to_csv(out/'interaction_summary.tsv', sep='\t', index=False)
    text = ['# Study 27: training versus held-out GroupSHAP', '',
        'All 81 accessions, the same fixed 200-SNP panel and three full-cohort kinship PCs. '
        'Chromosomes are five disjoint players; all three PCs form a sixth player. '
        'Exact enumeration of all 64 coalitions uses eight identical fit-only background '
        'donors per fold, shared by RF and KAN and by every evaluation role. '
        'These are actual group Shapley values, not sums of SNP SHAP values. '
        'Interaction values use the full Shapley interaction index, with no one-half factor. '
        'Values are in native Mean(TRL) C model-output units.', '',
        'Historical fixed-panel checkpoints were not retained. RF and KAN fits were '
        'reconstructed using unchanged settings, split assignments and seeds. '
        'RF predictions match the saved matched fixed-panel RF rerun; KAN predictions, '
        'best epochs, training traces and held-out group/interaction summaries match '
        'the historical KAN run within recorded numerical tolerances. '
        'Reconstructed checkpoints and preprocessing are now saved in this new directory. '
        'Original historical files remain unchanged. Transformer reconstruction was '
        'not included; no transformer training-attribution claim is made.', '',
        'RF fits all 64–65 outer-training rows. KAN fits only the 51–52 inner-fitting '
        'rows, with 13 validation rows used to select its early-stopping checkpoint. '
        'Training below means rows used to fit weights/trees. KAN validation, outer-training '
        '(fit plus validation) and held-out rows are saved separately. '
        'Training explanations include the background donors.', '',
        '## Paired comparisons', '',
        '| Model | Fitting R² | Held-out R² | Group rank rho (6 players) | Chromosome-only rho | Pair rank rho (15 pairs) | Chromosome-pair rho (10 pairs) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for model in ['rf','kan']:
        mm = metrics[metrics.model == model]
        rr = ranks[(ranks.model == model) & (ranks.role == 'train')]
        value = lambda level, key: rr.loc[rr.level == level,key].mean()
        text.append(f'| {model.upper()} | {mm.loc[mm.role=="train","r2"].mean():.4f} | '
                    f'{mm.loc[mm.role=="heldout","r2"].mean():.4f} | '
                    f'{value("group", "rho"):.4f} | {value("chromosome", "rho"):.4f} | '
                    f'{value("interaction", "rho"):.4f} | {value("chromosome_interaction", "rho"):.4f} |')
    text += ['', 'Statistics are arithmetic means of within-fold comparisons, not correlations '
        'of pooled summaries. Training observations recur across folds; the five '
        'models do not constitute independent replications. Main effects and '
        'interactions use every row in each role, including all 16–17 held-out rows. '
        'Coalitions are enumerated exactly for the chosen finite background; '
        'background-choice uncertainty has not been measured.', '',
        '## Chromosome and PC importance', '',
        '| Model/group | Fitting mean absolute attribution | Held-out mean absolute attribution |',
        '|---|---:|---:|']
    for model in ['rf','kan']:
        table = group_summary[group_summary.model == model].pivot(index='group', columns='role', values='mean_abs')
        for label, row in table.iterrows():
            text.append(f'| {model.upper()} / {label} | {row["train"]:.4f} | {row["heldout"]:.4f} |')
    text += ['', '## Largest chromosome-pair interactions by fitting magnitude', '',
        '| Model/pair | Fitting mean absolute index | Held-out mean absolute index |',
        '|---|---:|---:|']
    for model in ['rf','kan']:
        table = pair_summary[(pair_summary.model == model) & (pair_summary.group_b != 'PCs')].pivot(
            index=['group_a','group_b'], columns='role', values='mean_abs').sort_values('train', ascending=False).head(3)
        for (a,b), row in table.iterrows():
            text.append(f'| {model.upper()} / {a} × {b} | {row["train"]:.4f} | {row["heldout"]:.4f} |')
    text += ['', 'Pairs above are selected using fitting-set magnitude. Their held-out '
        'columns are descriptive comparisons, not independent tests of that selection.', '',
        '## Interpretation', '',
        'The comparison measures sensitivity of explanations to evaluation samples '
        'and the extent to which fitted-model reliance is similar in fitting and held-out '
        'accessions. It does not alter predictions, improve predictive accuracy, '
        'establish causal genes, or validate biological interactions. The small '
        'held-out folds and differing evaluation sample sizes limit precision. '
        'High chromosome agreement cannot establish agreement at SNP or gene resolution.', '',
        'The fixed SNP panel was selected using all phenotypes, including held-out '
        'phenotypes, before cross-validation. PCs also use the full-cohort kinship '
        'matrix. Prediction and explanations therefore condition on this panel '
        'and representation; training explanations do not remove this selection '
        'dependence or population-structure confounding. Held-out explanations '
        'remain primary, with in-sample descriptions supplementary. No p-values, '
        'confidence intervals or repeated-background stability claims are made.', '',
        '## Files and checks', '',
        '`metrics.tsv`, `rank_comparison.tsv`, per-fold and aggregated group/interaction '
        'TSVs, all local coalition/attribution arrays and predictions, fold membership '
        'and background IDs, reconstructed checkpoints, preprocessing, input matrix '
        'and input/output SHA256 receipts are saved here. `validation.json` records '
        'historical parity, split isolation, exhaustive grouping and local efficiency. '
        'Run the independent saved-checkpoint verifier for artifact hashes and '
        'coalition recomputation. Efficiency validates accounting, not convergence '
        'over possible backgrounds.', '',
        'Reproduce into a fresh directory with `.venv-kan/bin/python '
        'scripts/run_study27_training_groupshap.py --outdir outputs/NEW_DIRECTORY`.']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
    fig, axes = plt.subplots(1,2,figsize=(10,4.5),constrained_layout=True)
    for ax,model in zip(axes,['rf','kan']):
        table = group_summary[group_summary.model==model].pivot(index='group',columns='role',values='mean_abs')
        pos=np.arange(len(table))
        ax.barh(pos+.18,table.train,height=.36,label='Fitting samples')
        ax.barh(pos-.18,table.heldout,height=.36,label='Held-out samples')
        ax.set_yticks(pos,table.index)
        ax.invert_yaxis()
        ax.set_title(model.upper())
        ax.set_xlabel('Mean absolute group Shapley (native units)')
        ax.legend(fontsize=8)
    fig.suptitle('Study 27: fixed 200-SNP panel, five matched folds')
    fig.savefig(out/'training_vs_heldout_groupshap.png',dpi=180)
    fig.savefig(out/'training_vs_heldout_groupshap.pdf')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,default=ROOT/'outputs/study27_training_groupshap_20261010')
    args=parser.parse_args()
    out=args.outdir.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit('Choose a fresh output directory; existing results are protected.')
    out.mkdir(parents=True,exist_ok=True)
    history=ROOT/'outputs/study27_kan_historical'
    rf_history=ROOT/'outputs/study27_fixed_pc_baseline'
    input_paths=[Path(__file__),Path(loader.__file__),ROOT/'scripts/fivefold_common.py',
        ROOT/'scripts/study27_kan_historical.py',ROOT/'scripts/study27_kan_comparison.py',
        ROOT/'scripts/transformer_mamba_gwas_template.py',ROOT/'scripts/snp_selection.py',
        loader.BASELINE_HITS,loader.PHENO_ZIP,loader.KINSHIP_H5]
    input_paths += [history/n for n in ['run_config.json','feature_meta.tsv','splits.tsv',
        'predictions.tsv','cv_metrics.tsv','group_shap.tsv','group_interactions.tsv','training_trace.tsv']]
    input_paths += [rf_history/n for n in ['provenance.json','feature_meta.tsv','splits.json','predictions.tsv']]
    inputs_hash={str(p.relative_to(ROOT)):digest(p) for p in input_paths}
    # The large genotype source is represented by file metadata plus the extracted
    # matrix hash; the saved exact matrix is independently checkpoint-verifiable.
    Xdf,ys,meta=loader.load_arabidopsis_matrix()
    Xrf,y=Xdf.to_numpy(),ys.to_numpy()
    Xkan,ykan=Xrf.astype(np.float32),y.astype(np.float32)
    assert Xrf.shape==(81,203) and np.isfinite(Xrf).all() and np.isfinite(y).all()
    saved_meta=pd.read_csv(history/'feature_meta.tsv',sep='\t')
    np.testing.assert_array_equal(meta.feature.to_numpy(),saved_meta.feature.to_numpy())
    np.testing.assert_array_equal(meta.loc[meta.feature_type=='snp','snp_index'],saved_meta.loc[saved_meta.feature_type=='snp','snp_index'])
    saved_provenance=json.loads((rf_history/'provenance.json').read_text())
    import hashlib
    assert hashlib.sha256(Xrf.tobytes()).hexdigest()==saved_provenance['matrix_sha256']
    assert hashlib.sha256(y.tobytes()).hexdigest()==saved_provenance['phenotype_sha256']
    saved_splits=pd.read_csv(history/'splits.tsv',sep='\t',dtype={'accession':str})
    ids=np.empty(81,dtype='<U32')
    for row in saved_splits.itertuples():
        ids[row.row]=row.accession
    assert len(set(ids))==81
    # Verify accession/phenotype ordering independently against the original loader.
    from full_baseline_gwas import load_phenotype
    import h5py
    ymap,_=load_phenotype(loader.PHENO_ZIP,'27/study_27_values.csv','Mean(TRL) C')
    with h5py.File(loader.GENO_H5) as handle:
        all_ids=[a.decode() if isinstance(a,bytes) else str(a) for a in handle['accessions'][:]]
    np.testing.assert_array_equal(ids,[a for a in all_ids if a in ymap])
    np.testing.assert_allclose(y,[ymap[a] for a in ids],rtol=0,atol=1e-12)
    chroms=meta.loc[meta.feature_type=='snp','chromosome'].to_numpy()
    labels=sorted(set(chroms))+['PCs']
    groups=[np.flatnonzero(chroms==c) for c in labels[:-1]]+[np.arange(200,203)]
    write_npz(out/'cohort.npz',X_rf=Xrf,X_kan=Xkan,y=y,y_kan=ykan,sample_ids=ids,labels=np.array(labels))
    meta.to_csv(out/'feature_meta.tsv',sep='\t',index=False)
    write_json(out/'manifest.json',dict(design='fixed full-phenotype-selected 200-SNP panel; 3 full-cohort kinship PCs',
        models=['rf','kan'],folds=5,split_seed=42,model_seeds=[43,44,45,46,47],
        checkpoint_policy='reconstruct missing fits; require historical numerical parity; save new checkpoints',
        backgrounds=8,background_seed='9000 + fold',coalitions=64,group_labels=labels,
        groups=[g.tolist() for g in groups],units='native Mean(TRL) C',
        input_sha256=inputs_hash,genotype_source=dict(path=str(loader.GENO_H5),
            size=loader.GENO_H5.stat().st_size,mtime_ns=loader.GENO_H5.stat().st_mtime_ns),
        versions={n:importlib.metadata.version(n) for n in ['numpy','scipy','torch','pykan','scikit-learn','h5py','joblib']},
        device='CPU; exact small six-player game'))
    kan_pred=pd.read_csv(history/'predictions.tsv',sep='\t')
    rf_pred=pd.read_csv(rf_history/'predictions.tsv',sep='\t')
    reference_groups=pd.read_csv(history/'group_shap.tsv',sep='\t')
    reference_pairs=pd.read_csv(history/'group_interactions.tsv',sep='\t')
    ref_trace=pd.read_csv(history/'training_trace.tsv',sep='\t')
    ref_metric=pd.read_csv(history/'cv_metrics.tsv',sep='\t')
    rf_splits=json.loads((rf_history/'splits.json').read_text())
    metrics,group_rows,pair_rows,ranks,predictions,parity,designs=[],[],[],[],[],[],[]
    torch.set_num_threads(2)
    max_eff=0.
    with threadpool_limits(limits=2):
        for fold,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=42).split(Xrf),1):
            saved=lambda role: saved_splits.loc[(saved_splits.fold==fold)&(saved_splits.role==role),'row'].to_numpy(int)
            fit,val=saved('fit'),saved('validation')
            np.testing.assert_array_equal(te,saved('test'))
            np.testing.assert_array_equal(tr,rf_splits[fold-1]['train'])
            assert set(fit).isdisjoint(val) and set(fit)|set(val)==set(tr) and set(tr).isdisjoint(te)
            bg=np.random.default_rng(9000+fold).choice(fit,min(8,len(fit)),replace=False)
            assert set(bg)<=set(fit)
            design=dict(fold=fold,outer_train=tr.tolist(),fit=fit.tolist(),validation=val.tolist(),
                heldout=te.tolist(),background=bg.tolist(),background_ids=ids[bg].tolist())
            designs.append(design)
            for name in ['rf','kan']:
                print(f'Fold {fold}, {name.upper()}: reconstruct fit and check saved results',flush=True)
                target=out/f'fold_{fold}'/name
                target.mkdir(parents=True)
                if name=='rf':
                    model=RandomForestRegressor(n_estimators=500,max_features='sqrt',min_samples_leaf=2,
                        random_state=42+fold,n_jobs=2).fit(Xrf[tr],y[tr])
                    predict=model.predict
                    X=Xrf
                    actual_train=tr
                    expected=rf_pred[(rf_pred.fold==fold)&(rf_pred.model=='snps_plus_pcs')].sort_values('row')
                    np.testing.assert_array_equal(te,expected.row)
                    check_reference(rf_history/'predictions.tsv',parity,predict(X[te]),expected.predicted,1e-10)
                    joblib.dump(model,target/'model.joblib')
                    write_json(target/'preprocessing.json',dict(kind='unscaled historical inputs',model_seed=42+fold))
                    roles={'train':tr,'heldout':te,'outer_train':tr}
                else:
                    mean,sd=Xkan[fit].mean(0),Xkan[fit].std(0)
                    sd[sd<1e-8]=1.
                    X=(Xkan-mean)/sd
                    predict,trace,epoch=fit_kan(X,ykan,fit,val,42+fold,target)
                    expected=kan_pred[kan_pred.fold==fold].sort_values('row')
                    np.testing.assert_array_equal(te,expected.row)
                    check_reference(history/'predictions.tsv',parity,predict(X[te]),expected.predicted,1e-5)
                    assert epoch==int(ref_metric.loc[ref_metric.fold==fold,'best_epoch'].iloc[0])
                    check_reference(history/'training_trace.tsv',parity,
                        pd.DataFrame(trace)[['train_mse_standardized','validation_mse_standardized']].to_numpy(),
                        ref_trace[ref_trace.fold==fold][['train_mse_standardized','validation_mse_standardized']].to_numpy(),1e-6)
                    closure=inspect.getclosurevars(predict).nonlocals
                    model=closure['model']
                    torch.save(model.state_dict(),target/'model.pt')
                    write_json(target/'preprocessing.json',dict(kind='historical fit-only z score',
                        input_mean=mean,input_sd=sd,target_mean=closure['mu'],target_sd=closure['sd'],
                        width=[203,4,1],grid=3,k=3,model_seed=42+fold,best_epoch=epoch))
                    pd.DataFrame(trace).to_csv(target/'training_trace.tsv',sep='\t',index=False)
                    actual_train=fit
                    roles={'train':fit,'validation':val,'outer_train':tr,'heldout':te}
                # One coalition evaluation for all 81 rows gives identical local
                # estimates across subsets and saves exact signed local interactions.
                print(f'Fold {fold}, {name.upper()}: 64-coalition game on all 81 accessions',flush=True)
                game=exact_group_game(predict,X[bg],X,groups)
                pred=predict(X)
                np.testing.assert_allclose(game['coalitions'][-1,:,0],pred,rtol=0,atol=1e-5)
                np.testing.assert_allclose(game['coalitions'][0,:,0],predict(X[bg]).mean(),rtol=0,atol=1e-5)
                assert np.isfinite(game['phi']).all() and np.isfinite(game['interactions']).all()
                max_eff=max(max_eff,game['efficiency_error'])
                write_npz(target/'explanations.npz',sample_ids=ids,background_ids=ids[bg],
                    labels=np.array(labels),observed=y,predicted=pred,**game)
                held_phi=np.abs(game['phi'][te,:,0]).mean(0)
                held_int=np.abs(game['interactions'][te,:,0]).mean(0)
                if name=='kan':
                    ref=reference_groups[reference_groups.fold==fold].set_index('group').loc[labels]
                    check_reference(history/'group_shap.tsv',parity,held_phi,ref.mean_abs_shap.to_numpy(),1e-5)
                    ref=reference_pairs[reference_pairs.fold==fold].set_index(['group1','group2'])
                    expected=[ref.loc[(labels[a],labels[b]),'mean_abs_interaction'] for a,b in game['pairs']]
                    check_reference(history/'group_interactions.tsv',parity,held_int,expected,1e-5)
                for role,ix in roles.items():
                    metric=regression_metrics(y[ix],pred[ix])
                    metrics.append(dict(model=name,fold=fold,role=role,**metric))
                    predictions.extend(dict(model=name,fold=fold,role=role,row=int(i),sample_id=ids[i],
                        observed=float(y[i]),predicted=float(pred[i])) for i in ix)
                    phi=game['phi'][ix,:,0]
                    interaction=game['interactions'][ix,:,0]
                    mean_abs=np.abs(phi).mean(0)
                    pair_abs=np.abs(interaction).mean(0)
                    group_rows.extend(dict(model=name,fold=fold,role=role,group=label,n=len(ix),
                        mean_abs=float(mean_abs[j]),mean_signed=float(phi[:,j].mean())) for j,label in enumerate(labels))
                    pair_rows.extend(dict(model=name,fold=fold,role=role,group_a=labels[a],group_b=labels[b],n=len(ix),
                        mean_abs=float(pair_abs[j]),mean_signed=float(interaction[:,j].mean())) for j,(a,b) in enumerate(game['pairs']))
                    if role!='heldout':
                        no_pc=np.array([labels[b]!='PCs' for a,b in game['pairs']])
                        for level,a,b in [('group',mean_abs,held_phi),('chromosome',mean_abs[:-1],held_phi[:-1]),
                            ('interaction',pair_abs,held_int),('chromosome_interaction',pair_abs[no_pc],held_int[no_pc])]:
                            rho,overlap=compare(a,b)
                            ranks.append(dict(model=name,fold=fold,role=role,level=level,rho=rho,top3_overlap=overlap,n_scores=len(a)))
                print(f'Fold {fold}, {name.upper()}: fitting R2={metrics[-len(roles)]["r2"]:.4f}, '
                      f'group rho={next(r["rho"] for r in ranks if r["model"]==name and r["fold"]==fold and r["role"]=="train" and r["level"]=="group"):.4f}',flush=True)
    for filename,rows in [('metrics',metrics),('group_by_fold',group_rows),('interaction_by_fold',pair_rows),
        ('rank_comparison',ranks),('predictions',predictions)]:
        pd.DataFrame(rows).to_csv(out/(filename+'.tsv'),sep='\t',index=False)
    write_json(out/'splits.json',designs)
    write_json(out/'validation.json',dict(models=10,historical_reference_checks=parity,
        max_local_efficiency_error=max_eff,group_partition_passed=True,background_fit_only=True,
        heldout_coverage_per_model=81,validation_separate_from_actual_fitting=True))
    report(out)
    # Ensure every consumed source file is still identical at completion.
    for name,value in inputs_hash.items():
        assert digest(ROOT/name)==value,f'Input changed during run: {name}'
    write_json(out/'COMPLETE.json',dict(models=10,folds=5,
        manifest_sha256=digest(out/'manifest.json'),
        output_sha256={str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='COMPLETE.json'}))
    print((out/'REPORT.md').read_text(),flush=True)


if __name__=='__main__':
    main()
