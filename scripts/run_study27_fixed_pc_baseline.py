"""Matched PC-only RF comparison conditional on the historical fixed SNP panel."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

import run_phase5 as source


def run(out):
    X, y, meta = source.load_arabidopsis_matrix()
    pcs = meta.loc[meta.feature_type == 'covariate', 'feature'].tolist()
    saved_path = source.REPO_ROOT / 'outputs/random_forest_gwas_full/cv_metrics.tsv'
    saved = pd.read_csv(saved_path, sep='\t').sort_values('fold')
    reproduction_path = source.REPO_ROOT / 'outputs/study27_kan_historical/rf_reproduction.tsv'
    reproduced = pd.read_csv(reproduction_path, sep='\t').sort_values('fold')
    assert len(X) == 81 and len(pcs) == 3
    records, predictions, splits = [], [], []
    with threadpool_limits(limits=4):
        for fold, (train, test) in enumerate(KFold(5, shuffle=True, random_state=42).split(X), 1):
            splits.append(dict(fold=fold, train=train.tolist(), test=test.tolist()))
            for name, columns in [('snps_plus_pcs', X.columns.tolist()), ('pc_only', pcs)]:
                model = RandomForestRegressor(n_estimators=500, max_features='sqrt',
                    min_samples_leaf=2, random_state=42+fold, n_jobs=4)
                model.fit(X.iloc[train][columns], y.iloc[train])
                pred = model.predict(X.iloc[test][columns])
                score = r2_score(y.iloc[test], pred)
                if name == 'snps_plus_pcs':
                    np.testing.assert_allclose(score, reproduced.iloc[fold-1].r2, rtol=0, atol=1e-10)
                records.append(dict(model=name, fold=fold, r2=score,
                    rmse=float(np.sqrt(mean_squared_error(y.iloc[test], pred)))))
                predictions.extend(dict(model=name, fold=fold, row=int(i),
                    observed=float(y.iloc[i]), predicted=float(p)) for i,p in zip(test,pred))
    metrics = pd.DataFrame(records)
    paired = metrics.pivot(index='fold', columns='model', values='r2')
    paired['difference'] = paired.snps_plus_pcs - paired.pc_only
    summary = dict(n=len(X), n_snps=len(X.columns)-len(pcs), n_pcs=len(pcs),
        design='Fixed full-phenotype-selected SNP panel; original full-cohort kinship PCs; same folds and RF settings in both arms.',
        interpretation='Conditional descriptive comparison; phenotype-informed panel selection is not independently validated.',
        prior_rf_reproduction_passed=True,
        historical_reported_mean_r2=float(saved.r2.mean()),
        max_absolute_difference_from_historical_r2=float(np.max(np.abs(paired.snps_plus_pcs.to_numpy()-saved.r2.to_numpy()))),
        models={name:dict(mean_fold_r2=float(g.r2.mean()), sd_fold_r2=float(g.r2.std(ddof=1)))
                for name,g in metrics.groupby('model')},
        mean_paired_r2_difference=float(paired.difference.mean()),
        sd_paired_r2_difference=float(paired.difference.std(ddof=1)))
    provenance = dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        loader_sha256=hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest(),
        historical_metrics_sha256=hashlib.sha256(saved_path.read_bytes()).hexdigest(),
        prior_reproduction_sha256=hashlib.sha256(reproduction_path.read_bytes()).hexdigest(),
        matrix_sha256=hashlib.sha256(X.to_numpy().tobytes()).hexdigest(),
        phenotype_sha256=hashlib.sha256(y.to_numpy().tobytes()).hexdigest(),
        sklearn=importlib.metadata.version('scikit-learn'),
        forest=dict(n_estimators=500,max_features='sqrt',min_samples_leaf=2),
        split_seed=42, model_seeds=[43,44,45,46,47], pc_columns=pcs)
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out/'metrics.tsv',sep='\t',index=False)
    pd.DataFrame(predictions).to_csv(out/'predictions.tsv',sep='\t',index=False)
    paired.to_csv(out/'paired_comparison.tsv',sep='\t')
    for name,value in [('summary',summary),('provenance',provenance),('splits',splits)]:
        (out/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
    meta.to_csv(out/'feature_meta.tsv',sep='\t',index=False)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,default=source.REPO_ROOT/'outputs/study27_fixed_pc_baseline')
    run(parser.parse_args().outdir)
