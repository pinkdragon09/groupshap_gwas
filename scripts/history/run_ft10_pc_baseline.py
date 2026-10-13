"""Matched PC-only RF baseline, with its own provenance and checkpoint checks."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from threadpoolctl import threadpool_limits
from fivefold_common import (check_receipt,digest,fingerprint,fit_standardizer,
    regression_metrics,save_receipt,write_json,write_npz)
from run_study27_fivefold import inputs


def run(out):
    manifest=json.loads((out/'manifest.json').read_text())
    if not check_receipt(out,'PREPARED.json',manifest['fingerprint']):
        raise ValueError('Verified prepared panels required')
    config=manifest['config']
    protocol=dict(parent_fingerprint=manifest['fingerprint'],source_sha256=digest(Path(__file__)),
        model='PC-only RF with the same forest hyperparameters and seeds as the full RF',
        forest=config['rf'],seeds=config['seeds'])
    sig=fingerprint(protocol)
    directory=out/'pc_rf'
    directory.mkdir(exist_ok=True)
    if (directory/'manifest.json').exists():
        if json.loads((directory/'manifest.json').read_text())!=protocol:
            raise ValueError('PC-only baseline code/configuration changed')
    else:
        write_json(directory/'manifest.json',protocol)
    y=np.array(json.loads((out/'cohort.json').read_text())['y'])
    ids=np.array(json.loads((out/'cohort.json').read_text())['sample_ids'])
    splits=json.loads((out/'splits.json').read_text())
    records,receipts,predictions=[],[],{}
    with np.load(out/'panels.npz',allow_pickle=False) as panels,threadpool_limits(limits=4):
        for s in splits:
            fold=s['fold']
            x,_=inputs(panels[f'outer{fold}'],panels[f'outer{fold}_pcs'],s['train'])
            mu,sd=fit_standardizer(y[s['train']])
            for seed in config['seeds']:
                d=directory/f'fold_{fold}_seed_{seed}'
                d.mkdir(exist_ok=True)
                receipts.append(f'{d.name}/COMPLETE.json')
                if not check_receipt(d,'COMPLETE.json',sig):
                    rf=config['rf']
                    model=RandomForestRegressor(n_estimators=rf['trees'],max_features=rf['max_features'],
                        min_samples_leaf=rf['min_samples_leaf'],random_state=seed+100*fold,n_jobs=4)
                    model.fit(x[s['train'],200:],(y[s['train']]-mu)/sd)
                    model.n_jobs=1
                    joblib.dump(model,d/'model.joblib')
                    pred=model.predict(x[s['test'],200:])*sd+mu
                    write_npz(d/'evaluation.npz',sample_ids=ids[s['test']],observed=y[s['test']],predicted=pred)
                    save_receipt(d,'COMPLETE.json',sig,['model.joblib','evaluation.npz'])
                model=joblib.load(d/'model.joblib')
                pred=model.predict(x[s['test'],200:])*sd+mu
                with np.load(d/'evaluation.npz') as a:
                    np.testing.assert_allclose(pred,a['predicted'],atol=1e-10)
                    np.testing.assert_array_equal(a['sample_ids'],ids[s['test']])
                    np.testing.assert_array_equal(a['observed'],y[s['test']])
                records.append(dict(model='rf_pc_only',fold=fold,seed=seed,**regression_metrics(y[s['test']],pred)))
                predictions.setdefault(seed,[]).append((y[s['test']],pred))
    scores=[np.mean([r['r2'] for r in records if r['fold']==f]) for f in range(1,6)]
    pooled=[regression_metrics(np.concatenate([a for a,b in predictions[seed]]),
                                np.concatenate([b for a,b in predictions[seed]]))['r2'] for seed in config['seeds']]
    summary=dict(model='rf_pc_only',mean_fold_r2=np.mean(scores),sd_fold_r2=np.std(scores,ddof=1),
        mean_seed_pooled_oof_r2=np.mean(pooled),pooled_oof_r2_by_seed=pooled,
        completed_models=len(records),checkpoint_reproduction_passed=True)
    write_json(directory/'metrics.json',records)
    write_json(directory/'summary.json',summary)
    save_receipt(directory,'COMPLETE.json',sig,['manifest.json','metrics.json','summary.json']+receipts)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outdir',type=Path,required=True)
    run(p.parse_args().outdir.resolve())
