#!/usr/bin/env python3
"""Verify receipts, reconstructed checkpoints and complete Study 27 coalition games."""
import argparse
import json
from pathlib import Path

import joblib
from kan import KAN
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from fivefold_common import digest, exact_group_game, regression_metrics, write_json
from study27_kan_comparison import predictor

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,default=ROOT/'outputs/study27_training_groupshap_20261010')
    args=parser.parse_args()
    out=args.outdir.resolve()
    manifest,receipt=read(out/'manifest.json'),read(out/'COMPLETE.json')
    assert digest(out/'manifest.json')==receipt['manifest_sha256']
    for filename,expected in receipt['output_sha256'].items():
        assert digest(out/filename)==expected,f'Output hash mismatch: {filename}'
    for filename,expected in manifest['input_sha256'].items():
        assert digest(ROOT/filename)==expected,f'Input hash mismatch: {filename}'
    with np.load(out/'cohort.npz',allow_pickle=False) as data:
        Xrf,Xkan,y,ykan,ids,labels=[data[k].copy() for k in ['X_rf','X_kan','y','y_kan','sample_ids','labels']]
    assert len(set(ids))==81 and labels.tolist()==manifest['group_labels']
    groups=[np.array(g,dtype=int) for g in manifest['groups']]
    assert sorted(int(i) for g in groups for i in g)==list(range(203))
    assert groups[-1].tolist()==[200,201,202]
    meta=pd.read_csv(out/'feature_meta.tsv',sep='\t')
    for label,cols in zip(labels[:-1],groups[:-1]):
        assert set(meta.iloc[cols].chromosome)=={label}
    designs=read(out/'splits.json')
    metrics=pd.read_csv(out/'metrics.tsv',sep='\t')
    by_group=pd.read_csv(out/'group_by_fold.tsv',sep='\t')
    by_pair=pd.read_csv(out/'interaction_by_fold.tsv',sep='\t')
    historical_kan=pd.read_csv(ROOT/'outputs/study27_kan_historical/predictions.tsv',sep='\t')
    historical_rf=pd.read_csv(ROOT/'outputs/study27_fixed_pc_baseline/predictions.tsv',sep='\t')
    checks=[]
    tests=[]
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        for design in designs:
            fold=design['fold']
            tr,fit,val,te,bg=[np.array(design[k],dtype=int) for k in ['outer_train','fit','validation','heldout','background']]
            assert set(tr).isdisjoint(te) and set(tr)|set(te)==set(range(81))
            assert set(fit).isdisjoint(val) and set(fit)|set(val)==set(tr)
            assert len(bg)==len(set(bg))==8 and set(bg)<=set(fit)
            np.testing.assert_array_equal(ids[bg],design['background_ids'])
            tests.extend(te)
            for name in ['rf','kan']:
                directory=out/f'fold_{fold}'/name
                prep=read(directory/'preprocessing.json')
                if name=='rf':
                    model=joblib.load(directory/'model.joblib')
                    assert model.n_estimators==500 and model.min_samples_leaf==2 and model.max_features=='sqrt'
                    assert model.random_state==42+fold
                    predict=model.predict
                    X=Xrf
                    roles={'train':tr,'outer_train':tr,'heldout':te}
                    old=historical_rf[(historical_rf.fold==fold)&(historical_rf.model=='snps_plus_pcs')].sort_values('row')
                    atol=1e-10
                else:
                    mean=np.array(prep['input_mean'],dtype=np.float32)
                    sd=np.array(prep['input_sd'],dtype=np.float32)
                    expected_sd=Xkan[fit].std(0)
                    expected_sd[expected_sd<1e-8]=1.
                    np.testing.assert_array_equal(mean,Xkan[fit].mean(0))
                    np.testing.assert_array_equal(sd,expected_sd)
                    np.testing.assert_allclose(prep['target_mean'],float(ykan[fit].mean()),rtol=0,atol=0)
                    np.testing.assert_allclose(prep['target_sd'],float(ykan[fit].std()),rtol=0,atol=0)
                    X=(Xkan-mean)/sd
                    model=KAN(width=prep['width'],grid=prep['grid'],k=prep['k'],seed=prep['model_seed'],
                        auto_save=False,symbolic_enabled=False,ckpt_path=str(directory/'unused_checkpoints'))
                    model.load_state_dict(torch.load(directory/'model.pt',map_location='cpu',weights_only=True))
                    model.eval()
                    predict=predictor(model,prep['target_mean'],prep['target_sd'])
                    roles={'train':fit,'validation':val,'outer_train':tr,'heldout':te}
                    old=historical_kan[historical_kan.fold==fold].sort_values('row')
                    atol=1e-5
                np.testing.assert_array_equal(te,old.row)
                np.testing.assert_allclose(predict(X[te]),old.predicted,rtol=0,atol=atol)
                print(f'Verifying fold {fold}, {name.upper()}: saved checkpoint and all 64 coalitions',flush=True)
                game=exact_group_game(predict,X[bg],X,groups)
                with np.load(directory/'explanations.npz',allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['sample_ids'],ids)
                    np.testing.assert_array_equal(saved['background_ids'],ids[bg])
                    np.testing.assert_array_equal(saved['labels'],labels)
                    np.testing.assert_array_equal(saved['pairs'],game['pairs'])
                    np.testing.assert_allclose(saved['predicted'],predict(X),rtol=0,atol=atol)
                    for key in ['coalitions','phi','interactions']:
                        np.testing.assert_allclose(saved[key],game[key],rtol=0,atol=atol)
                    np.testing.assert_allclose(saved['phi'].sum(1),saved['coalitions'][-1]-saved['coalitions'][0],rtol=0,atol=1e-5)
                for role,ix in roles.items():
                    row=metrics[(metrics.model==name)&(metrics.fold==fold)&(metrics.role==role)].iloc[0]
                    actual=regression_metrics(y[ix],predict(X[ix]))
                    assert int(row.n)==len(ix)
                    for key in ['r2','rmse']:
                        np.testing.assert_allclose(row[key],actual[key],rtol=0,atol=1e-6)
                    summary=by_group[(by_group.model==name)&(by_group.fold==fold)&(by_group.role==role)].set_index('group').loc[labels]
                    np.testing.assert_allclose(summary.mean_abs,np.abs(game['phi'][ix,:,0]).mean(0),rtol=0,atol=1e-5)
                    summary=by_pair[(by_pair.model==name)&(by_pair.fold==fold)&(by_pair.role==role)].set_index(['group_a','group_b'])
                    np.testing.assert_allclose([summary.loc[(labels[a],labels[b]),'mean_abs'] for a,b in game['pairs']],
                        np.abs(game['interactions'][ix,:,0]).mean(0),rtol=0,atol=1e-5)
                checks.append(dict(fold=fold,model=name,all_coalitions_recomputed=True,
                    saved_predictions_reproduced=True,efficiency_error=game['efficiency_error']))
    assert sorted(tests)==list(range(81)) and len(checks)==10
    result=dict(passed=True,models=10,source_and_artifact_hashes_passed=True,
        grouping_and_membership_passed=True,heldout_coverage_passed=True,checks=checks,
        verifier_source_sha256=digest(Path(__file__)),completion_receipt_sha256=digest(out/'COMPLETE.json'))
    write_json(out/'VERIFIED.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
