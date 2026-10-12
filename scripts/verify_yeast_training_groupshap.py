#!/usr/bin/env python3
"""Verify source immutability, train/test separation and explanation conservation."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from fivefold_common import digest, write_json
from run_yeast_interactions import FastBinaryKAN
from yeast_fivefold_common import ROOT, TRAITS, new_kan


def verify(out):
    manifest=json.loads((out/'manifest.json').read_text())
    complete=json.loads((out/'COMPLETE.json').read_text())
    assert complete['status']=='complete' and complete['models']==20
    assert complete['original_inputs_unchanged']
    assert complete['fingerprint']==manifest['fingerprint']
    for section in ['input_sha256','code_sha256']:
        for path,sha in manifest[section].items():assert digest(ROOT/path)==sha,path
    for path,sha in complete['output_sha256'].items():assert digest(out/path)==sha,path
    source=ROOT/manifest['source']
    comparisons=pd.read_csv(out/'group_comparison_by_fold.tsv',sep='\t')
    markers=pd.read_csv(out/'marker_comparison_by_fold.tsv',sep='\t')
    assert len(comparisons)==len(markers)==20
    assert not comparisons.duplicated(['trait','fold','method']).any()
    checked=0;max_error=0.0
    for trait in TRAITS:
        cohort=json.loads((source/trait/'cohort.json').read_text())
        lookup={s:i for i,s in enumerate(cohort['sample_ids'])}
        all_tests=[]
        with np.load(source/trait/'panels.npz',allow_pickle=False) as panels:
            for fold in range(1,6):
                design=json.loads((out/trait/f'fold_{fold}'/'design.json').read_text())
                ids=design['sample_ids'];train=set(ids['train']);test=set(ids['heldout']);background=set(ids['background'])
                assert not train & test
                assert background <= train
                assert set(ids['interaction_train']) <= train-background
                assert set(ids['interaction_heldout']) <= test
                assert len(ids['interaction_train'])==len(ids['interaction_heldout'])==20
                all_tests.extend(ids['heldout'])
                x=panels[f'outer{fold}']
                columns=[c for g in design['groups'] for c in g['indices']]
                assert sorted(columns)==list(range(x.shape[1]))
                group_names=[g['name'] for g in design['groups']]
                for method in ['rf','kan']:
                    target=out/trait/f'fold_{fold}'/method
                    receipt=json.loads((target/'COMPLETE.json').read_text())
                    assert receipt['fingerprint']==manifest['fingerprint']
                    for path,sha in receipt['outputs_sha256'].items():assert digest(target/path)==sha,path
                    if method=='rf':
                        model=joblib.load(source/trait/f'fold_{fold}/rf_42/model.joblib');model.n_jobs=1;predict=model.predict
                    else:
                        model=new_kan(x.shape[1],42+100*fold)
                        model.load_state_dict(torch.load(source/trait/f'fold_{fold}/kan_42/model.pt',map_location='cpu',weights_only=True))
                        predict=FastBinaryKAN(model).predict
                    bg=x[[lookup[s] for s in ids['background']]]
                    baseline=np.asarray(predict(bg),dtype=float).mean()
                    with np.load(target/'local_explanations.npz',allow_pickle=False) as local:
                        np.testing.assert_array_equal(local['group_names'],group_names)
                        for role in ['train','heldout']:
                            np.testing.assert_array_equal(local[f'{role}_sample_ids'],ids[role])
                            np.testing.assert_array_equal(local[f'{role}_interaction_sample_ids'],ids[f'interaction_{role}'])
                            phi=local[f'{role}_phi'];half=local[f'{role}_phi_halves']
                            assert phi.shape==(len(ids[role]),len(group_names)) and np.isfinite(phi).all()
                            np.testing.assert_allclose(half.mean(0),phi,atol=1e-12)
                            expected=np.asarray(predict(x[[lookup[s] for s in ids[role]]]),dtype=float)-baseline
                            error=float(np.abs(phi.sum(1)-expected).max());max_error=max(max_error,error)
                            np.testing.assert_allclose(phi.sum(1),expected,rtol=1e-4,atol=3e-5)
                            index=local[f'{role}_group_interactions']
                            assert index.shape==(20,len(list(itertools.combinations(group_names,2))))
                            assert np.isfinite(index).all()
                        heldout=np.load(source/trait/f'fold_{fold}/{method}_42/evaluation.npz',allow_pickle=False)
                        shared=json.loads((source/trait/f'fold_{fold}/shared.json').read_text())
                        np.testing.assert_allclose(predict(x[[lookup[s] for s in ids['heldout']]])*shared['target_sd']+shared['target_mean'],heldout['predicted'],rtol=1e-4,atol=1e-5)
                    train_ranking=pd.read_csv(target/'training_marker_interactions.csv')
                    old=pd.read_csv(source/trait/f'fold_{fold}/interactions'/f'{method}.csv')
                    assert set(map(tuple,train_ranking[['j_idx','k_idx']].to_numpy()))==set(map(tuple,old[['j_idx','k_idx']].to_numpy()))
                    assert train_ranking.mean_abs_interaction.is_monotonic_decreasing
                    checked+=1
        assert len(all_tests)==len(set(all_tests))==len(cohort['sample_ids'])
    write_json(out/'validation.json',dict(status='passed',models_checked=checked,
        all_heldout_samples_explained_once=True,source_and_output_checksums_verified=True,
        checkpoint_predictions_verified=True,training_background_only=True,
        maximum_efficiency_error_against_reloaded_predictor=max_error))
    print(f'Verified {checked} models: source/output hashes, sample separation, complete out-of-fold coverage, saved prediction parity and local attribution conservation (max error {max_error:.3g}).')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outdir',type=Path,default=ROOT/'outputs/yeast_bloom_training_groupshap')
    torch.set_num_threads(2)
    verify(p.parse_args().outdir.resolve())
