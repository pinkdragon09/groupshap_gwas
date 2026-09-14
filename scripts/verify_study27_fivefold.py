"""Verify input integrity, PCA projection, panel scores and actual checkpoints."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from fivefold_common import check_receipt, digest, fit_standardizer, validate_splits, write_json
from run_study27_fivefold import ROOT, MODELS, inputs, new_model, predictor, project_pcs, rank_chunk


def verify(out):
    manifest = json.loads((out/'manifest.json').read_text())
    config, signature = manifest['config'], manifest['fingerprint']
    for path, expected in {**manifest['source'], **manifest['inputs']}.items():
        if digest(ROOT/path) != expected:
            raise ValueError('Changed source/input: '+path)
    if not check_receipt(out, 'PREPARED.json', signature):
        raise ValueError('Missing prepared artifacts')
    cohort = json.loads((out/'cohort.json').read_text())
    splits = json.loads((out/'splits.json').read_text())
    selected = json.loads((out/'selected_features.json').read_text())
    states = json.loads((out/'pca_transforms.json').read_text())
    y = np.array(cohort['y'])
    validate_splits(splits, len(y))
    results = []
    torch.set_num_threads(4)
    with np.load(out/'panels.npz', allow_pickle=False) as panels, threadpool_limits(limits=4):
        for s in splits:
            fold = s['fold']
            for stage, role in [('outer','train'), ('inner','fit')]:
                key = f'{stage}{fold}'
                tr = s[role]
                pcs, state = project_pcs(panels['genotype_gram'], tr)
                np.testing.assert_allclose(pcs, panels[key+'_pcs'], atol=1e-9)
                np.testing.assert_allclose(state['dual_components'], states[key]['dual_components'], atol=1e-9)
                score = rank_chunk(panels[key][tr].T, y[tr], pcs[tr])
                np.testing.assert_allclose(score, [r['training_score'] for r in selected[key]], rtol=1e-6)
            d = out/f'fold_{fold}'
            if not check_receipt(d, 'COMPLETE.json', signature):
                continue
            x, prep = inputs(panels[f'outer{fold}'], panels[f'outer{fold}_pcs'], s['train'])
            mu, sd = fit_standardizer(y[s['train']])
            shared = json.loads((d/'shared.json').read_text())
            np.testing.assert_allclose(shared['target_mean'], mu)
            np.testing.assert_allclose(shared['target_sd'], sd)
            for key in prep:
                np.testing.assert_allclose(shared['outer_preprocessing'][key], prep[key])
            imu, isd = fit_standardizer(y[s['fit']])
            np.testing.assert_allclose(shared['inner_target_mean'], imu)
            np.testing.assert_allclose(shared['inner_target_sd'], isd)
            _, iprep = inputs(panels[f'inner{fold}'], panels[f'inner{fold}_pcs'], s['fit'])
            for key in iprep:
                np.testing.assert_allclose(shared['inner_preprocessing'][key], iprep[key])
            for seed in config['seeds']:
                for name in MODELS:
                    md = d/f'{name}_{seed}'
                    if not check_receipt(md, 'COMPLETE.json', signature):
                        raise ValueError('Missing model receipt')
                    if name=='rf':
                        model = joblib.load(md/'model.joblib')
                    else:
                        model = new_model(name, seed+100*fold)
                        model.load_state_dict(torch.load(md/'model.pt', map_location='cpu', weights_only=True))
                        model.eval()
                    pred = predictor(model, mu, sd)(x[s['test']])
                    with np.load(md/'evaluation.npz', allow_pickle=False) as a:
                        np.testing.assert_allclose(pred, a['predicted'], rtol=1e-7, atol=1e-7)
                        error = float(np.max(np.abs(pred-a['predicted'])))
                    results.append(dict(fold=fold, model=name, seed=seed, max_prediction_error=error))
    if not results:
        raise ValueError('No completed folds to verify')
    write_json(out/'checkpoint_validation.json', dict(passed=True, checked_models=len(results),
        full_fivefold_complete=len(results)==45, input_and_source_hashes_checked=True,
        training_pca_projection_and_panel_scores_checked=True, results=results))
    print(f'Passed checkpoint reproduction and preprocessing checks: {len(results)}/45 models', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outdir', type=Path, required=True)
    verify(p.parse_args().outdir.resolve())
