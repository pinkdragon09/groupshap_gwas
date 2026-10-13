#!/usr/bin/env python3
"""Verify yeast fold selection, split hygiene, receipts and model checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from fivefold_common import check_receipt, digest, fit_standardizer, validate_splits, write_json
from yeast_fivefold_common import ROOT, TRAITS, load_trait, map_references, new_kan, predict_kan, select_panel


def verify(out: Path):
    manifest = json.loads((out/'manifest.json').read_text())
    config, signature = manifest['config'], manifest['fingerprint']
    for path, expected in manifest['inputs'].items():
        if digest(Path(path)) != expected:
            raise ValueError('Changed input: '+path)
    for path, expected in manifest['source'].items():
        if digest(ROOT/path) != expected:
            raise ValueError('Changed source: '+path)

    results = []
    torch.set_num_threads(config['threads'])
    with threadpool_limits(limits=config['threads']):
        for trait in TRAITS:
            directory = out/trait
            if not check_receipt(directory, 'PREPARED.json', signature):
                raise ValueError(f'Missing prepared receipt: {trait}')
            cohort = json.loads((directory/'cohort.json').read_text())
            splits = json.loads((directory/'splits.json').read_text())
            selected = json.loads((directory/'selected_features.json').read_text())
            references = json.loads((directory/'reference_mapping.json').read_text())
            ids, raw_x, y, marker_meta = load_trait(trait)
            if ids != cohort['sample_ids']:
                raise ValueError(f'Cohort ID mismatch: {trait}')
            np.testing.assert_allclose(y, cohort['y'], rtol=0, atol=0)
            validate_splits(splits, len(ids))
            with np.load(directory/'panels.npz', allow_pickle=False) as panels:
                for split in splits:
                    fold = split['fold']
                    for stage, role in [('outer', 'train'), ('inner', 'fit')]:
                        key = f'{stage}{fold}'
                        expected_ix, expected_rows = select_panel(raw_x[split[role]], y[split[role]], marker_meta,
                            config['max_markers'], config['min_maf'], config['prune_bp'])
                        np.testing.assert_array_equal(panels[key+'_indices'], expected_ix)
                        np.testing.assert_array_equal(panels[key], raw_x[:, expected_ix])
                        if selected[key] != expected_rows:
                            raise ValueError(f'Fold-local selection mismatch: {trait} {key}')
                        if stage == 'outer' and references[key] != map_references(
                                trait, expected_rows, config['mapping_distance_bp']):
                            raise ValueError(f'Reference mapping mismatch: {trait} {key}')

                    fold_dir = directory/f'fold_{fold}'
                    if not check_receipt(fold_dir, 'COMPLETE.json', signature):
                        continue
                    shared = json.loads((fold_dir/'shared.json').read_text())
                    train, test = np.asarray(split['train']), np.asarray(split['test'])
                    fit = np.asarray(split['fit'])
                    mu, sd = fit_standardizer(y[train])
                    imu, isd = fit_standardizer(y[fit])
                    np.testing.assert_allclose([shared['target_mean'], shared['target_sd']], [mu, sd])
                    np.testing.assert_allclose([shared['inner_target_mean'], shared['inner_target_sd']], [imu, isd])
                    for role in ['train', 'test', 'fit', 'validation']:
                        np.testing.assert_array_equal(shared[role+'_ids'], np.asarray(ids)[split[role]])
                    bg = np.random.default_rng(config['background_seed']+fold).choice(
                        train, config['background'], replace=False)
                    ev = np.random.default_rng(config['evaluation_seed']+fold).choice(
                        test, min(config['evaluation'], len(test)), replace=False)
                    np.testing.assert_array_equal(shared['background_ids'], np.asarray(ids)[bg])
                    np.testing.assert_array_equal(shared['interaction_evaluation_ids'], np.asarray(ids)[ev])

                    outer_x = panels[f'outer{fold}']
                    for family in ['rf', 'kan']:
                        for seed in config['seeds']:
                            model_dir = fold_dir/f'{family}_{seed}'
                            if not check_receipt(model_dir, 'COMPLETE.json', signature):
                                raise ValueError(f'Missing checkpoint receipt: {trait} fold {fold} {family} {seed}')
                            if family == 'rf':
                                model = joblib.load(model_dir/'model.joblib')
                                pred = model.predict(outer_x[test])*sd+mu
                            else:
                                model = new_kan(outer_x.shape[1], seed+100*fold)
                                model.load_state_dict(torch.load(model_dir/'model.pt', map_location='cpu', weights_only=True))
                                pred = predict_kan(model, outer_x[test], mu, sd)
                            with np.load(model_dir/'evaluation.npz', allow_pickle=False) as saved:
                                np.testing.assert_array_equal(saved['sample_ids'], np.asarray(ids)[test])
                                np.testing.assert_array_equal(saved['observed'], y[test])
                                np.testing.assert_allclose(pred, saved['predicted'], rtol=1e-7, atol=1e-7)
                                error = float(np.max(np.abs(pred-saved['predicted'])))
                            results.append(dict(trait=trait, fold=fold, family=family,
                                                seed=seed, max_prediction_error=error))
    if not results:
        raise ValueError('No completed yeast folds to verify')
    write_json(out/'checkpoint_validation.json', dict(passed=True, checked_models=len(results),
        full_fivefold_complete=len(results)==60, input_and_source_hashes_checked=True,
        fold_local_selection_and_reference_mapping_checked=True,
        split_target_scale_and_interaction_membership_checked=True, results=results))
    print(f'Passed yeast checkpoint and preprocessing checks: {len(results)}/60 models', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, required=True)
    verify(parser.parse_args().outdir.resolve())
