#!/usr/bin/env python3
"""Refresh active ALE artifacts; archive old files and preserve model checkpoints.

Run with .venv-kan/bin/python scripts/refresh_ale_results.py.
The first archive is immutable across reruns; it is never used as model input.
"""
import argparse
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fivefold_common import check_receipt, digest, write_json
from xai_gwas.attribution.ale import ale_effect

REVIEW = ROOT / 'outputs/ale_review_20261010'


def archive(path):
    path = Path(path)
    target = REVIEW / 'historical' / path.relative_to(ROOT)
    if path.exists() and not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    return target


def phase5():
    import run_phase5 as phase
    X, y, meta = phase.load_arabidopsis_matrix()
    train, _ = next(iter(KFold(5, shuffle=True, random_state=42).split(X)))
    model = RandomForestRegressor(n_estimators=500, max_features='sqrt',
        min_samples_leaf=2, random_state=43, n_jobs=4).fit(X.iloc[train], y.iloc[train])
    path = ROOT / 'outputs/phase5/ale_results.tsv'
    old = pd.read_csv(archive(path), sep='\t')
    rows = []
    for snp in old.feature.unique():
        result = ale_effect(model, X, snp, feature_type='discrete')
        result.insert(0, 'feature', snp)
        m = meta[meta.feature == snp].iloc[0]
        result['chromosome'], result['position'] = m.chromosome, int(m.position)
        rows.append(result)
    new = pd.concat(rows, ignore_index=True)
    new.to_csv(path, sep='\t', index=False)
    comparison = old.merge(new, on=['feature','x','n_obs','chromosome','position'], suffixes=('_old','_new'))
    comparison.to_csv(REVIEW / 'phase5_comparison.tsv', sep='\t', index=False)
    write_json(path.with_name('ale_provenance.json'), dict(PyALE='1.2.0',
        estimator='adjacent-state pooled discrete ALE', n_rows=len(X),
        population='all 81 rows; model fitted on fold 1 training rows; descriptive, not held-out',
        genotype_codes=[0, 1], historical_sha256=digest(archive(path)),
        result_sha256=digest(path), refresh_source_sha256=digest(Path(__file__))))
    return comparison


def study34():
    import run_study34_xai as study
    scan = pd.read_csv(study.SCAN_CSV).nlargest(50, 'cv_r2').reset_index(drop=True)
    ids, _, indices, y = study.align_samples(study.load_phenotype())
    pcs = study.load_kinship_pcs(ids, study.N_PCS)
    blocks = [study.load_window_snps(int(r.snp_start), int(r.snp_start)+study.WINDOW_SIZE, indices)
              for r in scan.itertuples()]
    raw = np.hstack([np.hstack(blocks).astype(np.float32), pcs])
    scaler = StandardScaler().fit(raw)
    X = scaler.transform(raw)
    model = RandomForestRegressor(n_estimators=study.N_TREES, n_jobs=4,
        random_state=study.RANDOM_STATE, oob_score=True).fit(X, y)
    path = ROOT / 'outputs/study34_xai/ale_top_snps.csv'
    old = pd.read_csv(archive(path))
    ranked = np.argsort(model.feature_importances_[:5000])[::-1][:study.ALE_TOP_N]
    np.testing.assert_array_equal(ranked, old.feature_idx)
    np.testing.assert_allclose(model.feature_importances_[ranked], old.rf_importance, rtol=1e-8, atol=1e-10)
    new = old.copy(); curves = []
    for index, row in new.iterrows():
        j = int(row.feature_idx)
        states, values = study.compute_ale_1d(model, X, j)
        new.loc[index, 'ale_range'] = np.ptp(values)
        dosage = states * scaler.scale_[j] + scaler.mean_[j]
        for state, code, value in zip(states, dosage, values):
            curves.append(dict(feature_idx=j, genotype_code=float(code),
                model_input=float(state), ale=float(value), n_obs=int((X[:, j]==state).sum())))
    new.to_csv(path, index=False)
    pd.DataFrame(curves).to_csv(path.with_name('ale_curves.csv'), index=False)
    old.merge(new[['feature_idx','ale_range']], on='feature_idx', suffixes=('_old','_new')).to_csv(
        REVIEW/'study34_comparison.csv', index=False)
    write_json(path.with_name('ale_provenance.json'), dict(PyALE='1.2.0', n_rows=len(y),
        population='full-data fitted RF; descriptive, not held-out',
        historical_top10_and_importances_reproduced=True, historical_sha256=digest(archive(path)),
        result_sha256=digest(path), scan_sha256=digest(Path(study.SCAN_CSV)),
        refresh_source_sha256=digest(Path(__file__))))
    return new


def fivefold(out):
    import torch
    from run_study27_fivefold import inputs, new_model, predictor
    from summarize_study27_fivefold import summarize
    from PyALE import ale as reference_ale
    torch.set_num_threads(4)
    manifest = json.loads((out/'manifest.json').read_text())
    signature = manifest['fingerprint']
    if not check_receipt(out, 'COMPLETE.json', signature):
        raise ValueError(f'Invalid completion receipt: {out}')
    raw = json.loads((out/'snp_ranks_ale_by_fold_seed.json').read_text())
    summary_path = out/'snp_selection_ale_summary.json'
    old = json.loads(archive(summary_path).read_text())
    for name in ['summary_provenance.json','SUMMARY_COMPLETE.json']:
        archive(out/name)
    checks = []
    with np.load(out/'panels.npz') as panels:
        for split in json.loads((out/'splits.json').read_text()):
            fold = split['fold']; d = out/f'fold_{fold}'
            shared = json.loads((d/'shared.json').read_text())
            X, _ = inputs(panels[f'outer{fold}'], panels[f'outer{fold}_pcs'], split['train'])
            X = X[split['test']]
            for name in ['rf','transformer','kan']:
                for seed in manifest['config']['seeds']:
                    md = d/f'{name}_{seed}'
                    if not check_receipt(md, 'COMPLETE.json', signature):
                        raise ValueError(f'Invalid model receipt: {md}')
                    if name == 'rf':
                        model = joblib.load(md/'model.joblib')
                        model.n_jobs = 4
                    else:
                        model = new_model(name, seed+100*fold)
                        model.load_state_dict(torch.load(md/'model.pt', map_location='cpu', weights_only=True))
                        model.eval()
                    predict = predictor(model, shared['target_mean'], shared['target_sd'])
                    with np.load(md/'evaluation.npz') as a:
                        actual = predict(X)
                        np.testing.assert_allclose(actual, a['predicted'], rtol=1e-6, atol=1e-6)
                        p = ((X[:, :200]+1)/2).mean(axis=0)
                        np.testing.assert_allclose(p, a['test_allele_frequency'])
                        delta = a['binary_ale_local_difference'].mean(axis=0)
                        np.testing.assert_allclose(a['binary_ale_levels'],
                            np.column_stack([-p*delta, (1-p)*delta]), rtol=1e-6, atol=1e-6)
                        # Independent package check at the highest-ranked supported SNP.
                        supported = (p > 0) & (p < 1)
                        order = np.argsort(-a['snp_permutation_delta_mse'].mean(1), kind='stable')
                        j = next(int(j) for j in order if supported[j])
                        class Adapter:
                            def predict(self, frame):
                                return predict(frame.to_numpy(dtype=X.dtype))
                        frame = pd.DataFrame(X, columns=[f'x{k}' for k in range(X.shape[1])])
                        ref = reference_ale(X=frame, model=Adapter(), feature=[f'x{j}'],
                            feature_type='discrete', include_CI=False, plot=False)
                        np.testing.assert_allclose(ref.eff, a['binary_ale_levels'][j], rtol=1e-4, atol=1e-5)
                        checks.append(dict(fold=fold, model=name, seed=seed,
                            prediction_max_error=float(np.max(np.abs(actual-a['predicted']))),
                            pyale_feature_column=j,
                            pyale_max_error=float(np.max(np.abs(ref.eff.to_numpy()-a['binary_ale_levels'][j]))),
                            supported_snps=int(supported.sum()), unsupported_snps=int((~supported).sum()),
                            checkpoint_sha256=digest(md/('model.joblib' if name=='rf' else 'model.pt')),
                            evaluation_sha256=digest(md/'evaluation.npz')))
            print(f'{out.name}: checked fold {fold}/5', flush=True)
    summarize(out)
    new = json.loads(summary_path.read_text())
    indexed = {(r['model'],r['feature']):r for r in old}
    changed = []
    for r in new:
        previous = indexed[r['model'],r['feature']]['conditional_mean_binary_ale_contrast']
        current = r['conditional_mean_binary_ale_contrast']
        if current != previous:
            changed.append(dict(model=r['model'],feature=r['feature'], old_contrast=previous,
                new_contrast=current, selected_folds=r['folds_selected'], supported_folds=r['folds_with_both_test_alleles']))
    result = dict(dataset=out.name, checked_models=len(checks),
        formula_checked_all_columns=True, direct_PyALE_checks=len(checks),
        unsupported_model_fold_snp_rows=sum(c['unsupported_snps'] for c in checks),
        summary_rows=len(new), changed_summary_rows=len(changed),
        null_summary_rows=sum(r['conditional_mean_binary_ale_contrast'] is None for r in new),
        checks=checks, changes=changed, original_fingerprint=signature,
        original_training_sources_not_reexecuted=True,
        refresh_source_sha256=digest(Path(__file__)))
    write_json(REVIEW/f'{out.name}_validation.json', result)
    # A concise active result note; raw checkpoint contrasts are kept immutable.
    (out/'ALE_REVIEW.md').write_text(
        f'# ALE review (PyALE 1.2.0)\n\nChecked all 45 saved models and all saved contrast arrays. '
        f'Direct PyALE comparison: one highest-ranked supported SNP per model.\n\n'
        f'{result["unsupported_model_fold_snp_rows"]} monomorphic model/fold/SNP rows excluded '
        f'from empirical ALE aggregation; {len(changed)} summary rows changed; '
        f'{result["null_summary_rows"]} have no supported fold and now report null.\n\n'
        'Binary ALE equals the centered marginal PDP contrast on the same rows; it is '
        'not an LD-conditioned or causal effect. Predictions and GroupSHAP arrays are unchanged.\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', choices=['phase5','study34','fivefold','all'], default='all')
    args = parser.parse_args()
    REVIEW.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=4):
        if args.only in ['phase5','all']:
            print(phase5().to_string(index=False), flush=True)
        if args.only in ['study34','all']:
            print(study34().to_string(index=False), flush=True)
        if args.only in ['fivefold','all']:
            for name in ['study27_20260913_inductive','ft10_20260913_inductive']:
                fivefold(ROOT/'outputs/fivefold_v1'/name)
    write_json(REVIEW/f'run_{args.only}.json', dict(PyALE=importlib.metadata.version('PyALE'),
        numpy=np.__version__, pandas=pd.__version__, source_sha256=digest(Path(__file__))))


if __name__ == '__main__':
    main()
