"""Train RFs on selected Bloom 2013 phenotypes and save artifacts.

Per phenotype:
  - Load (X, y) via load_yeast_bloom
  - Apply MAF >= 0.05 filter
  - 5-fold CV R^2 (random_state=42)
  - Fit final RF on all data, save to outputs/yeast_bloom/<pheno>/model.joblib
  - Compute marginal QTL p-values via vectorized correlation OLS
    (no PCs — BYxRM is a 2-strain cross with no population structure)
  - Save: model.joblib, model_meta.json, marker_meta.parquet, marginal_pvals.parquet

Used by yeast_bloom_run_experiment.py.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from load_yeast_bloom import load_geno_2013, load_pheno_2013  # noqa: E402

OUT = REPO / "outputs" / "yeast_bloom"


def vectorized_marginal_pvalues(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-marker p-value from a univariate Pearson correlation t-test.

    For each column j: r_j = corr(X[:, j], y); t_j = r_j sqrt(n-2)/sqrt(1-r_j^2);
    p_j = 2 * (1 - F_t(|t_j|; n-2)). Uses scipy.stats.t.sf for numerical
    stability in the upper tail.
    """
    n = X.shape[0]
    Xc = X - X.mean(axis=0, dtype=np.float64)
    yc = y - y.mean(dtype=np.float64)
    sx = np.sqrt(np.maximum((Xc.astype(np.float64) ** 2).sum(axis=0), 1e-30))
    sy = np.sqrt(np.maximum((yc.astype(np.float64) ** 2).sum(), 1e-30))
    r = (Xc.astype(np.float64) * yc.reshape(-1, 1)).sum(axis=0) / (sx * sy)
    r = np.clip(r, -0.9999999, 0.9999999)
    t_stat = r * np.sqrt(n - 2) / np.sqrt(1 - r ** 2)
    pvals = 2.0 * scipy_stats.t.sf(np.abs(t_stat), df=n - 2)
    return pvals


def train_for_phenotype(phenotype: str, seed: int = 42) -> dict:
    out_dir = OUT / phenotype
    out_dir.mkdir(parents=True, exist_ok=True)

    geno_df, _, marker_meta_full = load_geno_2013(encoding="01")
    pheno = load_pheno_2013()
    if phenotype not in pheno.columns:
        raise KeyError(f"{phenotype!r} not in pheno columns")

    y_full = pheno[phenotype]
    keep = y_full.notna()
    common = geno_df.index[geno_df.index.isin(y_full.index[keep])]
    geno_sub = geno_df.loc[common]
    y = y_full.loc[common].to_numpy(dtype=np.float64)
    n_samples = len(y)

    # MAF filter
    X_arr = geno_sub.to_numpy(dtype=np.float32)
    maf = X_arr.mean(axis=0)
    maf = np.minimum(maf, 1 - maf)
    keep_marker_mask = maf >= 0.05
    X = X_arr[:, keep_marker_mask]
    marker_ids = list(geno_sub.columns[keep_marker_mask])
    marker_meta = marker_meta_full.iloc[keep_marker_mask.nonzero()[0]].reset_index(drop=True)

    # CV R^2
    t0 = time.time()
    rf_cv = RandomForestRegressor(
        n_estimators=200, max_features="sqrt", min_samples_leaf=2,
        random_state=seed, n_jobs=-1,
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    fold_r2 = cross_val_score(rf_cv, X, y, cv=cv, scoring="r2", n_jobs=1)
    cv_time = time.time() - t0

    # Fit final model on all data
    t1 = time.time()
    rf = RandomForestRegressor(
        n_estimators=200, max_features="sqrt", min_samples_leaf=2,
        random_state=seed, n_jobs=-1,
    )
    rf.fit(X, y)
    fit_time = time.time() - t1

    # Marginal p-values (no PCs — BYxRM is a 2-strain cross)
    t2 = time.time()
    pvals = vectorized_marginal_pvalues(X, y)
    pval_time = time.time() - t2

    # Save artifacts
    joblib.dump(rf, out_dir / "model.joblib")
    np.save(out_dir / "X_filtered.npy", X)
    np.save(out_dir / "y.npy", y)
    np.save(out_dir / "marker_idx_to_id.npy", np.array(marker_ids, dtype=object))
    pd.DataFrame({
        "marker_id": marker_ids,
        "chr": marker_meta["chr"].to_list(),
        "pos": marker_meta["pos"].to_list(),
        "marginal_pvalue": pvals,
        "neg_log10_p": -np.log10(np.maximum(pvals, 1e-300)),
        "filtered_index": np.arange(len(marker_ids)),
    }).to_parquet(out_dir / "marker_meta.parquet")

    meta = {
        "phenotype": phenotype,
        "n_samples": int(n_samples),
        "n_markers_pre_maf": int(X_arr.shape[1]),
        "n_markers_post_maf": int(X.shape[1]),
        "rf": {
            "n_estimators": 200,
            "max_features": "sqrt",
            "min_samples_leaf": 2,
            "random_state": seed,
        },
        "cv_fold_r2": [float(x) for x in fold_r2],
        "mean_cv_r2": float(fold_r2.mean()),
        "std_cv_r2": float(fold_r2.std(ddof=1)),
        "cv_seconds": cv_time,
        "fit_seconds": fit_time,
        "pval_seconds": pval_time,
        "min_marginal_p": float(pvals.min()),
        "n_markers_p_lt_1e-3": int((pvals < 1e-3).sum()),
        "n_markers_p_lt_1e-5": int((pvals < 1e-5).sum()),
    }
    (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    if len(sys.argv) > 1:
        phenotypes = sys.argv[1:]
    else:
        phenotypes = ["Neomycin", "Lactate"]
    for p in phenotypes:
        print(f"\n=== Training RF for {p} ===")
        train_for_phenotype(p)
