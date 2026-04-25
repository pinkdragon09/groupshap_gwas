#!/usr/bin/env python3
"""
Kinship-only RF baseline for FT10.

Computes the same 5-fold CV R² as ``run_ft10_nonlinear.py`` but with the
SNP layer stripped — features are the 3 kinship-derived PCs only. The
delta to the full-model R² (kinship-PCs + 200 GWAS SNPs) is the
"does the SNP layer add signal" question.

Pass criterion:
    full_model_R2 - kinship_only_R2 >= 0.10  → PASS
                                       >= 0.20  → STRICT-PASS

Usage:
    python3 scripts/ft10_kinship_only_baseline.py [--phenotype FT10]

Imports the helpers (``load_phenotype``, ``compute_pcs``) from
``run_ft10_nonlinear.py``; only the feature matrix differs.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent))
from run_ft10_nonlinear import (  # noqa: E402
    GENO_H5,
    KINSHIP_H5,
    N_FOLDS,
    N_PCS,
    N_TREES,
    PHENO_ZIP,
    RANDOM_STATE,
    STUDY_VALUES_FILE,
    compute_pcs,
    load_phenotype,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phenotype", default="FT10",
                    help="Phenotype column in study_1_values.csv (default: FT10)")
    args = ap.parse_args()

    t0 = time.time()
    print(f"Loading {args.phenotype} phenotype …")
    y_by_acc = load_phenotype(PHENO_ZIP, STUDY_VALUES_FILE, args.phenotype)
    print(f"  {len(y_by_acc)} accessions with {args.phenotype} values")

    print("Aligning with 1001 Genomes accessions …")
    with h5py.File(GENO_H5, "r") as f:
        geno_accs = [x.decode() if isinstance(x, bytes) else str(x)
                     for x in f["accessions"][:]]
    common_accs = [acc for acc in geno_accs if acc in y_by_acc]
    y = np.array([y_by_acc[acc] for acc in common_accs], dtype=float)
    print(f"  Matched {len(common_accs)} accessions  "
          f"mean={y.mean():.1f}  sd={y.std():.1f}")

    print(f"Computing {N_PCS} kinship PCs …")
    pcs = compute_pcs(KINSHIP_H5, common_accs, N_PCS)
    pc_names = [f"PC{i+1}" for i in range(N_PCS)]
    X = pd.DataFrame(pcs, columns=pc_names)

    print(f"\nTraining RandomForest on PCs ONLY ({N_FOLDS}-fold CV) …")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    metrics = []
    for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(X), 1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        model = RandomForestRegressor(
            n_estimators=N_TREES,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=RANDOM_STATE + fold_idx,
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        r2 = r2_score(y_te, pred)
        rmse = float(np.sqrt(np.mean((y_te - pred) ** 2)))
        metrics.append({"fold": fold_idx, "r2": r2, "rmse": rmse,
                        "n_train": len(tr_idx), "n_test": len(te_idx)})
        print(f"  Fold {fold_idx}: R²={r2:.4f}  RMSE={rmse:.2f}")

    mean_r2 = float(np.mean([m["r2"] for m in metrics]))
    mean_rmse = float(np.mean([m["rmse"] for m in metrics]))
    print(f"\n  Kinship-only mean CV R² = {mean_r2:.4f}")
    print(f"  Kinship-only mean CV RMSE = {mean_rmse:.2f}")
    print(f"\nElapsed: {time.time()-t0:.1f}s")

    # Persist alongside the FT10 RF outputs for direct comparison.
    out_dir = Path("outputs/ft10")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = out_dir / f"kinship_only_baseline_{args.phenotype.lower()}.tsv"
    pd.DataFrame(metrics).to_csv(out_tsv, sep="\t", index=False)
    print(f"  Wrote {out_tsv}")


if __name__ == "__main__":
    main()
