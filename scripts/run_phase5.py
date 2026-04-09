#!/usr/bin/env python3
"""
Phase 5: Conditional Permutation Importance (CPI) and ALE analysis.

Replicates fold-1 logic from random_forest_gwas_template.py inline to
obtain a fitted random-forest model, then computes:
  - CPI for chromosome groups
  - ALE for the top 3 SNPs by mean_abs_shap (from global_shap_ranking.tsv)

Outputs:
  outputs/phase5/cpi_results.tsv
  outputs/phase5/ale_results.tsv
"""

from __future__ import annotations

import csv
import io
import sys
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

# Ensure scripts/ is importable
SCRIPTS_DIR = Path(__file__).parent.resolve()
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from grouped_shap_utils import (
    ale_effect,
    build_group_cols_from_meta,
    conditional_permutation_importance,
)
from snp_selection import load_top_snp_indices

# ---------------------------------------------------------------------------
# Paths (same defaults as random_forest_gwas_template.py)
# ---------------------------------------------------------------------------
REPO_ROOT = SCRIPTS_DIR.parent
PHENO_ZIP = REPO_ROOT / "data/arapheno/bulk/database.zip"
GENO_H5 = REPO_ROOT / "data/1001genomes/genotypes/1001_SNP_MATRIX/imputed_snps_binary.hdf5"
KINSHIP_H5 = REPO_ROOT / "data/1001genomes/genotypes/1001_SNP_MATRIX/kinship_ibs_mac5.hdf5"
BASELINE_HITS = REPO_ROOT / "outputs/full_baseline_gwas/study27_mean_trl_top_hits.tsv"
SHAP_RANKING = REPO_ROOT / "outputs/random_forest_gwas_full/global_shap_ranking.tsv"
OUTDIR = REPO_ROOT / "outputs/phase5"

STUDY_VALUES_FILE = "27/study_27_values.csv"
PHENOTYPE_COL = "Mean(TRL) C"
TOP_SNPS = 500
N_PCS = 3
N_ESTIMATORS = 500
RANDOM_STATE = 42
N_REPEATS_CPI = 20
N_BINS_ALE = 20


# ---------------------------------------------------------------------------
# Data loading (inlined from random_forest_gwas_template.py)
# ---------------------------------------------------------------------------

def load_phenotype(pheno_zip: Path, study_values_file: str, phenotype_col: str) -> dict[str, float]:
    with zipfile.ZipFile(pheno_zip) as zf:
        rows = list(csv.DictReader(io.StringIO(zf.read(study_values_file).decode("utf-8"))))
    y_by_acc: dict[str, float] = {}
    for row in rows:
        acc = row["accession_id"].strip()
        value = row[phenotype_col].strip()
        if acc and value:
            y_by_acc[acc] = float(value)
    return y_by_acc


def compute_pcs(kinship_h5: Path, common_accs: list[str], n_pcs: int) -> np.ndarray:
    with h5py.File(kinship_h5, "r") as f:
        kinship = f["kinship"][:]
        kin_accs = [x.decode() if isinstance(x, bytes) else str(x) for x in f["accessions"][:]]
    kin_index = {acc: i for i, acc in enumerate(kin_accs)}
    ix = np.asarray([kin_index[acc] for acc in common_accs], dtype=int)
    sub_k = kinship[np.ix_(ix, ix)]
    eigvals, eigvecs = np.linalg.eigh(sub_k)
    pcs = eigvecs[:, -n_pcs:]
    pc_std = pcs.std(axis=0)
    pc_std[pc_std < 1e-8] = 1.0
    pcs = (pcs - pcs.mean(axis=0)) / pc_std
    return pcs


def chromosome_labels_and_bounds(h5f: h5py.File):
    labels = [f"Chr{x.decode() if isinstance(x, bytes) else x}" for x in h5f["positions"].attrs["chrs"]]
    bounds = h5f["positions"].attrs["chr_regions"]
    return labels, bounds


def chromosome_for_index(idx: int, labels: list[str], bounds) -> str:
    for chrom_idx, (start, stop) in enumerate(bounds):
        if int(start) <= idx < int(stop):
            return labels[chrom_idx]
    return "Chr?"


def load_arabidopsis_matrix():
    y_by_acc = load_phenotype(PHENO_ZIP, STUDY_VALUES_FILE, PHENOTYPE_COL)

    with h5py.File(GENO_H5, "r") as f:
        geno_accs = [x.decode() if isinstance(x, bytes) else str(x) for x in f["accessions"][:]]
        geno_acc_index = {acc: i for i, acc in enumerate(geno_accs)}
        common_accs = [acc for acc in geno_accs if acc in y_by_acc]
        geno_ix = np.asarray([geno_acc_index[acc] for acc in common_accs], dtype=int)
        snp_indices = load_top_snp_indices(BASELINE_HITS, TOP_SNPS)
        sorted_snp_indices = sorted(snp_indices)
        positions = f["positions"][sorted_snp_indices]
        snps = f["snps"][sorted_snp_indices][:, geno_ix].astype(float)
        labels, bounds = chromosome_labels_and_bounds(f)

    pcs = compute_pcs(KINSHIP_H5, common_accs, N_PCS)
    y = pd.Series([y_by_acc[acc] for acc in common_accs], name=PHENOTYPE_COL, dtype=float)

    snp_cols = [f"snp_{idx}" for idx in sorted_snp_indices]
    X_snp = pd.DataFrame(snps.T, columns=snp_cols)
    X_cov = pd.DataFrame(pcs, columns=[f"pc{i}" for i in range(1, N_PCS + 1)])
    X = pd.concat([X_snp, X_cov], axis=1)

    snp_meta = pd.DataFrame(
        {
            "feature": snp_cols,
            "snp_index": sorted_snp_indices,
            "chromosome": [chromosome_for_index(idx, labels, bounds) for idx in sorted_snp_indices],
            "position": positions.astype(int),
            "feature_type": "snp",
        }
    )
    covar_meta = pd.DataFrame(
        {
            "feature": X_cov.columns,
            "snp_index": np.nan,
            "chromosome": "",
            "position": np.nan,
            "feature_type": "covariate",
        }
    )
    feature_meta = pd.concat([snp_meta, covar_meta], ignore_index=True)
    return X, y, feature_meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("Loading Arabidopsis dataset ...")
    X, y, feature_meta = load_arabidopsis_matrix()
    print(f"  Accessions: {len(y)}, Features: {X.shape[1]}")

    # ── Fold 1 split (mirrors run_cv in template) ──────────────────────────
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold1_train_idx, fold1_test_idx = next(iter(kf.split(X)))
    X_train = X.iloc[fold1_train_idx].reset_index(drop=True)
    X_test  = X.iloc[fold1_test_idx].reset_index(drop=True)
    y_train = y.iloc[fold1_train_idx].reset_index(drop=True)
    y_test  = y.iloc[fold1_test_idx].reset_index(drop=True)

    print(f"  Fold 1 train: {len(X_train)}, test: {len(X_test)}")

    # ── Fit RF (fold 1) ────────────────────────────────────────────────────
    print("Fitting RandomForest on fold 1 ...")
    rf = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_features="sqrt",
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE + 1,  # fold_idx=1
    )
    rf.fit(X_train, y_train)

    from sklearn.metrics import r2_score
    r2_train = r2_score(y_train, rf.predict(X_train))
    r2_test  = r2_score(y_test,  rf.predict(X_test))
    print(f"  Train R²: {r2_train:.4f}  |  Test R² (fold 1): {r2_test:.4f}")

    # ── CPI: chromosome groups ─────────────────────────────────────────────
    print("\nComputing Conditional Permutation Importance (CPI) ...")
    group_names, group_cols_list = build_group_cols_from_meta(feature_meta, group_col="chromosome")
    print(f"  Groups: {group_names}")

    # Use full dataset for CPI (more stable with 81 samples)
    cpi_df = conditional_permutation_importance(
        model=rf,
        X=X,
        y=y,
        group_cols=group_cols_list,
        group_names=group_names,
        n_repeats=N_REPEATS_CPI,
        random_state=RANDOM_STATE,
    )
    print(cpi_df.to_string(index=False))
    cpi_path = OUTDIR / "cpi_results.tsv"
    cpi_df.to_csv(cpi_path, sep="\t", index=False)
    print(f"\n  Saved: {cpi_path}")

    # ── ALE: top 3 SNPs by mean_abs_shap ──────────────────────────────────
    print("\nComputing ALE for top 3 SNPs ...")
    shap_ranking = pd.read_csv(SHAP_RANKING, sep="\t")
    top_snps = (
        shap_ranking[shap_ranking["feature_type"] == "snp"]
        .head(3)["feature"]
        .tolist()
    )
    print(f"  Top SNPs: {top_snps}")

    ale_records = []
    for snp in top_snps:
        ale_df = ale_effect(model=rf, X=X, feature_col=snp, n_bins=N_BINS_ALE)
        ale_df.insert(0, "feature", snp)
        meta_row = feature_meta[feature_meta["feature"] == snp]
        if not meta_row.empty:
            ale_df["chromosome"] = meta_row.iloc[0]["chromosome"]
            ale_df["position"]   = int(meta_row.iloc[0]["position"])
        else:
            ale_df["chromosome"] = ""
            ale_df["position"]   = np.nan
        ale_records.append(ale_df)

    ale_combined = pd.concat(ale_records, ignore_index=True)
    ale_path = OUTDIR / "ale_results.tsv"
    ale_combined.to_csv(ale_path, sep="\t", index=False)
    print(f"  Saved: {ale_path}")
    print("\nALE summary (per feature):")
    for snp in top_snps:
        sub = ale_combined[ale_combined["feature"] == snp]
        ale_range = sub["ale"].max() - sub["ale"].min()
        chrom = sub["chromosome"].iloc[0] if len(sub) > 0 else "?"
        pos   = int(sub["position"].iloc[0]) if len(sub) > 0 else 0
        ale_min = sub["ale"].min()
        ale_max = sub["ale"].max()
        print(f"  {snp} ({chrom}:{pos})  ALE range: {ale_range:.4f}  [{ale_min:.4f}, {ale_max:.4f}]")

    print("\nPhase 5 complete.")


if __name__ == "__main__":
    main()
