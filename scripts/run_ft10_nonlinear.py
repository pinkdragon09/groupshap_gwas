#!/usr/bin/env python3
"""
FT10 Nonlinear ML Pipeline — Grouped Shapley vs. GWAS.

Phenotype: FT10 (flowering time at 10°C), AraPheno Study 1, ~135 accessions.
Known causal loci: FLC (AT5G10140, Chr5:3,173,132) and FRI (AT4G00650, Chr4:270,148).

Pipeline:
  1. Load FT10 phenotype from AraPheno bulk zip.
  2. Align with 1001 Genomes HDF5 genotypes; add kinship-derived PCs.
  3. Load top-200 GWAS hits (pre-computed in outputs/full_baseline_gwas_ft10/).
  4. Extract SNP genotype matrix for those 200 positions.
  5. Train RandomForestRegressor (200 trees, 5-fold CV, random_state=42).
  6. Compute per-SNP TreeSHAP importance (mean |shap| across folds).
  7. Build gene-window grouping (Araport11 GFF, ±25 kb per gene).
  8. Compute grouped Shapley (MC, small eval set) at gene-window level.
  9. Run CPI at gene-window level.
 10. Save all outputs to outputs/ft10/.
 11. Check whether FLC / FRI appear in the feature panel or top rankings.
 12. Compute Spearman ρ between GWAS and RF-SHAP rankings.
 13. Write summary.md.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

try:
    import shap as shap_lib
except ImportError as exc:
    raise SystemExit("Missing dependency: shap  (pip install shap)") from exc

# We run from the repo root.
sys.path.insert(0, str(Path(__file__).parent))
from grouped_shap_utils import (
    assign_snps_to_gene_windows,
    conditional_permutation_importance,
    grouped_shapley_all,
    summarize_grouped_shapley,
)

# ─── Paths ────────────────────────────────────────────────────────────────────

PHENO_ZIP  = Path("data/arapheno/bulk/database.zip")
GENO_H5    = Path("data/1001genomes/genotypes/1001_SNP_MATRIX/imputed_snps_binary.hdf5")
KINSHIP_H5 = Path("data/1001genomes/genotypes/1001_SNP_MATRIX/kinship_ibs_mac5.hdf5")
GFF_GZ     = Path("data/reference/Araport11_GFF3_genes_transposons.Mar92021.gff.gz")
GWAS_HITS  = Path("outputs/full_baseline_gwas_ft10/study1_ft10_top_hits.tsv")
OUTDIR     = Path("outputs/ft10")

STUDY_VALUES_FILE = "1/study_1_values.csv"
PHENOTYPE_COL     = "FT10"
N_PCS             = 3
TOP_SNPS          = 200     # GWAS-prefiltered feature panel size
N_TREES           = 200
N_FOLDS           = 5
RANDOM_STATE      = 42
GENE_WINDOW_BP    = 25_000  # ±25 kb

# Known causal loci
FLC = {"gene_id": "AT5G10140", "chrom": "Chr5", "pos": 3_173_132, "symbol": "FLC"}
FRI = {"gene_id": "AT4G00650", "chrom": "Chr4", "pos":   270_148, "symbol": "FRI"}
LOCUS_WINDOW_BP = 500_000  # ±500 kb to call "near FLC/FRI"

# Grouped Shapley MC settings — kept small for tractability with 86 gene windows.
# At 86 groups each needing 2 × mc_samples value-function calls per eval point,
# thread-based parallelism is bottlenecked by pandas DataFrame construction (GIL).
# Serial execution (n_jobs=1) with fewer eval points is faster in practice.
MC_BG       = 5    # background rows for marginalisation
MC_EVAL     = 3    # eval rows (enough to estimate group importances)
MC_SAMPLES  = 15   # MC permutations per group

# ─── Helpers ──────────────────────────────────────────────────────────────────


def load_phenotype(pheno_zip: Path, study_values_file: str, phenotype_col: str) -> dict[str, float]:
    with zipfile.ZipFile(pheno_zip) as zf:
        rows = list(csv.DictReader(io.StringIO(zf.read(study_values_file).decode())))
    return {
        r["accession_id"].strip(): float(r[phenotype_col])
        for r in rows
        if r.get("accession_id", "").strip() and r.get(phenotype_col, "").strip()
    }


def compute_pcs(kinship_h5: Path, common_accs: list[str], n_pcs: int) -> np.ndarray:
    with h5py.File(kinship_h5, "r") as f:
        K = f["kinship"][:]
        kin_accs = [x.decode() if isinstance(x, bytes) else str(x) for x in f["accessions"][:]]
    idx = np.array([kin_accs.index(acc) for acc in common_accs])
    sub_k = K[np.ix_(idx, idx)]
    eigvals, eigvecs = np.linalg.eigh(sub_k)
    pcs = eigvecs[:, -n_pcs:]
    pcs = (pcs - pcs.mean(0)) / pcs.std(0)
    return pcs


def load_top_gwas_snps(gwas_hits: Path, top_k: int) -> pd.DataFrame:
    df = pd.read_csv(gwas_hits, sep="\t")
    df = df.sort_values("p_value").head(top_k).reset_index(drop=True)
    df["gwas_rank"] = df.index + 1
    return df


def extract_snp_matrix(
    geno_h5: Path,
    snp_indices: list[int],
    acc_ix: np.ndarray,
) -> np.ndarray:
    """Extract specific SNP rows for the given accession indices. Returns (n_samples, n_snps)."""
    with h5py.File(geno_h5, "r") as f:
        snps_ds = f["snps"]
        cols = []
        for si in snp_indices:
            col = snps_ds[si, acc_ix].astype(float)
            cols.append(col)
    return np.column_stack(cols)


def load_gene_db_from_gff(gff_gz: Path, chroms: set[str]) -> pd.DataFrame:
    """Parse Araport11 GFF3; return DataFrame(gene_id, symbol, chrom, start, end)."""
    records = []
    with gzip.open(gff_gz, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue
            chrom = parts[0]
            if chrom not in chroms:
                continue
            start = int(parts[3])
            end   = int(parts[4])
            attrs = parts[8]
            gid_m = re.search(r"ID=([^;]+)", attrs)
            if not gid_m:
                continue
            gene_id = gid_m.group(1).split(".")[0]
            sym_m = re.search(r"(?:symbol|Name)=([^;]+)", attrs)
            symbol = sym_m.group(1) if sym_m else gene_id
            records.append({"gene_id": gene_id, "symbol": symbol,
                            "chrom": chrom, "start": start, "end": end})
    return pd.DataFrame(records)


def check_locus_presence(
    snp_meta: pd.DataFrame,
    locus: dict,
    window_bp: int = LOCUS_WINDOW_BP,
) -> list[dict]:
    """Return SNPs in snp_meta within ±window_bp of a known locus."""
    mask = (
        (snp_meta["chromosome"] == locus["chrom"]) &
        (snp_meta["position"].between(locus["pos"] - window_bp, locus["pos"] + window_bp))
    )
    hits = snp_meta[mask].copy()
    hits["dist_to_locus"] = (hits["position"] - locus["pos"]).abs()
    return hits.to_dict("records")


def run_rf_cv(
    X_df: pd.DataFrame,
    y: np.ndarray,
    n_folds: int = N_FOLDS,
    n_trees: int = N_TREES,
    random_state: int = RANDOM_STATE,
) -> tuple[list[dict], pd.DataFrame]:
    """Train RF with K-fold CV. Returns (metrics_list, per-SNP SHAP importance)."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    all_metrics = []
    shap_importances = np.zeros(X_df.shape[1])

    for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(X_df), 1):
        X_tr, X_te = X_df.iloc[tr_idx], X_df.iloc[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        model = RandomForestRegressor(
            n_estimators=n_trees,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=random_state + fold_idx,
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        r2 = r2_score(y_te, pred)
        rmse = float(np.sqrt(np.mean((y_te - pred) ** 2)))
        all_metrics.append({"fold": fold_idx, "r2": round(r2, 4), "rmse": round(rmse, 3),
                             "n_train": len(tr_idx), "n_test": len(te_idx)})
        print(f"  Fold {fold_idx}: R²={r2:.4f}  RMSE={rmse:.2f}")

        explainer = shap_lib.TreeExplainer(model, X_tr.values)
        shap_vals = explainer.shap_values(X_te.values)
        shap_importances += np.abs(shap_vals).mean(axis=0)

    shap_importances /= n_folds
    shap_df = pd.DataFrame({
        "feature": X_df.columns,
        "mean_abs_shap": shap_importances,
        "shap_rank": pd.Series(shap_importances).rank(ascending=False).astype(int).values,
    }).sort_values("shap_rank").reset_index(drop=True)

    return all_metrics, shap_df


def run_grouped_shap(
    model_full: RandomForestRegressor,
    X_snp: pd.DataFrame,
    X_full: pd.DataFrame,
    y: np.ndarray,
    group_names: list[str],
    group_cols: list[list[str]],
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute grouped Shapley (MC) and CPI for the given grouping scheme.

    model_full was trained on X_full (SNPs + PCs).  Grouped Shapley is run on
    X_full so that PC covariates are always present and are marginalised over
    the background distribution (they are not assigned to any SNP group).
    CPI permutes only SNP group columns, leaving PCs intact.
    """
    rng = np.random.RandomState(random_state)
    bg_idx = rng.choice(len(X_full), size=min(MC_BG, len(X_full)), replace=False)
    ev_idx = rng.choice(len(X_full), size=min(MC_EVAL, len(X_full)), replace=False)
    X_bg = X_full.iloc[bg_idx]
    X_ev = X_full.iloc[ev_idx]

    print(f"  Grouped Shapley: {len(group_names)} groups, {len(X_ev)} eval pts, {MC_SAMPLES} MC samples …")
    phi_df = grouped_shapley_all(
        model_full, X_bg, X_ev, group_cols, group_names,
        mc_samples=MC_SAMPLES, n_jobs=1,  # serial: faster than threads due to pandas GIL overhead
    )
    summary_df = summarize_grouped_shapley(phi_df, group_names)

    print("  CPI …")
    y_series = pd.Series(y, name="phenotype")
    # CPI uses X_full so that PCs remain in all permuted designs;
    # only the SNP group columns are permuted.
    cpi_df = conditional_permutation_importance(
        model_full, X_full, y_series, group_cols, group_names, n_repeats=20, random_state=random_state
    )

    return summary_df, cpi_df


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ── 1. Phenotype ──────────────────────────────────────────────────────────
    print("Loading FT10 phenotype …")
    y_by_acc = load_phenotype(PHENO_ZIP, STUDY_VALUES_FILE, PHENOTYPE_COL)
    print(f"  {len(y_by_acc)} accessions with FT10 values")

    # ── 2. Genotype alignment ─────────────────────────────────────────────────
    print("Aligning with 1001 Genomes HDF5 …")
    with h5py.File(GENO_H5, "r") as f:
        geno_accs = [x.decode() if isinstance(x, bytes) else str(x) for x in f["accessions"][:]]
    common_accs = [acc for acc in geno_accs if acc in y_by_acc]
    geno_acc_idx = {acc: i for i, acc in enumerate(geno_accs)}
    acc_ix = np.array([geno_acc_idx[acc] for acc in common_accs], dtype=int)
    y = np.array([y_by_acc[acc] for acc in common_accs], dtype=float)
    print(f"  Matched {len(common_accs)} accessions  mean FT10={y.mean():.1f} sd={y.std():.1f}")

    # ── 3. Kinship PCs ────────────────────────────────────────────────────────
    print("Computing kinship PCs …")
    pcs = compute_pcs(KINSHIP_H5, common_accs, N_PCS)

    # ── 4. Top-200 GWAS SNPs ──────────────────────────────────────────────────
    print("Loading top-200 GWAS SNPs …")
    gwas_df = load_top_gwas_snps(GWAS_HITS, TOP_SNPS)
    print(f"  {len(gwas_df)} SNPs from {gwas_df['chromosome'].nunique()} chromosomes")

    # ── 5. Extract SNP genotype matrix ────────────────────────────────────────
    print("Extracting SNP genotype matrix …")
    snp_indices = gwas_df["snp_index"].tolist()
    G = extract_snp_matrix(GENO_H5, snp_indices, acc_ix)  # (n_samples, 200)

    # Build feature names: Chr_Pos
    snp_names = [
        f"{row['chromosome']}_{row['position']}"
        for _, row in gwas_df.iterrows()
    ]
    # Add PC covariates as extra columns (but we will keep them separate for Shapley)
    pc_names = [f"PC{i+1}" for i in range(N_PCS)]
    X_snp = pd.DataFrame(G, columns=snp_names)
    X_full = pd.concat([X_snp, pd.DataFrame(pcs, columns=pc_names)], axis=1)

    # SNP metadata for grouping
    snp_meta = gwas_df[["snp_index", "chromosome", "position", "neg_log10_p", "gwas_rank"]].copy()
    snp_meta["feature"] = snp_names
    snp_meta["feature_type"] = "snp"

    # ── 6. Check FLC / FRI in feature panel ───────────────────────────────────
    print("\n── FLC / FRI locus check ──────────────────────────────────────────")
    flc_hits = check_locus_presence(snp_meta, FLC)
    fri_hits = check_locus_presence(snp_meta, FRI)
    print(f"  SNPs ±{LOCUS_WINDOW_BP//1000}kb of FLC (Chr5:{FLC['pos']:,}): {len(flc_hits)}")
    print(f"  SNPs ±{LOCUS_WINDOW_BP//1000}kb of FRI (Chr4:{FRI['pos']:,}): {len(fri_hits)}")
    if not flc_hits and not fri_hits:
        print("  ** PREFILTER PARADOX: neither FLC nor FRI is represented in the top-200 GWAS panel **")

    # ── 7. RF + CV ────────────────────────────────────────────────────────────
    print("\nTraining RandomForest (5-fold CV) …")
    cv_metrics, shap_df = run_rf_cv(X_full, y)
    mean_r2  = float(np.mean([m["r2"]   for m in cv_metrics]))
    mean_rmse = float(np.mean([m["rmse"] for m in cv_metrics]))
    print(f"  Mean CV R²={mean_r2:.4f}  RMSE={mean_rmse:.2f}")

    # Add SHAP rank to GWAS df (SNPs only, exclude PCs)
    shap_snp = shap_df[shap_df["feature"].isin(snp_names)].copy()
    shap_snp["shap_rank"] = shap_snp["mean_abs_shap"].rank(ascending=False).astype(int)
    merged = gwas_df.merge(
        shap_snp[["feature", "mean_abs_shap", "shap_rank"]],
        left_on=pd.Index(snp_names)[gwas_df.index],
        right_on="feature",
        how="left",
    )
    # Simpler merge via snp name column
    snp_meta2 = snp_meta.copy()
    snp_meta2 = snp_meta2.merge(shap_snp[["feature", "mean_abs_shap", "shap_rank"]], on="feature", how="left")

    # Spearman rho: GWAS rank vs SHAP rank (among the 200 SNPs)
    valid = snp_meta2[["gwas_rank", "shap_rank"]].dropna()
    rho, pval_spearman = scipy_stats.spearmanr(valid["gwas_rank"], valid["shap_rank"])
    print(f"\n  Spearman ρ (GWAS rank vs RF-SHAP rank): {rho:.4f}  p={pval_spearman:.3e}")

    # ── 8. Gene-window grouping ────────────────────────────────────────────────
    print("\nBuilding gene-window grouping (Araport11 GFF, ±25 kb) …")
    all_chroms = set(snp_meta["chromosome"].unique())
    gene_df = load_gene_db_from_gff(GFF_GZ, all_chroms)
    print(f"  Loaded {len(gene_df)} genes on {gene_df['chrom'].nunique()} chromosomes")

    manifest = assign_snps_to_gene_windows(
        snp_meta[["feature", "chromosome", "position", "feature_type"]].copy(),
        gene_df,
        window_bp=GENE_WINDOW_BP,
        snp_chrom_col="chromosome",
        snp_pos_col="position",
        snp_feature_col="feature",
        gene_chrom_col="chrom",
        gene_start_col="start",
        gene_end_col="end",
        gene_id_col="gene_id",
    )
    print(f"  {len(manifest)} SNP→gene assignments  ({manifest['group'].nunique()} gene windows)")

    group_names_gw = sorted(manifest["group"].unique().tolist())
    group_cols_gw = [
        manifest.loc[manifest["group"] == g, "feature"].tolist()
        for g in group_names_gw
    ]
    # Filter to features present in X_snp (not PCs)
    feat_set = set(X_snp.columns)
    group_cols_gw = [[f for f in gc if f in feat_set] for gc in group_cols_gw]
    pairs = [(n, c) for n, c in zip(group_names_gw, group_cols_gw) if c]
    group_names_gw = [p[0] for p in pairs]
    group_cols_gw  = [p[1] for p in pairs]

    # ── 9. Full model for grouped Shapley ─────────────────────────────────────
    print("Training full RF on all data for grouped Shapley …")
    model_full = RandomForestRegressor(
        n_estimators=N_TREES, max_features="sqrt", min_samples_leaf=2,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    model_full.fit(X_full, y)
    full_r2 = r2_score(y, model_full.predict(X_full))
    print(f"  Full-data R² (in-sample): {full_r2:.4f}")

    # ── 10. Grouped Shapley + CPI ─────────────────────────────────────────────
    print(f"\nGrouped Shapley + CPI on {len(group_names_gw)} gene-window groups …")
    t_shap = time.time()
    grp_shap_summary, cpi_df = run_grouped_shap(
        model_full, X_snp, X_full, y, group_names_gw, group_cols_gw
    )
    print(f"  Done in {time.time()-t_shap:.0f}s")

    # Add CPI rank
    cpi_df["cpi_rank"] = cpi_df["mean_importance"].rank(ascending=False).astype(int)

    # ── 11. Check FLC / FRI in grouped results ────────────────────────────────
    flc_gene_id = FLC["gene_id"]
    fri_gene_id = FRI["gene_id"]
    flc_in_grp  = grp_shap_summary[grp_shap_summary["group"] == flc_gene_id]
    fri_in_grp  = grp_shap_summary[grp_shap_summary["group"] == fri_gene_id]
    flc_in_cpi  = cpi_df[cpi_df["group"] == flc_gene_id]
    fri_in_cpi  = cpi_df[cpi_df["group"] == fri_gene_id]

    # ── 12. Top-10 comparisons ────────────────────────────────────────────────
    top10_gwas  = snp_meta2.nsmallest(10, "gwas_rank")[
        ["feature", "chromosome", "position", "neg_log10_p", "gwas_rank", "mean_abs_shap", "shap_rank"]
    ].copy()
    top10_shap  = snp_meta2.nsmallest(10, "shap_rank")[
        ["feature", "chromosome", "position", "neg_log10_p", "gwas_rank", "mean_abs_shap", "shap_rank"]
    ].copy()
    top10_gw    = grp_shap_summary.head(10).reset_index(drop=True)
    top10_cpi   = cpi_df.head(10).reset_index(drop=True)

    # ── 13. Save outputs ──────────────────────────────────────────────────────
    print("\nSaving outputs …")

    # CV metrics
    pd.DataFrame(cv_metrics).to_csv(OUTDIR / "cv_metrics.tsv", sep="\t", index=False)

    # Per-SNP SHAP
    snp_meta2.to_csv(OUTDIR / "per_snp_shap.tsv", sep="\t", index=False)

    # Gene-window Shapley summary
    grp_shap_summary.to_csv(OUTDIR / "gene_window_shapley.tsv", sep="\t", index=False)

    # CPI
    cpi_df.to_csv(OUTDIR / "gene_window_cpi.tsv", sep="\t", index=False)

    # Gene-window manifest
    manifest.to_csv(OUTDIR / "gene_window_manifest.tsv", sep="\t", index=False)

    # Top-10 tables
    top10_gwas.to_csv(OUTDIR / "top10_gwas.tsv", sep="\t", index=False)
    top10_shap.to_csv(OUTDIR / "top10_shap.tsv", sep="\t", index=False)
    top10_gw.to_csv(OUTDIR / "top10_gene_window.tsv", sep="\t", index=False)
    top10_cpi.to_csv(OUTDIR / "top10_cpi.tsv", sep="\t", index=False)

    elapsed = round(time.time() - t0, 1)

    # ── 14. Summary ───────────────────────────────────────────────────────────
    def fmt_locus_check(hits: list[dict], locus: dict, window_kb: int) -> str:
        if not hits:
            return f"NOT in panel (no SNP within ±{window_kb}kb)"
        best = min(hits, key=lambda h: h["dist_to_locus"])
        return (f"PRESENT: nearest SNP {best['chromosome']}:{best['position']:,} "
                f"({best['dist_to_locus']//1000}kb away, GWAS rank={best['gwas_rank']})")

    flc_panel_str = fmt_locus_check(flc_hits, FLC, LOCUS_WINDOW_BP // 1000)
    fri_panel_str = fmt_locus_check(fri_hits, FRI, LOCUS_WINDOW_BP // 1000)

    def grp_rank(df: pd.DataFrame, gene_id: str, rank_col: str, val_col: str) -> str:
        row = df[df["group"] == gene_id]
        if row.empty:
            return "not in panel"
        r = int(row.iloc[0][rank_col]) if rank_col in row.columns else "?"
        v = float(row.iloc[0][val_col])
        return f"rank {r}, value {v:.4f}"

    summary_lines = [
        "# FT10 Nonlinear ML Pipeline — Summary",
        "",
        "## Experimental setup",
        f"- Phenotype: FT10 (AraPheno Study 1)",
        f"- N accessions matched: {len(common_accs)}",
        f"- Feature panel: top-{TOP_SNPS} GWAS SNPs (from {GWAS_HITS.name})",
        f"- Model: RandomForestRegressor ({N_TREES} trees, {N_FOLDS}-fold CV, random_state={RANDOM_STATE})",
        f"- Grouping: gene-window ±{GENE_WINDOW_BP//1000}kb (Araport11)",
        "",
        "## RF cross-validation performance",
        f"- Mean CV R²:  {mean_r2:.4f}",
        f"- Mean CV RMSE: {mean_rmse:.2f} days",
        f"- Full-data R² (in-sample): {full_r2:.4f}",
        "",
        "## CV metrics by fold",
    ]
    for m in cv_metrics:
        summary_lines.append(
            f"  - Fold {m['fold']}: R²={m['r2']:.4f}  RMSE={m['rmse']:.3f}"
            f"  (n_train={m['n_train']}, n_test={m['n_test']})"
        )

    summary_lines += [
        "",
        "## Known causal loci in feature panel",
        f"- FLC ({FLC['gene_id']}, {FLC['chrom']}:{FLC['pos']:,}): {flc_panel_str}",
        f"- FRI ({FRI['gene_id']}, {FRI['chrom']}:{FRI['pos']:,}): {fri_panel_str}",
        "",
        "### Interpretation",
    ]
    if not flc_hits and not fri_hits:
        summary_lines += [
            "**PREFILTER PARADOX detected.** Neither FLC nor FRI appears in the top-200 GWAS",
            "feature panel. The RF and grouped Shapley models cannot recover these canonical",
            "loci because the GWAS pre-filter excluded them entirely. This is a concrete",
            "empirical demonstration of the prefilter paradox:",
            "using GWAS p-values to select features for a nonlinear XAI analysis introduces",
            "a systematic bias that can exclude causally important loci while retaining",
            "statistically significant but biologically secondary hits.",
        ]
    else:
        summary_lines.append("FLC and/or FRI are present in the panel — see LD passenger analysis below.")

    summary_lines += [
        "",
        "## Top GWAS hits (linear, PC-adjusted)",
        "| Rank | SNP | Chrom | Position | −log₁₀p | SHAP rank |",
        "|------|-----|-------|----------|---------|-----------|",
    ]
    for _, row in top10_gwas.iterrows():
        summary_lines.append(
            f"| {int(row['gwas_rank'])} | {row['feature']} | {row['chromosome']} |"
            f" {int(row['position']):,} | {float(row['neg_log10_p']):.2f} |"
            f" {int(row['shap_rank']) if pd.notna(row.get('shap_rank')) else '—'} |"
        )

    summary_lines += [
        "",
        "## Top RF-SHAP hits",
        "| SHAP rank | SNP | Chrom | Position | mean|SHAP| | GWAS rank |",
        "|-----------|-----|-------|----------|------------|-----------|",
    ]
    for _, row in top10_shap.iterrows():
        summary_lines.append(
            f"| {int(row['shap_rank'])} | {row['feature']} | {row['chromosome']} |"
            f" {int(row['position']):,} | {float(row['mean_abs_shap']):.4f} |"
            f" {int(row['gwas_rank'])} |"
        )

    summary_lines += [
        "",
        f"## Spearman ρ (GWAS rank vs RF-SHAP rank, n={len(valid)} SNPs)",
        f"  ρ = {rho:.4f}   p = {pval_spearman:.3e}",
        "",
        "## Top gene-window grouped Shapley",
        "| Rank | Gene window | mean|φ| | sd|φ| |",
        "|------|-------------|---------|-------|",
    ]
    for i, row in top10_gw.iterrows():
        summary_lines.append(
            f"| {i+1} | {row['group']} | {row['mean_abs_phi']:.4f} | {row['sd_abs_phi']:.4f} |"
        )

    summary_lines += [
        "",
        "## Top gene-window CPI",
        "| Rank | Gene window | CPI mean | CPI sd |",
        "|------|-------------|----------|--------|",
    ]
    for _, row in top10_cpi.iterrows():
        summary_lines.append(
            f"| {int(row['cpi_rank'])} | {row['group']} | {row['mean_importance']:.4f} |"
            f" {row['sd_importance']:.4f} |"
        )

    summary_lines += [
        "",
        "## FLC/FRI in grouped Shapley ranking",
        f"- FLC ({FLC['gene_id']}): {grp_rank(grp_shap_summary, flc_gene_id, 'group', 'mean_abs_phi') if not flc_in_grp.empty else 'not in panel (GWAS prefilter excluded region)'}",
        f"- FRI ({FRI['gene_id']}): {grp_rank(grp_shap_summary, fri_gene_id, 'group', 'mean_abs_phi') if not fri_in_grp.empty else 'not in panel (GWAS prefilter excluded region)'}",
        "",
        "## FLC/FRI in CPI ranking",
        f"- FLC ({FLC['gene_id']}): {grp_rank(cpi_df, flc_gene_id, 'cpi_rank', 'mean_importance') if not flc_in_cpi.empty else 'not in panel (GWAS prefilter excluded region)'}",
        f"- FRI ({FRI['gene_id']}): {grp_rank(cpi_df, fri_gene_id, 'cpi_rank', 'mean_importance') if not fri_in_cpi.empty else 'not in panel (GWAS prefilter excluded region)'}",
        "",
        "## LD passenger analysis",
    ]

    # LD passenger: GWAS top-5 vs SHAP top-5 — where do ranks diverge?
    snp_meta_sorted = snp_meta2.sort_values("gwas_rank")
    top5_gwas_features = set(snp_meta_sorted.head(5)["feature"])
    top5_shap_features = set(snp_meta2.nsmallest(5, "shap_rank")["feature"])
    shared = top5_gwas_features & top5_shap_features
    gwas_only = top5_gwas_features - top5_shap_features
    shap_only = top5_shap_features - top5_gwas_features

    summary_lines += [
        f"Among the top-5 SNPs by GWAS p-value vs RF-SHAP importance:",
        f"- Shared (both top-5): {len(shared)} — {', '.join(sorted(shared)) or 'none'}",
        f"- GWAS top-5 only: {len(gwas_only)} — {', '.join(sorted(gwas_only)) or 'none'}",
        f"- SHAP top-5 only: {len(shap_only)} — {', '.join(sorted(shap_only)) or 'none'}",
        "",
        "Rank discordance (|GWAS rank − SHAP rank| > 50):",
    ]
    discordant = snp_meta2.dropna(subset=["gwas_rank", "shap_rank"]).copy()
    discordant["rank_diff"] = (discordant["gwas_rank"] - discordant["shap_rank"]).abs()
    discordant_top = discordant[discordant["rank_diff"] > 50].nlargest(5, "rank_diff")
    if discordant_top.empty:
        summary_lines.append("  None detected (all SNPs: |GWAS rank − SHAP rank| ≤ 50).")
    else:
        for _, row in discordant_top.iterrows():
            summary_lines.append(
                f"  {row['feature']} ({row['chromosome']}:{int(row['position']):,}): "
                f"GWAS rank {int(row['gwas_rank'])}, SHAP rank {int(row['shap_rank'])}, "
                f"diff {int(row['rank_diff'])}"
            )

    summary_lines += [
        "",
        "## Files",
        "| File | Description |",
        "|------|-------------|",
        "| cv_metrics.tsv | CV R²/RMSE per fold |",
        "| per_snp_shap.tsv | GWAS + SHAP ranks for all 200 SNPs |",
        "| gene_window_shapley.tsv | Grouped Shapley summary per gene window |",
        "| gene_window_cpi.tsv | CPI per gene window |",
        "| gene_window_manifest.tsv | SNP→gene window assignments |",
        "| top10_gwas.tsv / top10_shap.tsv | Top-10 by each method |",
        "| top10_gene_window.tsv / top10_cpi.tsv | Top-10 gene windows |",
        "",
        f"Total runtime: {elapsed:.0f}s",
    ]

    summary_text = "\n".join(summary_lines) + "\n"
    (OUTDIR / "summary.md").write_text(summary_text)
    print(summary_text)
    print(f"\n✓  All outputs written to {OUTDIR}/")
    print(f"   Total elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
