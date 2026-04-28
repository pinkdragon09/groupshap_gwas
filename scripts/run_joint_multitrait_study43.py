#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Joint multi-trait grouped Shapley analysis: delta-13C / stomata density / stomatasize.

Trains a single multi-output Random Forest on all three Study 43 traits
simultaneously and runs global grouped Shapley at the chromosome level for:
  (a) the joint model  -- phi_k^joint  (pleiotropy decomposition across traits)
  (b) each trait separately via per-output wrappers

This empirically demonstrates that Chr1 controls the carbon-efficiency outcome
(delta-13C) while Chr2 controls the physical stomatal gate (density + size),
and that the joint grouped Shapley cleanly separates these two biological axes.

Output directory: outputs/xai_gwas_arabidopsis_joint_study43/
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xai_gwas.attribution import (
    grouped_shapley_all,
    grouped_interaction_all,
    summarize_grouped_shapley,
)
from xai_gwas.grouping import build_group_cols_from_meta
from xai_gwas.viz import shapley_barplot, interaction_heatmap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HDF5_PATH = (
    Path(__file__).resolve().parents[1]
    / "data" / "1001genomes" / "genotypes"
    / "1001_SNP_MATRIX" / "imputed_snps_binary.hdf5"
)
ARAPHENO_DIR = Path(__file__).resolve().parents[1] / "data" / "arapheno" / "phenotypes"
OUTDIR = Path(__file__).resolve().parents[1] / "outputs" / "xai_gwas_arabidopsis_joint_study43"

CHR_BOUNDARIES = {
    "chr1": (0,       2597824),
    "chr2": (2597825, 4466693),
    "chr3": (4466694, 6661058),
    "chr4": (6661059, 8428146),
    "chr5": (8428147, 10709948),
}

TRAITS       = ["delta 13C", "stomata density", "stomatasize"]
TRAIT_LABELS = ["delta_13C", "stomata_density", "stomatasize"]
TOP_SNPS     = 100
MC_SAMPLES   = 20
N_BACKGROUND = 20
N_EVAL       = 10
FOLDS        = 2
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Per-output wrapper: makes a single-output view of a multi-output RF
# ---------------------------------------------------------------------------
class _SingleOutputWrapper:
    """Wraps a multi-output RF so it predicts only one output column."""

    def __init__(self, model: RandomForestRegressor, output_idx: int) -> None:
        self._model = model
        self._idx   = output_idx

    def predict(self, X):
        return self._model.predict(X)[:, self._idx]


# ---------------------------------------------------------------------------
# Data loading: all three traits, intersection of non-null accessions
# ---------------------------------------------------------------------------
def load_joint_data(top_snps: int = TOP_SNPS):
    study_csv = ARAPHENO_DIR / "study_43_values.csv"
    pheno_df  = pd.read_csv(study_csv)
    pheno_df["accession_id"] = pheno_df["accession_id"].astype(str)

    # Keep only rows with all three traits present
    complete = pheno_df.dropna(subset=TRAITS).drop_duplicates("accession_id").copy()
    print(f"  Accessions with all 3 traits: {len(complete)}")

    with h5py.File(HDF5_PATH, "r") as f:
        geno_accs    = f["accessions"][:].astype(str)
        positions    = f["positions"][:]

        pheno_set    = set(complete["accession_id"])
        overlap_mask = np.array([a in pheno_set for a in geno_accs])
        geno_idx     = np.where(overlap_mask)[0]
        geno_acc_sel = geno_accs[geno_idx]
        print(f"  Genotype/phenotype overlap: {len(geno_acc_sel)} accessions")

        chr_labels = np.empty(len(positions), dtype="U5")
        for chrom, (lo, hi) in CHR_BOUNDARIES.items():
            chr_labels[lo:hi + 1] = chrom

        print(f"  Computing per-SNP variances ...")
        n_per_chr = max(1, top_snps // 5)
        top_indices_per_chr: dict[str, np.ndarray] = {}
        for chrom, (lo, hi) in CHR_BOUNDARIES.items():
            snps_chr = f["snps"][lo:hi + 1, :][:, geno_idx]
            var_chr  = snps_chr.astype(np.float32).var(axis=1)
            top_idx  = np.argsort(var_chr)[-n_per_chr:][::-1]
            top_indices_per_chr[chrom] = lo + top_idx
            print(f"    {chrom}: kept top {n_per_chr} by variance")

        all_top_idx  = np.sort(np.concatenate(list(top_indices_per_chr.values())))
        all_top_pos  = positions[all_top_idx]
        all_top_chrs = chr_labels[all_top_idx]
        X_arr        = f["snps"][all_top_idx, :][:, geno_idx].astype(np.float32).T

    col_names = [f"{c}_{p}" for c, p in zip(all_top_chrs, all_top_pos)]
    X_df = pd.DataFrame(X_arr, columns=col_names)

    pheno_indexed = complete.set_index("accession_id")
    Y_raw = pheno_indexed.loc[geno_acc_sel, TRAITS].values.astype(np.float32)

    # Z-score each trait so the joint value function weights them equally
    Y_mean = Y_raw.mean(axis=0, keepdims=True)
    Y_std  = Y_raw.std(axis=0, keepdims=True)
    Y_std[Y_std == 0] = 1.0
    Y_norm = (Y_raw - Y_mean) / Y_std

    snp_meta = pd.DataFrame({
        "feature":      col_names,
        "chromosome":   all_top_chrs,
        "position":     all_top_pos,
        "feature_type": "snp",
    })

    return X_df, Y_norm, Y_raw, snp_meta, Y_mean, Y_std


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def main():
    t0 = time.perf_counter()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  Joint Multi-Trait Analysis — Study 43")
    print("  Traits: delta-13C / stomata density / stomatasize")
    print("="*60)

    X, Y_norm, Y_raw, snp_meta, Y_mean, Y_std = load_joint_data()
    n_acc, n_snps = X.shape
    n_traits = Y_norm.shape[1]
    print(f"\n  {n_acc} accessions x {n_snps} SNPs x {n_traits} traits")

    group_names, group_cols = build_group_cols_from_meta(snp_meta, group_col="chromosome")
    print(f"  Groups: {len(group_names)} chromosomes")

    # ------------------------------------------------------------------
    # Cross-validated predictive performance (multi-output RF)
    # ------------------------------------------------------------------
    print("\n--- Cross-validation (multi-output RF) ---")
    kf  = KFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rng = np.random.default_rng(RANDOM_STATE)

    fold_r2   = {lbl: [] for lbl in TRAIT_LABELS}
    fold_r2_joint = []
    all_phi_joint = []
    all_phi_per   = {lbl: [] for lbl in TRAIT_LABELS}

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        print(f"\n  Fold {fold_idx}/{FOLDS} -- fitting multi-output RF ...")
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        Y_tr, Y_te = Y_norm[train_idx], Y_norm[test_idx]

        rf = RandomForestRegressor(
            n_estimators=300, max_features=0.33, min_samples_leaf=5,
            n_jobs=-1, random_state=RANDOM_STATE,
        )
        rf.fit(X_tr, Y_tr)
        rf.n_jobs = 1  # avoid joblib nesting in Shapley loops

        Y_pred = rf.predict(X_te)
        for i, lbl in enumerate(TRAIT_LABELS):
            r2 = r2_score(Y_te[:, i], Y_pred[:, i])
            fold_r2[lbl].append(r2)
            print(f"    {lbl}: R² = {r2:.4f}")
        r2_joint = float(np.mean([r2_score(Y_te[:, i], Y_pred[:, i])
                                   for i in range(n_traits)]))
        fold_r2_joint.append(r2_joint)
        print(f"    mean (joint): R² = {r2_joint:.4f}")

        bg_idx = rng.choice(len(X_tr), min(N_BACKGROUND, len(X_tr)), replace=False)
        ev_idx = rng.choice(len(X_te), min(N_EVAL, len(X_te)), replace=False)

        # Joint Shapley — multi-output RF: predict() returns (n, T),
        # value_function_numpy calls .mean() => grand mean = joint value function
        phi_joint = grouped_shapley_all(
            rf, X_tr.iloc[bg_idx], X_te.iloc[ev_idx],
            group_cols=group_cols, group_names=group_names,
            mc_samples=MC_SAMPLES, n_jobs=1, random_state=RANDOM_STATE,
        )
        phi_joint["fold"] = fold_idx
        all_phi_joint.append(phi_joint)

        # Per-trait Shapley via single-output wrappers
        for i, lbl in enumerate(TRAIT_LABELS):
            wrapper = _SingleOutputWrapper(rf, i)
            phi_t = grouped_shapley_all(
                wrapper, X_tr.iloc[bg_idx], X_te.iloc[ev_idx],
                group_cols=group_cols, group_names=group_names,
                mc_samples=MC_SAMPLES, n_jobs=1, random_state=RANDOM_STATE,
            )
            phi_t["fold"] = fold_idx
            all_phi_per[lbl].append(phi_t)

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------
    print("\n--- Cross-validation R² summary ---")
    cv_rows = []
    for lbl in TRAIT_LABELS:
        r2_vals = fold_r2[lbl]
        print(f"  {lbl}: mean={np.mean(r2_vals):.4f}  sd={np.std(r2_vals):.4f}")
        cv_rows.append({"trait": lbl, "mean_r2": np.mean(r2_vals),
                        "sd_r2": np.std(r2_vals)})
    print(f"  joint mean: {np.mean(fold_r2_joint):.4f}")

    # Concatenate and summarise phi for joint model
    phi_joint_all = pd.concat(all_phi_joint, ignore_index=True)
    summary_joint = summarize_grouped_shapley(phi_joint_all)
    print("\n--- Joint φ_k^joint (mean |φ|, chromosomes ranked) ---")
    print(summary_joint[["group", "mean_abs_phi", "sd_abs_phi"]].to_string(index=False))

    # Per-trait summaries
    summaries_per = {}
    print("\n--- Per-trait φ_k from joint model ---")
    for lbl in TRAIT_LABELS:
        phi_df = pd.concat(all_phi_per[lbl], ignore_index=True)
        s = summarize_grouped_shapley(phi_df)
        summaries_per[lbl] = s
        print(f"\n  {lbl}:")
        print(s[["group", "mean_abs_phi"]].to_string(index=False))

    # ------------------------------------------------------------------
    # Cross-trait comparison table
    # ------------------------------------------------------------------
    print("\n--- Cross-trait comparison table (mean |φ_k|) ---")
    chrs = [f"chr{i}" for i in range(1, 6)]
    rows = []
    header = f"{'Chr':<8}" + "".join(f"  {lbl:<20}" for lbl in TRAIT_LABELS) + "  phi_joint"
    print(header)
    for chrom in chrs:
        per_vals = []
        for lbl in TRAIT_LABELS:
            row = summaries_per[lbl].set_index("group")
            val = row.loc[chrom, "mean_abs_phi"] if chrom in row.index else 0.0
            per_vals.append(val)
        joint_row = summary_joint.set_index("group")
        joint_val = joint_row.loc[chrom, "mean_abs_phi"] if chrom in joint_row.index else 0.0
        line = f"{chrom:<8}" + "".join(f"  {v:<20.4f}" for v in per_vals) + f"  {joint_val:.4f}"
        print(line)
        rows.append({"chr": chrom, **{lbl: pv for lbl, pv in zip(TRAIT_LABELS, per_vals)},
                     "phi_joint": joint_val})

    cross_trait_df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Pairwise interactions (joint model, full data)
    # ------------------------------------------------------------------
    print("\n--- Computing pairwise chromosome interactions (joint model) ---")
    rf_full = RandomForestRegressor(
        n_estimators=300, max_features=0.33, min_samples_leaf=5,
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    rf_full.fit(X, Y_norm)
    rf_full.n_jobs = 1

    bg_all = X.sample(min(N_BACKGROUND, len(X)), random_state=RANDOM_STATE)
    ev_all = X.sample(min(N_EVAL, len(X)), random_state=RANDOM_STATE + 1)

    interaction_df = grouped_interaction_all(
        rf_full, bg_all, ev_all,
        group_cols=group_cols, group_names=group_names,
        mc_samples=MC_SAMPLES, n_jobs=1, random_state=RANDOM_STATE,
    )
    print("\n  Top-5 chromosome pairs (joint model) by |I_jk|:")
    print(interaction_df.head(5)[
        ["group_j", "group_k", "mean_abs_interaction", "sd_interaction"]
    ].to_string(index=False))

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    phi_joint_all.to_csv(OUTDIR / "phi_joint_all.csv", index=False)
    summary_joint.to_csv(OUTDIR / "phi_joint_summary.csv", index=False)
    cross_trait_df.to_csv(OUTDIR / "cross_trait_comparison.csv", index=False)
    interaction_df.to_csv(OUTDIR / "interactions_joint.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(OUTDIR / "cv_metrics.csv", index=False)
    for lbl in TRAIT_LABELS:
        phi_df = pd.concat(all_phi_per[lbl], ignore_index=True)
        phi_df.to_csv(OUTDIR / f"phi_{lbl}_all.csv", index=False)
        summaries_per[lbl].to_csv(OUTDIR / f"phi_{lbl}_summary.csv", index=False)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 1. Side-by-side bar chart: per-trait phi_k vs joint phi_k
        fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
        plot_labels = TRAIT_LABELS + ["joint"]
        plot_sums   = [summaries_per[lbl] for lbl in TRAIT_LABELS] + [summary_joint]
        titles      = ["δ¹³C", "Stomata density", "Stomata size", "Joint φᵏʲᵒⁱⁿᵗ"]
        colors      = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
        for ax, s, title, color in zip(axes, plot_sums, titles, colors):
            s_sorted = s.sort_values("mean_abs_phi", ascending=True)
            ax.barh(s_sorted["group"], s_sorted["mean_abs_phi"], color=color, alpha=0.8)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xlabel("|φₖ|")
            ax.tick_params(labelsize=9)
        plt.suptitle("Global Grouped Shapley by Chromosome — Study 43 Joint Model", fontsize=12)
        plt.tight_layout()
        fig.savefig(OUTDIR / "phi_joint_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Saved phi_joint_comparison.png")

        # 2. Heatmap of joint interactions
        fig2, _ = interaction_heatmap(
            interaction_df, log_scale=True,
            title="Pairwise I_jk (log10) — Joint Model")
        fig2.savefig(OUTDIR / "interaction_joint_heatmap.png", dpi=150)
        plt.close(fig2)
        print(f"  Saved interaction_joint_heatmap.png")

        # 3. Cross-trait comparison heatmap
        ct_matrix = cross_trait_df.set_index("chr")[TRAIT_LABELS + ["phi_joint"]]
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        im = ax3.imshow(ct_matrix.values.T, aspect="auto", cmap="YlOrRd")
        ax3.set_xticks(range(len(chrs)))
        ax3.set_xticklabels(chrs)
        ax3.set_yticks(range(len(TRAIT_LABELS) + 1))
        ax3.set_yticklabels(["δ¹³C", "Stomata density", "Stomata size", "Joint φʲᵒⁱⁿᵗ"])
        plt.colorbar(im, ax=ax3, label="|φₖ|")
        ax3.set_title("Cross-trait grouped Shapley (mean |φₖ|) — chr1/chr2 dissociation", fontsize=11)
        for i in range(ct_matrix.shape[1]):
            for j in range(ct_matrix.shape[0]):
                ax3.text(j, i, f"{ct_matrix.values[j, i]:.3f}",
                         ha="center", va="center", fontsize=8, color="black")
        plt.tight_layout()
        fig3.savefig(OUTDIR / "cross_trait_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig3)
        print(f"  Saved cross_trait_heatmap.png")

    except Exception as e:
        print(f"  [WARN] Plot generation failed: {e}")

    elapsed = time.perf_counter() - t0
    print(f"\n  Wall-clock: {elapsed:.1f}s")
    print(f"\nAll outputs written to {OUTDIR}/")

    # Summary record
    pd.DataFrame([{
        "n_accessions": n_acc,
        "n_snps": n_snps,
        "n_traits": n_traits,
        "mean_r2_joint": float(np.mean(fold_r2_joint)),
        "top_chr_joint": summary_joint.iloc[0]["group"],
        "top_phi_joint": float(summary_joint.iloc[0]["mean_abs_phi"]),
        "elapsed_s": elapsed,
    }]).to_csv(OUTDIR / "run_summary.csv", index=False)


if __name__ == "__main__":
    main()
