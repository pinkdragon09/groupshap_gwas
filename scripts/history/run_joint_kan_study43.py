#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Joint multi-trait grouped Shapley (KAN backbone) — Study 43.

Extends the joint RF analysis (run_joint_multitrait_study43.py) to a KAN
backbone.  Three separate 2-layer KANs are trained (one per trait: δ¹³C /
stomata density / stomatasize).  Grouped Shapley at the chromosome level is
then computed:

  (a) per-trait  — via each single-output KAN wrapped as predict(X)
  (b) joint      — via a JointKanWrapper that returns (n, 3) predictions;
                   the mean over all samples × traits is the joint value
                   function, identical to what the multi-output RF does.

This empirically extends the KAN curvature-interaction result from yeast (Bloom panel) to Arabidopsis
climate phenotypes: KAN grouped interactions at the chromosome level should be
measurably stronger than the RF baseline due to KAN's B-spline edge curvature.

Output directory: outputs/xai_gwas_arabidopsis_joint_kan_study43/
Comparison CSV against RF:  <outdir>/comparison_kan_vs_rf.csv

Requires: .venv-kan activated (pykan 0.2.8 + torch) — run as:
  .venv-kan/bin/python scripts/run_joint_kan_study43.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xai_gwas.attribution import (
    grouped_shapley_all,
    grouped_interaction_all,
    summarize_grouped_shapley,
)
from xai_gwas.grouping import build_group_cols_from_meta
from kan import KAN  # noqa: E402  # requires .venv-kan

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HDF5_PATH = (
    ROOT / "data" / "1001genomes" / "genotypes"
    / "1001_SNP_MATRIX" / "imputed_snps_binary.hdf5"
)
ARAPHENO_DIR = ROOT / "data" / "arapheno" / "phenotypes"
OUTDIR = ROOT / "outputs" / "xai_gwas_arabidopsis_joint_kan_study43"
RF_OUTDIR = ROOT / "outputs" / "xai_gwas_arabidopsis_joint_study43"
CKPT_BASE = OUTDIR / "_pykan_ckpt"

CHR_BOUNDARIES = {
    "chr1": (0,       2597824),
    "chr2": (2597825, 4466693),
    "chr3": (4466694, 6661058),
    "chr4": (6661059, 8428146),
    "chr5": (8428147, 10709948),
}

TRAITS       = ["delta 13C", "stomata density", "stomatasize"]
TRAIT_LABELS = ["delta_13C", "stomata_density", "stomatasize"]

# Matching the joint RF baseline params exactly for a fair comparison
TOP_SNPS     = 100
MC_SAMPLES   = 20
N_BACKGROUND = 20
N_EVAL       = 10
FOLDS        = 2
RANDOM_STATE = 42

# KAN architecture (production Bloom config)
KAN_HIDDEN   = 8
KAN_GRID     = 5
KAN_K        = 3
KAN_EPOCHS   = 200
KAN_PATIENCE = 30
KAN_LR       = 1e-3
KAN_BATCH    = 64


# ---------------------------------------------------------------------------
# KAN predict shim — numpy .predict(X) interface around a torch KAN
# ---------------------------------------------------------------------------
class _KanShim:
    def __init__(self, model: KAN) -> None:
        self.model = model

    def predict(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        Xt = torch.as_tensor(X, dtype=torch.float32)
        with torch.no_grad():
            out = self.model(Xt).cpu().numpy()
        return out.ravel() if out.ndim > 1 else out


class _JointKanWrapper:
    """Wraps three single-output KANs into a multi-output predict().

    Returns (n, 3) predictions so that .mean() gives the grand mean over
    all samples and traits — identical to what a multi-output RF returns.
    """

    def __init__(self, kans: list[KAN]) -> None:
        self._kans = kans

    def predict(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        Xt = torch.as_tensor(X, dtype=torch.float32)
        cols = []
        for kan in self._kans:
            with torch.no_grad():
                col = kan(Xt).cpu().numpy().ravel()
            cols.append(col)
        return np.column_stack(cols)  # (n, 3)


# ---------------------------------------------------------------------------
# Data loading — identical to the RF baseline script
# ---------------------------------------------------------------------------
def load_joint_data(top_snps: int = TOP_SNPS):
    study_csv = ARAPHENO_DIR / "study_43_values.csv"
    pheno_df  = pd.read_csv(study_csv)
    pheno_df["accession_id"] = pheno_df["accession_id"].astype(str)

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

        print("  Computing per-SNP variances ...")
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
# KAN training — single-output, Adam + MSE + early stopping on val MSE
# ---------------------------------------------------------------------------
def _make_kan(p_in: int, seed: int, ckpt_dir: Path) -> KAN:
    return KAN(
        width=[p_in, KAN_HIDDEN, 1], grid=KAN_GRID, k=KAN_K,
        seed=seed, ckpt_path=str(ckpt_dir), auto_save=False,
    )


def _train_kan(
    model: KAN,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    Xt  = torch.as_tensor(X_train, dtype=torch.float32)
    yt  = torch.as_tensor(y_train, dtype=torch.float32).unsqueeze(1)
    Xv  = torch.as_tensor(X_val,   dtype=torch.float32)
    yv  = torch.as_tensor(y_val,   dtype=torch.float32).unsqueeze(1)

    opt = torch.optim.Adam(model.parameters(), lr=KAN_LR)
    n = Xt.shape[0]
    n_batches = max(1, n // KAN_BATCH)
    rng = np.random.default_rng(seed)

    best_val   = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(KAN_EPOCHS):
        model.train()
        perm = rng.permutation(n)
        for b in range(n_batches):
            idx = perm[b * KAN_BATCH:(b + 1) * KAN_BATCH]
            opt.zero_grad()
            loss = ((model(Xt[idx]) - yt[idx]) ** 2).mean()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_mse = float(((model(Xv) - yv) ** 2).mean())
        if val_mse < best_val - 1e-6:
            best_val = val_mse
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= KAN_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()


def _train_three_kans(
    X_tr: np.ndarray, Y_tr: np.ndarray,
    X_val: np.ndarray, Y_val: np.ndarray,
    p_in: int, seed: int, ckpt_dir: Path,
) -> list[KAN]:
    """Train one KAN per trait; return list of 3 trained KAN models."""
    kans = []
    for i, lbl in enumerate(TRAIT_LABELS):
        ckpt_i = ckpt_dir / lbl
        ckpt_i.mkdir(parents=True, exist_ok=True)
        kan = _make_kan(p_in, seed + i, ckpt_i)
        _train_kan(kan, X_tr, Y_tr[:, i], X_val, Y_val[:, i], seed=seed + i)
        kans.append(kan)
        kan_r2 = float(r2_score(
            Y_val[:, i],
            kan(torch.as_tensor(X_val, dtype=torch.float32)).detach().numpy().ravel()
        ))
        print(f"    {lbl}: val R² = {kan_r2:.4f}")
    return kans


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.perf_counter()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    CKPT_BASE.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  Joint Multi-Trait KAN Analysis — Study 43")
    print("  Traits: delta-13C / stomata density / stomatasize")
    print("=" * 60)

    X, Y_norm, Y_raw, snp_meta, Y_mean, Y_std = load_joint_data()
    n_acc, n_snps = X.shape
    n_traits = Y_norm.shape[1]
    p_in = n_snps
    print(f"\n  {n_acc} accessions x {n_snps} SNPs x {n_traits} traits")

    X_arr = X.to_numpy().astype(np.float32)

    group_names, group_cols = build_group_cols_from_meta(snp_meta, group_col="chromosome")
    print(f"  Groups: {len(group_names)} chromosomes")

    # ------------------------------------------------------------------
    # Cross-validated predictive performance (3 KANs per fold)
    # ------------------------------------------------------------------
    print("\n--- Cross-validation (3 separate KANs, one per trait) ---")
    kf  = KFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rng = np.random.default_rng(RANDOM_STATE)

    fold_r2          = {lbl: [] for lbl in TRAIT_LABELS}
    fold_r2_joint    = []
    all_phi_joint    = []
    all_phi_per      = {lbl: [] for lbl in TRAIT_LABELS}

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_arr), 1):
        print(f"\n  Fold {fold_idx}/{FOLDS} -- training 3 KANs ...")
        X_tr_full = X_arr[train_idx]
        Y_tr_full = Y_norm[train_idx]
        X_te      = X_arr[test_idx]
        Y_te      = Y_norm[test_idx]

        # 10% of train as early-stopping val
        n_tr = len(X_tr_full)
        perm = rng.permutation(n_tr)
        n_val = max(1, n_tr // 10)
        X_val_f = X_tr_full[perm[:n_val]]
        Y_val_f = Y_tr_full[perm[:n_val]]
        X_tr_f  = X_tr_full[perm[n_val:]]
        Y_tr_f  = Y_tr_full[perm[n_val:]]

        ckpt_fold = CKPT_BASE / f"fold{fold_idx}"
        kans = _train_three_kans(
            X_tr_f, Y_tr_f, X_val_f, Y_val_f,
            p_in=p_in, seed=RANDOM_STATE + fold_idx * 10,
            ckpt_dir=ckpt_fold,
        )

        # Per-trait test R²
        r2_fold_vals = []
        for i, lbl in enumerate(TRAIT_LABELS):
            with torch.no_grad():
                yhat = kans[i](torch.as_tensor(X_te, dtype=torch.float32)).numpy().ravel()
            r2 = float(r2_score(Y_te[:, i], yhat))
            fold_r2[lbl].append(r2)
            r2_fold_vals.append(r2)
            print(f"    {lbl}: test R² = {r2:.4f}")
        r2_joint = float(np.mean(r2_fold_vals))
        fold_r2_joint.append(r2_joint)
        print(f"    mean (joint): R² = {r2_joint:.4f}")

        # Sample background + eval indices from fold splits
        bg_idx  = rng.choice(len(X_tr_f), min(N_BACKGROUND, len(X_tr_f)), replace=False)
        ev_idx  = rng.choice(len(X_te),   min(N_EVAL,       len(X_te)),   replace=False)
        X_bg_df = pd.DataFrame(X_tr_f[bg_idx], columns=X.columns)
        X_ev_df = pd.DataFrame(X_te[ev_idx],   columns=X.columns)

        # Joint grouped Shapley — JointKanWrapper returns (n, 3); .mean() = grand mean
        joint_wrapper = _JointKanWrapper(kans)
        phi_joint = grouped_shapley_all(
            joint_wrapper, X_bg_df, X_ev_df,
            group_cols=group_cols, group_names=group_names,
            mc_samples=MC_SAMPLES, n_jobs=1, random_state=RANDOM_STATE,
        )
        phi_joint["fold"] = fold_idx
        all_phi_joint.append(phi_joint)

        # Per-trait grouped Shapley — one KanShim per trait
        for i, lbl in enumerate(TRAIT_LABELS):
            shim = _KanShim(kans[i])
            phi_t = grouped_shapley_all(
                shim, X_bg_df, X_ev_df,
                group_cols=group_cols, group_names=group_names,
                mc_samples=MC_SAMPLES, n_jobs=1, random_state=RANDOM_STATE,
            )
            phi_t["fold"] = fold_idx
            all_phi_per[lbl].append(phi_t)

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------
    print("\n--- CV R² summary (KAN) ---")
    cv_rows = []
    for lbl in TRAIT_LABELS:
        r2_vals = fold_r2[lbl]
        print(f"  {lbl}: mean={np.mean(r2_vals):.4f}  sd={np.std(r2_vals):.4f}")
        cv_rows.append({"trait": lbl, "mean_r2": np.mean(r2_vals),
                        "sd_r2": np.std(r2_vals)})
    print(f"  joint mean: {np.mean(fold_r2_joint):.4f}")

    phi_joint_all = pd.concat(all_phi_joint, ignore_index=True)
    summary_joint = summarize_grouped_shapley(phi_joint_all)
    print("\n--- Joint φ_k^KAN (chromosomes ranked) ---")
    print(summary_joint[["group", "mean_abs_phi", "sd_abs_phi"]].to_string(index=False))

    summaries_per = {}
    print("\n--- Per-trait φ_k (KAN) ---")
    for lbl in TRAIT_LABELS:
        phi_df = pd.concat(all_phi_per[lbl], ignore_index=True)
        s = summarize_grouped_shapley(phi_df)
        summaries_per[lbl] = s
        print(f"\n  {lbl}:")
        print(s[["group", "mean_abs_phi"]].to_string(index=False))

    # Cross-trait comparison table (KAN)
    chrs = [f"chr{i}" for i in range(1, 6)]
    rows_kan = []
    print("\n--- Cross-trait comparison table (KAN mean |φ_k|) ---")
    header = f"{'Chr':<8}" + "".join(f"  {lbl:<22}" for lbl in TRAIT_LABELS) + "  phi_joint_kan"
    print(header)
    for chrom in chrs:
        per_vals = []
        for lbl in TRAIT_LABELS:
            row = summaries_per[lbl].set_index("group")
            val = row.loc[chrom, "mean_abs_phi"] if chrom in row.index else 0.0
            per_vals.append(val)
        j_row = summary_joint.set_index("group")
        jv = j_row.loc[chrom, "mean_abs_phi"] if chrom in j_row.index else 0.0
        line = f"{chrom:<8}" + "".join(f"  {v:<22.4f}" for v in per_vals) + f"  {jv:.4f}"
        print(line)
        rows_kan.append({"chr": chrom,
                         **{lbl: pv for lbl, pv in zip(TRAIT_LABELS, per_vals)},
                         "phi_joint_kan": jv})
    cross_trait_kan = pd.DataFrame(rows_kan)

    # ------------------------------------------------------------------
    # Pairwise chromosome interactions (joint KAN wrapper, full dataset)
    # ------------------------------------------------------------------
    print("\n--- Training final 3 KANs on full dataset for interactions ---")
    ckpt_full = CKPT_BASE / "full"
    rng2 = np.random.default_rng(RANDOM_STATE + 999)
    n_val_full = max(1, n_acc // 10)
    perm_full = rng2.permutation(n_acc)
    X_val_full = X_arr[perm_full[:n_val_full]]
    Y_val_full = Y_norm[perm_full[:n_val_full]]
    X_tr_full  = X_arr[perm_full[n_val_full:]]
    Y_tr_full  = Y_norm[perm_full[n_val_full:]]

    kans_full = _train_three_kans(
        X_tr_full, Y_tr_full, X_val_full, Y_val_full,
        p_in=p_in, seed=RANDOM_STATE + 1000,
        ckpt_dir=ckpt_full,
    )
    joint_wrapper_full = _JointKanWrapper(kans_full)

    bg_arr = X_arr[rng2.choice(n_acc, min(N_BACKGROUND, n_acc), replace=False)]
    ev_arr = X_arr[rng2.choice(n_acc, min(N_EVAL, n_acc), replace=False)]
    bg_df  = pd.DataFrame(bg_arr, columns=X.columns)
    ev_df  = pd.DataFrame(ev_arr, columns=X.columns)

    print("\n--- Computing pairwise chromosome interactions (joint KAN) ---")
    interaction_df = grouped_interaction_all(
        joint_wrapper_full, bg_df, ev_df,
        group_cols=group_cols, group_names=group_names,
        mc_samples=MC_SAMPLES, n_jobs=1, random_state=RANDOM_STATE,
    )
    print("\n  Top-5 chromosome pairs (joint KAN) by |I_jk|:")
    print(interaction_df.head(5)[
        ["group_j", "group_k", "mean_abs_interaction", "sd_interaction"]
    ].to_string(index=False))

    # ------------------------------------------------------------------
    # Compare KAN vs RF (φ_joint, pairwise interactions)
    # ------------------------------------------------------------------
    comparison_rows = []
    rf_cross_path = RF_OUTDIR / "cross_trait_comparison.csv"
    rf_int_path   = RF_OUTDIR / "interactions_joint.csv"

    if rf_cross_path.exists():
        rf_cross = pd.read_csv(rf_cross_path)
        rf_cross = rf_cross.set_index("chr")
        print("\n--- KAN vs RF comparison (φ_joint) ---")
        print(f"{'Chr':<8}  {'φ_joint_KAN':>14}  {'φ_joint_RF':>14}  {'ratio KAN/RF':>14}")
        for chrom in chrs:
            kan_v = cross_trait_kan.set_index("chr").loc[chrom, "phi_joint_kan"]
            rf_v  = rf_cross.loc[chrom, "phi_joint"] if chrom in rf_cross.index else 0.0
            ratio = kan_v / rf_v if rf_v > 0 else float("nan")
            print(f"{chrom:<8}  {kan_v:>14.4f}  {rf_v:>14.4f}  {ratio:>14.2f}")
            comparison_rows.append({"chr": chrom, "phi_joint_kan": kan_v,
                                    "phi_joint_rf": rf_v, "ratio_kan_rf": ratio})
    else:
        print(f"  [WARN] RF cross-trait file not found at {rf_cross_path}")

    # Interaction magnitude comparison
    if rf_int_path.exists():
        rf_int = pd.read_csv(rf_int_path)
        median_kan = float(interaction_df["mean_abs_interaction"].median())
        median_rf  = float(rf_int["mean_abs_interaction"].median())
        max_kan    = float(interaction_df["mean_abs_interaction"].max())
        max_rf     = float(rf_int["mean_abs_interaction"].max())
        print(f"\n  KAN interaction median |I_jk|: {median_kan:.4g}  "
              f"max: {max_kan:.4g}")
        print(f"  RF  interaction median |I_jk|: {median_rf:.4g}  "
              f"max: {max_rf:.4g}")
        ratio_median = median_kan / median_rf if median_rf > 0 else float("nan")
        print(f"  KAN/RF ratio (median): {ratio_median:.2f}x")
    else:
        print(f"  [WARN] RF interaction file not found at {rf_int_path}")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    phi_joint_all.to_csv(OUTDIR / "phi_joint_kan_all.csv", index=False)
    summary_joint.to_csv(OUTDIR / "phi_joint_kan_summary.csv", index=False)
    cross_trait_kan.to_csv(OUTDIR / "cross_trait_comparison_kan.csv", index=False)
    interaction_df.to_csv(OUTDIR / "interactions_joint_kan.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(OUTDIR / "cv_metrics_kan.csv", index=False)
    if comparison_rows:
        pd.DataFrame(comparison_rows).to_csv(OUTDIR / "comparison_kan_vs_rf.csv", index=False)
    for lbl in TRAIT_LABELS:
        phi_df = pd.concat(all_phi_per[lbl], ignore_index=True)
        phi_df.to_csv(OUTDIR / f"phi_{lbl}_kan_all.csv", index=False)
        summaries_per[lbl].to_csv(OUTDIR / f"phi_{lbl}_kan_summary.csv", index=False)

    elapsed = time.perf_counter() - t0
    print(f"\n  Wall-clock: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"\nAll outputs written to {OUTDIR}/")

    pd.DataFrame([{
        "n_accessions": n_acc,
        "n_snps":       n_snps,
        "n_traits":     n_traits,
        "mean_r2_joint_kan":  float(np.mean(fold_r2_joint)),
        "top_chr_kan":  summary_joint.iloc[0]["group"],
        "top_phi_kan":  float(summary_joint.iloc[0]["mean_abs_phi"]),
        "elapsed_s":    elapsed,
    }]).to_csv(OUTDIR / "run_summary.csv", index=False)


if __name__ == "__main__":
    main()
