#!/usr/bin/env python3
"""
Arabidopsis-ready random-forest + SHAP runner for the downloaded local data.

Default inputs:
- AraPheno bulk database zip
- 1001 Genomes imputed SNP HDF5 matrix
- 1001 Genomes kinship HDF5 for principal components
- baseline top-hit table from the local PC-adjusted linear GWAS scan

The intended workflow is:
1. pick one quantitative AraPheno trait
2. prefilter SNPs using the baseline hit table
3. add kinship-derived PCs as covariates
4. train a random-forest regressor with cross-validation
5. compute Tree SHAP rankings across folds

Use --debug for a fast smoke test with fewer SNPs, folds, and trees.
"""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

try:
    import shap
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: shap") from exc

from grouped_shap_utils import (
    build_group_cols_from_meta,
    grouped_shapley_all,
    summarize_grouped_shapley,
)


PHENO_ZIP = Path("data/arapheno/bulk/database.zip")
GENO_H5 = Path("data/1001genomes/genotypes/1001_SNP_MATRIX/imputed_snps_binary.hdf5")
KINSHIP_H5 = Path("data/1001genomes/genotypes/1001_SNP_MATRIX/kinship_ibs_mac5.hdf5")
BASELINE_HITS = Path("outputs/full_baseline_gwas/study27_mean_trl_top_hits.tsv")
DEFAULT_OUTDIR = Path("outputs/random_forest_gwas")

DEFAULT_STUDY_VALUES_FILE = "27/study_27_values.csv"
DEFAULT_PHENOTYPE_COL = "Mean(TRL) C"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pheno-zip", default=str(PHENO_ZIP))
    parser.add_argument("--geno-h5", default=str(GENO_H5))
    parser.add_argument("--kinship-h5", default=str(KINSHIP_H5))
    parser.add_argument("--baseline-hits", default=str(BASELINE_HITS))
    parser.add_argument("--study-values-file", default=DEFAULT_STUDY_VALUES_FILE)
    parser.add_argument("--phenotype-col", default=DEFAULT_PHENOTYPE_COL)
    parser.add_argument("--top-snps", type=int, default=500)
    parser.add_argument("--n-pcs", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-features", default="sqrt")
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run a smaller smoke test with fewer SNPs, folds, and trees.",
    )
    # Grouped Shapley options
    parser.add_argument(
        "--group-by",
        default="chromosome",
        choices=["chromosome", "none"],
        help=(
            "Grouping scheme for cooperative-game grouped Shapley. "
            "'chromosome' groups SNPs by chromosome (Arabidopsis: 5 groups). "
            "'none' skips grouped Shapley and outputs per-SNP SHAP only."
        ),
    )
    parser.add_argument(
        "--n-background-grouped",
        type=int,
        default=100,
        help="Background samples used to marginalise the grouped Shapley value function.",
    )
    parser.add_argument(
        "--n-eval-grouped",
        type=int,
        default=50,
        help="Evaluation points per fold for grouped Shapley (subset of test set).",
    )
    return parser.parse_args()


def apply_debug_overrides(args: argparse.Namespace) -> None:
    if not args.debug:
        return
    args.top_snps = min(args.top_snps, 100)
    args.folds = min(args.folds, 3)
    args.n_estimators = min(args.n_estimators, 100)
    args.n_background_grouped = min(args.n_background_grouped, 30)
    args.n_eval_grouped = min(args.n_eval_grouped, 10)


def load_phenotype(pheno_zip: Path, study_values_file: str, phenotype_col: str) -> dict[str, float]:
    with zipfile.ZipFile(pheno_zip) as zf:
        rows = list(csv.DictReader(io.StringIO(zf.read(study_values_file).decode("utf-8"))))

    y_by_acc: dict[str, float] = {}
    for row in rows:
        acc = row["accession_id"].strip()
        value = row[phenotype_col].strip()
        if not acc or not value:
            continue
        y_by_acc[acc] = float(value)
    if not y_by_acc:
        raise SystemExit(f"No usable phenotype rows found in {study_values_file} for {phenotype_col}.")
    return y_by_acc


def load_top_snp_indices(baseline_hits: Path, top_snps: int) -> list[int]:
    hits = pd.read_csv(baseline_hits, sep="\t")
    if "snp_index" not in hits.columns:
        raise SystemExit(f"Baseline hit table missing snp_index column: {baseline_hits}")
    hits = hits.sort_values("p_value", ascending=True)
    return hits["snp_index"].astype(int).head(top_snps).tolist()


def compute_pcs(kinship_h5: Path, common_accs: list[str], n_pcs: int) -> np.ndarray:
    with h5py.File(kinship_h5, "r") as f:
        kinship = f["kinship"][:]
        kin_accs = [x.decode() if isinstance(x, bytes) else str(x) for x in f["accessions"][:]]

    kin_index = {acc: i for i, acc in enumerate(kin_accs)}
    ix = np.asarray([kin_index[acc] for acc in common_accs], dtype=int)
    sub_k = kinship[np.ix_(ix, ix)]
    eigvals, eigvecs = np.linalg.eigh(sub_k)
    pcs = eigvecs[:, -n_pcs:]
    pcs = (pcs - pcs.mean(axis=0)) / pcs.std(axis=0)
    return pcs


def chromosome_labels_and_bounds(h5f: h5py.File) -> tuple[list[str], np.ndarray]:
    labels = [f"Chr{x.decode() if isinstance(x, bytes) else x}" for x in h5f["positions"].attrs["chrs"]]
    bounds = h5f["positions"].attrs["chr_regions"]
    return labels, bounds


def chromosome_for_index(idx: int, labels: list[str], bounds: np.ndarray) -> str:
    for chrom_idx, (start, stop) in enumerate(bounds):
        if int(start) <= idx < int(stop):
            return labels[chrom_idx]
    return "Chr?"


def load_arabidopsis_matrix(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    y_by_acc = load_phenotype(Path(args.pheno_zip), args.study_values_file, args.phenotype_col)
    snp_indices = load_top_snp_indices(Path(args.baseline_hits), args.top_snps)
    sorted_snp_indices = sorted(snp_indices)

    with h5py.File(args.geno_h5, "r") as f:
        geno_accs = [x.decode() if isinstance(x, bytes) else str(x) for x in f["accessions"][:]]
        common_accs = [acc for acc in geno_accs if acc in y_by_acc]
        geno_ix = np.asarray([geno_accs.index(acc) for acc in common_accs], dtype=int)
        positions = f["positions"][sorted_snp_indices]
        snps = f["snps"][sorted_snp_indices][:, geno_ix].astype(float)
        labels, bounds = chromosome_labels_and_bounds(f)

    pcs = compute_pcs(Path(args.kinship_h5), common_accs, args.n_pcs)
    y = pd.Series([y_by_acc[acc] for acc in common_accs], name=args.phenotype_col, dtype=float)

    snp_cols = [f"snp_{idx}" for idx in sorted_snp_indices]
    X_snp = pd.DataFrame(snps.T, columns=snp_cols)
    X_cov = pd.DataFrame(pcs, columns=[f"pc{i}" for i in range(1, args.n_pcs + 1)])
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


def build_model(args: argparse.Namespace, fold_idx: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_features=args.max_features,
        min_samples_leaf=args.min_samples_leaf,
        n_jobs=-1,
        random_state=args.random_state + fold_idx,
    )


def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    feature_meta: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (metrics_df, per_snp_shap_df, grouped_shap_df).

    Per-SNP SHAP uses shap.TreeExplainer — each SNP is an individual player.
    Grouped Shapley uses the cooperative-game interventional value function
    (v(S; x) = E[f(x_S, X_{-S})]) — players = chromosome groups.  This is
    the group-additive estimand.
    """
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.random_state)
    metrics = []
    shap_tables = []
    grouped_tables = []

    # Build group structure once (same for all folds)
    if args.group_by != "none":
        group_names, group_cols_list = build_group_cols_from_meta(
            feature_meta, group_col=args.group_by
        )

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model = build_model(args, fold_idx)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        metrics.append(
            {
                "fold": fold_idx,
                "r2": r2_score(y_test, pred),
                "rmse": rmse,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            }
        )

        # ── Per-SNP SHAP (individual player, Tree SHAP) ──────────────────────
        # Each SNP is treated as a separate player.  Under LD, attributions for
        # correlated SNPs are split arbitrarily and do NOT correspond to the
        # group-additive estimand studied in the theory.  Use grouped Shapley
        # below for the theoretically grounded group-level attribution.
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        fold_importance = feature_meta.copy()
        fold_importance["mean_abs_shap"] = np.abs(shap_values).mean(axis=0)
        fold_importance["fold"] = fold_idx
        shap_tables.append(fold_importance)

        # ── Cooperative-game grouped Shapley ─────────────────────────────────
        # Players = chromosome groups.  Value function v(S; x) marginalises
        # non-coalition groups (AND covariates) over background samples.
        # This matches the group-additive estimand.
        if args.group_by != "none":
            rng = np.random.RandomState(args.random_state + fold_idx)
            bg_idx = rng.choice(
                len(X_train), size=min(args.n_background_grouped, len(X_train)), replace=False
            )
            eval_idx = rng.choice(
                len(X_test), size=min(args.n_eval_grouped, len(X_test)), replace=False
            )
            X_bg = X_train.iloc[bg_idx].reset_index(drop=True)
            X_eval = X_test.iloc[eval_idx].reset_index(drop=True)

            phi_df = grouped_shapley_all(model, X_bg, X_eval, group_cols_list, group_names)
            phi_df["fold"] = fold_idx
            grouped_tables.append(phi_df)

    grouped_df = pd.concat(grouped_tables, ignore_index=True) if grouped_tables else pd.DataFrame()
    return pd.DataFrame(metrics), pd.concat(shap_tables, ignore_index=True), grouped_df


def summarize_shap(shap_df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = ["feature", "feature_type", "snp_index", "chromosome", "position"]
    summary = (
        shap_df.groupby(keep_cols, dropna=False, as_index=False)
        .agg(
            mean_abs_shap=("mean_abs_shap", "mean"),
            sd_abs_shap=("mean_abs_shap", "std"),
            n_folds=("mean_abs_shap", "count"),
        )
        .sort_values("mean_abs_shap", ascending=False)
    )
    return summary


def write_outputs(
    outdir: Path,
    args: argparse.Namespace,
    X: pd.DataFrame,
    y: pd.Series,
    metrics_df: pd.DataFrame,
    shap_df: pd.DataFrame,
    shap_summary: pd.DataFrame,
    grouped_df: pd.DataFrame,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(outdir / "cv_metrics.tsv", sep="\t", index=False)
    shap_df.to_csv(outdir / "fold_shap_importance.tsv", sep="\t", index=False)
    shap_summary.to_csv(outdir / "global_shap_ranking.tsv", sep="\t", index=False)

    if not grouped_df.empty:
        group_names = [c for c in grouped_df.columns if c != "fold"]
        summary_grp = summarize_grouped_shapley(
            grouped_df.drop(columns=["fold"]), group_names
        )
        grouped_df.to_csv(outdir / "fold_grouped_shapley.tsv", sep="\t", index=False)
        summary_grp.to_csv(outdir / "grouped_shapley_ranking.tsv", sep="\t", index=False)

    summary_lines = [
        f"study_values_file: {args.study_values_file}",
        f"phenotype_col: {args.phenotype_col}",
        f"matched_accessions: {len(y)}",
        f"top_snps: {args.top_snps}",
        f"n_pcs: {args.n_pcs}",
        f"folds: {args.folds}",
        f"n_estimators: {args.n_estimators}",
        f"group_by: {args.group_by}",
        f"n_background_grouped: {args.n_background_grouped}",
        f"n_eval_grouped: {args.n_eval_grouped}",
        f"debug: {args.debug}",
        f"feature_count_total: {X.shape[1]}",
        f"feature_count_snps: {sum(col.startswith('snp_') for col in X.columns)}",
        f"mean_cv_r2: {metrics_df['r2'].mean():.6f}",
        f"mean_cv_rmse: {metrics_df['rmse'].mean():.6f}",
    ]
    (outdir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="ascii")


def main() -> None:
    args = parse_args()
    apply_debug_overrides(args)
    outdir = Path(args.outdir)

    X, y, feature_meta = load_arabidopsis_matrix(args)
    metrics_df, shap_df, grouped_df = run_cv(X, y, feature_meta, args)
    shap_summary = summarize_shap(shap_df)
    write_outputs(outdir, args, X, y, metrics_df, shap_df, shap_summary, grouped_df)

    print(f"Saved outputs to: {outdir}")
    print(f"Matched accessions: {len(y)}")
    print(f"Mean CV R2: {metrics_df['r2'].mean():.4f}")
    print(f"Top per-SNP feature: {shap_summary.iloc[0]['feature']}")
    if not grouped_df.empty:
        group_names = [c for c in grouped_df.columns if c != "fold"]
        summary_grp = summarize_grouped_shapley(grouped_df.drop(columns=["fold"]), group_names)
        print(f"Top chromosome group: {summary_grp.iloc[0]['group']}")


if __name__ == "__main__":
    main()
