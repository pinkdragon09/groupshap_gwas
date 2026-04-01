#!/usr/bin/env python3
"""
Template pipeline for a SHAP-based GWAS experiment on a filtered SNP matrix.

Assumptions:
- Genotype matrix is a TSV with samples on rows and SNPs on columns.
- Phenotype TSV has columns: sample_id, phenotype
- Optional covariate TSV has sample_id plus covariate columns.
- Optional baseline GWAS TSV is a PLINK .glm.linear file used to prefilter SNPs.

This script is intentionally scoped to the post-filtering stage. For a real
genome-wide study, use PLINK for QC and initial screening first.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

try:
    import shap
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: shap") from exc

try:
    from xgboost import XGBRegressor
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: xgboost") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--genotype-tsv", required=True)
    parser.add_argument("--phenotype-tsv", required=True)
    parser.add_argument("--covariate-tsv")
    parser.add_argument("--sample-id-col", default="sample_id")
    parser.add_argument("--phenotype-col", default="phenotype")
    parser.add_argument("--baseline-glm")
    parser.add_argument("--top-snps", type=int, default=500)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--outdir", default="outputs/shap_gwas")
    return parser.parse_args()


def load_tables(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    geno = pd.read_csv(args.genotype_tsv, sep="\t")
    pheno = pd.read_csv(args.phenotype_tsv, sep="\t")

    merged = geno.merge(pheno[[args.sample_id_col, args.phenotype_col]], on=args.sample_id_col)

    if args.covariate_tsv:
        covar = pd.read_csv(args.covariate_tsv, sep="\t")
        merged = merged.merge(covar, on=args.sample_id_col)
        covar_cols = [c for c in covar.columns if c != args.sample_id_col]
    else:
        covar_cols = []

    feature_cols = [
        c for c in merged.columns
        if c not in {args.sample_id_col, args.phenotype_col, *covar_cols}
    ]

    if args.baseline_glm:
        top_features = select_top_snps_from_glm(args.baseline_glm, args.top_snps)
        feature_cols = [c for c in feature_cols if c in top_features]

    if not feature_cols:
        raise SystemExit("No SNP columns available after filtering.")

    X = merged[feature_cols + covar_cols].copy()
    y = merged[args.phenotype_col].copy()
    return X, y, feature_cols, covar_cols


def select_top_snps_from_glm(glm_path: str, top_snps: int) -> set[str]:
    glm = pd.read_csv(glm_path, sep=r"\s+")
    glm = glm[glm["TEST"] == "ADD"].copy()
    glm = glm.dropna(subset=["P"])
    glm = glm.sort_values("P", ascending=True)
    return set(glm["ID"].head(top_snps).astype(str))


def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    snp_cols: list[str],
    covar_cols: list[str],
    folds: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    metrics = []
    shap_tables = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=random_state + fold_idx,
        )
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        metrics.append(
            {
                "fold": fold_idx,
                "r2": r2_score(y_test, pred),
                "rmse": mean_squared_error(y_test, pred, squared=False),
            }
        )

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        fold_importance = pd.DataFrame(
            {
                "feature": X.columns,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
                "feature_type": ["snp" if col in snp_cols else "covariate" for col in X.columns],
                "fold": fold_idx,
            }
        )
        shap_tables.append(fold_importance)

    return pd.DataFrame(metrics), pd.concat(shap_tables, ignore_index=True)


def summarize_shap(shap_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        shap_df.groupby(["feature", "feature_type"], as_index=False)
        .agg(
            mean_abs_shap=("mean_abs_shap", "mean"),
            sd_abs_shap=("mean_abs_shap", "std"),
            n_folds=("mean_abs_shap", "count"),
        )
        .sort_values("mean_abs_shap", ascending=False)
    )
    return summary


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    X, y, snp_cols, covar_cols = load_tables(args)
    metrics_df, shap_df = run_cv(X, y, snp_cols, covar_cols, args.folds, args.random_state)
    shap_summary = summarize_shap(shap_df)

    metrics_df.to_csv(outdir / "cv_metrics.tsv", sep="\t", index=False)
    shap_df.to_csv(outdir / "fold_shap_importance.tsv", sep="\t", index=False)
    shap_summary.to_csv(outdir / "global_shap_ranking.tsv", sep="\t", index=False)

    print("Saved:")
    print(outdir / "cv_metrics.tsv")
    print(outdir / "fold_shap_importance.tsv")
    print(outdir / "global_shap_ranking.tsv")


if __name__ == "__main__":
    main()
