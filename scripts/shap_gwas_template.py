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

Default nonlinear baseline:
- Random forest regressor with Tree SHAP.

Optional secondary model:
- XGBoost regressor if explicitly selected.
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
    grouped_shapley_all,
    load_group_manifest,
    summarize_grouped_shapley,
)


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
    parser.add_argument(
        "--model",
        choices=["random_forest", "xgboost"],
        default="random_forest",
        help="Primary baseline is random_forest; xgboost is optional.",
    )
    parser.add_argument("--outdir", default="outputs/shap_gwas_random_forest")
    parser.add_argument(
        "--group-manifest",
        default=None,
        help=(
            "Optional TSV with columns 'feature' and 'group' defining the SNP grouping "
            "for cooperative-game grouped Shapley (players = groups, not individual SNPs). "
            "Groups can be chromosomes, LD blocks, gene windows, etc. "
            "When omitted, only per-SNP SHAP is computed."
        ),
    )
    parser.add_argument(
        "--n-background-grouped",
        type=int,
        default=100,
        help="Background samples for the grouped Shapley value function.",
    )
    parser.add_argument(
        "--n-eval-grouped",
        type=int,
        default=50,
        help="Evaluation points per fold for grouped Shapley.",
    )
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


def build_model(model_name: str, random_state: int):
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        )

    if model_name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "Missing dependency: xgboost. Use --model random_forest or install xgboost."
            ) from exc

        return XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=random_state,
        )

    raise SystemExit(f"Unsupported model: {model_name}")


def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    snp_cols: list[str],
    covar_cols: list[str],
    folds: int,
    random_state: int,
    model_name: str,
    group_names: list[str] | None,
    group_cols_list: list[list[str]] | None,
    n_background_grouped: int,
    n_eval_grouped: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (metrics_df, per_snp_shap_df, grouped_shap_df).

    Per-SNP SHAP: shap.TreeExplainer — each SNP is an individual player.
    Grouped Shapley: cooperative-game interventional value function over
      user-supplied groups.  Requires --group-manifest.  This is the
      theoretically grounded group-additive estimand.
    """
    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    metrics = []
    shap_tables = []
    grouped_tables = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model = build_model(model_name, random_state + fold_idx)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        metrics.append(
            {
                "fold": fold_idx,
                "r2": r2_score(y_test, pred),
                "rmse": mean_squared_error(y_test, pred, squared=False),
            }
        )

        # ── Per-SNP SHAP (individual player) ─────────────────────────────────
        # Each SNP is a separate player.  Under LD, attributions for correlated
        # SNPs are split arbitrarily and do NOT recover the group-additive
        # estimand.  Use grouped Shapley (below) for
        # theoretically grounded group-level attribution.
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

        # ── Cooperative-game grouped Shapley (when group manifest provided) ───
        if group_names is not None and group_cols_list is not None:
            rng = np.random.RandomState(random_state + fold_idx)
            bg_idx = rng.choice(
                len(X_train), size=min(n_background_grouped, len(X_train)), replace=False
            )
            eval_idx = rng.choice(
                len(X_test), size=min(n_eval_grouped, len(X_test)), replace=False
            )
            X_bg = X_train.iloc[bg_idx].reset_index(drop=True)
            X_eval = X_test.iloc[eval_idx].reset_index(drop=True)
            phi_df = grouped_shapley_all(model, X_bg, X_eval, group_cols_list, group_names)
            phi_df["fold"] = fold_idx
            grouped_tables.append(phi_df)

    grouped_df = pd.concat(grouped_tables, ignore_index=True) if grouped_tables else pd.DataFrame()
    return pd.DataFrame(metrics), pd.concat(shap_tables, ignore_index=True), grouped_df


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

    group_names = None
    group_cols_list = None
    if args.group_manifest:
        group_names, group_cols_list, _ = load_group_manifest(args.group_manifest)
        # Keep only groups whose columns are present in X
        group_names, group_cols_list = zip(
            *[(n, c) for n, c in zip(group_names, group_cols_list) if all(col in X.columns for col in c)]
        ) if group_names else ([], [])
        group_names = list(group_names)
        group_cols_list = list(group_cols_list)

    metrics_df, shap_df, grouped_df = run_cv(
        X,
        y,
        snp_cols,
        covar_cols,
        args.folds,
        args.random_state,
        args.model,
        group_names,
        group_cols_list,
        args.n_background_grouped,
        args.n_eval_grouped,
    )
    shap_summary = summarize_shap(shap_df)

    metrics_df.to_csv(outdir / "cv_metrics.tsv", sep="\t", index=False)
    shap_df.to_csv(outdir / "fold_shap_importance.tsv", sep="\t", index=False)
    shap_summary.to_csv(outdir / "global_shap_ranking.tsv", sep="\t", index=False)

    if not grouped_df.empty:
        grp_names = [c for c in grouped_df.columns if c != "fold"]
        summary_grp = summarize_grouped_shapley(grouped_df.drop(columns=["fold"]), grp_names)
        grouped_df.to_csv(outdir / "fold_grouped_shapley.tsv", sep="\t", index=False)
        summary_grp.to_csv(outdir / "grouped_shapley_ranking.tsv", sep="\t", index=False)

    (outdir / "model_info.txt").write_text(
        f"model\t{args.model}\nfolds\t{args.folds}\nrandom_state\t{args.random_state}\n"
        f"group_manifest\t{args.group_manifest or 'none'}\n",
        encoding="ascii",
    )

    print("Saved:")
    print(outdir / "cv_metrics.tsv")
    print(outdir / "fold_shap_importance.tsv")
    print(outdir / "global_shap_ranking.tsv")
    print(outdir / "model_info.txt")
    if not grouped_df.empty:
        print(outdir / "fold_grouped_shapley.tsv")
        print(outdir / "grouped_shapley_ranking.tsv")


if __name__ == "__main__":
    main()
