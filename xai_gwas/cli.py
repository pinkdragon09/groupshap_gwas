"""Command-line interface for xai-gwas.

Usage examples
--------------
# Full pipeline on a PLINK dataset:
    xai-gwas run --bfile data/sim --phenotype-col PHENO --grouping window --outdir results/

# Run on a merged TSV (project-native format):
    xai-gwas run --tsv data/merged.tsv --phenotype-col FT10 --grouping ld_corr --outdir results/

# Compute only grouped Shapley (skip interaction):
    xai-gwas run --tsv data/merged.tsv --phenotype-col FT10 --no-interaction --outdir results/

Subcommands
-----------
run      Full grouped-SHAP attribution pipeline
summary  Summarise outputs from a previous run (reads phi CSV from outdir)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xai-gwas",
        description="Grouped Shapley and KAN explainability for GWAS / QTL mapping.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- run subcommand ---
    run_p = sub.add_parser("run", help="Run grouped-SHAP attribution pipeline.")

    # Input (mutually exclusive: PLINK bfile or TSV)
    input_grp = run_p.add_mutually_exclusive_group(required=True)
    input_grp.add_argument("--bfile", help="PLINK binary prefix (.bed/.bim/.fam).")
    input_grp.add_argument("--tsv", help="Merged genotype-phenotype TSV file.")

    run_p.add_argument("--phenotype-col", required=True, help="Phenotype column name.")
    run_p.add_argument("--phenotype-file", default=None,
                       help="Separate PLINK-format phenotype file (for --bfile inputs).")
    run_p.add_argument("--grouping",
                       choices=["window", "ld_corr", "plink_blocks", "manifest"],
                       default="window",
                       help="SNP grouping strategy.")
    run_p.add_argument("--manifest", default=None,
                       help="Path to group manifest TSV (required for --grouping manifest).")
    run_p.add_argument("--window-size", type=int, default=50_000,
                       help="Window size in bp for 'window' grouping (default 50 kb).")
    run_p.add_argument("--r2-threshold", type=float, default=0.5,
                       help="r² threshold for LD correlation clustering.")
    run_p.add_argument("--top-k-snps", type=int, default=100,
                       help="SNPs to retain after variance pre-filtering.")
    run_p.add_argument("--mc-samples", type=int, default=50,
                       help="Monte Carlo permutations per eval point per group.")
    run_p.add_argument("--n-background", type=int, default=50,
                       help="Background samples for value-function marginalisation.")
    run_p.add_argument("--n-eval", type=int, default=20,
                       help="Eval points for grouped Shapley estimation.")
    run_p.add_argument("--folds", type=int, default=3,
                       help="Cross-validation folds for RF fitting.")
    run_p.add_argument("--no-interaction", action="store_true",
                       help="Skip pairwise interaction index computation.")
    run_p.add_argument("--no-plots", action="store_true",
                       help="Skip figure generation (useful for headless HPC).")
    run_p.add_argument("--random-state", type=int, default=42)
    run_p.add_argument("--n-jobs", type=int, default=-1,
                       help="Joblib parallelism (-1 = all cores).")
    run_p.add_argument("--outdir", default="results/xai_gwas",
                       help="Output directory.")

    # --- summary subcommand ---
    sum_p = sub.add_parser("summary", help="Summarise outputs from a previous run.")
    sum_p.add_argument("--outdir", required=True, help="Output directory from a prior run.")
    sum_p.add_argument("--top-n", type=int, default=20, help="Top N groups to report.")

    return parser


# ---------------------------------------------------------------------------
# Run subcommand
# ---------------------------------------------------------------------------

def _run(args: argparse.Namespace) -> None:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score

    from xai_gwas.data import load_tsv, load_plink_bfile
    from xai_gwas.grouping import (
        load_group_manifest, build_window_grouping,
        group_by_ld_correlation, group_by_plink_blocks,
    )
    from xai_gwas.attribution import (
        grouped_shapley_all, grouped_interaction_all, summarize_grouped_shapley,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    print("Loading data ...")
    if args.tsv:
        X, y = load_tsv(args.tsv, phenotype_col=args.phenotype_col)
    else:
        X, y = load_plink_bfile(
            args.bfile,
            phenotype_path=args.phenotype_file,
            phenotype_col=args.phenotype_col,
        )
    if y is None:
        print("ERROR: no phenotype found. Provide --phenotype-col or --phenotype-file.")
        sys.exit(1)

    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]
    print(f"  {X.shape[0]} samples × {X.shape[1]} SNPs loaded.")

    # 2. Variance pre-filter
    variances = X.var().sort_values(ascending=False)
    top_snps = variances.head(args.top_k_snps).index.tolist()
    top_snps = [s for s in top_snps if s in X.columns]
    X = X[top_snps]
    print(f"  Retained {len(top_snps)} SNPs after variance pre-filter.")

    # 3. Grouping
    print(f"Building groups (strategy={args.grouping}) ...")
    if args.grouping == "manifest":
        if not args.manifest:
            print("ERROR: --manifest is required for --grouping manifest.")
            sys.exit(1)
        group_names, group_cols, _ = load_group_manifest(args.manifest)
    elif args.grouping == "ld_corr":
        group_names, group_cols = group_by_ld_correlation(X, r2_threshold=args.r2_threshold)
    elif args.grouping == "plink_blocks":
        group_names, group_cols = group_by_plink_blocks(X)
    else:  # window
        group_names, group_cols = build_window_grouping(list(X.columns), window_size=args.window_size)

    # Keep only groups whose columns are present
    group_names_f, group_cols_f = [], []
    for gn, gc in zip(group_names, group_cols):
        gc_present = [c for c in gc if c in X.columns]
        if gc_present:
            group_names_f.append(gn)
            group_cols_f.append(gc_present)
    group_names, group_cols = group_names_f, group_cols_f
    print(f"  {len(group_names)} groups defined.")

    # 4. Cross-validated RF + grouped Shapley
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.random_state)
    all_phi, fold_metrics = [], []

    rng = np.random.default_rng(args.random_state)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        print(f"  Fold {fold_idx}/{args.folds} — fitting RF ...")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        rf = RandomForestRegressor(
            n_estimators=300, max_features=0.33, min_samples_leaf=5,
            n_jobs=args.n_jobs, random_state=args.random_state,
        )
        rf.fit(X_train, y_train)

        r2 = r2_score(y_test, rf.predict(X_test))
        fold_metrics.append({"fold": fold_idx, "r2_test": r2})
        print(f"    test R²={r2:.4f}")

        # Background + eval samples
        bg_idx = rng.choice(len(X_train), size=min(args.n_background, len(X_train)), replace=False)
        ev_idx = rng.choice(len(X_test), size=min(args.n_eval, len(X_test)), replace=False)
        X_bg = X_train.iloc[bg_idx]
        X_ev = X_test.iloc[ev_idx]

        phi_df = grouped_shapley_all(
            rf, X_bg, X_ev, group_cols, group_names=group_names,
            mc_samples=args.mc_samples, n_jobs=args.n_jobs,
            random_state=args.random_state,
        )
        phi_df["fold"] = fold_idx
        all_phi.append(phi_df)

    # Save metrics
    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df.to_csv(outdir / "cv_metrics.csv", index=False)
    print(f"\nMean test R² = {metrics_df['r2_test'].mean():.4f}")

    # Concatenate and summarise Shapley values
    phi_all = pd.concat(all_phi, ignore_index=True)
    phi_all.to_csv(outdir / "phi_all.csv", index=False)

    summary = summarize_grouped_shapley(phi_all)
    summary.to_csv(outdir / "phi_summary.csv", index=False)
    print(f"\nTop 5 groups by |φ|:")
    print(summary.head(5).to_string(index=False))

    # 5. Interaction (optional)
    if not args.no_interaction:
        print("\nComputing pairwise Shapley interactions (full dataset background) ...")
        bg_all = X.sample(min(args.n_background, len(X)), random_state=args.random_state)
        ev_all = X.sample(min(args.n_eval, len(X)), random_state=args.random_state + 1)
        # Use first-fold RF for interaction (fit on majority of data)
        first_train_idx = list(kf.split(X))[0][0]
        rf_full = RandomForestRegressor(
            n_estimators=300, max_features=0.33, min_samples_leaf=5,
            n_jobs=args.n_jobs, random_state=args.random_state,
        )
        rf_full.fit(X.iloc[first_train_idx], y.iloc[first_train_idx])

        interaction_df = grouped_interaction_all(
            rf_full, bg_all, ev_all, group_cols, group_names,
            mc_samples=args.mc_samples, n_jobs=args.n_jobs,
            random_state=args.random_state,
        )
        interaction_df.to_csv(outdir / "interactions.csv", index=False)
        print("Top 5 interactions:")
        print(interaction_df.head(5).to_string(index=False))

    # 6. Plots (optional)
    if not args.no_plots:
        _make_plots(summary, outdir)

    print(f"\nDone. Results in {outdir}/")


def _make_plots(summary_df, outdir: Path) -> None:
    try:
        from xai_gwas.viz import shapley_barplot
        import matplotlib
        matplotlib.use("Agg")
        fig, _ = shapley_barplot(summary_df, title="Grouped Shapley Values")
        fig.savefig(outdir / "phi_barplot.png", dpi=150)
        print(f"  Saved phi_barplot.png")
    except Exception as e:
        print(f"  Warning: could not generate plots ({e})")


# ---------------------------------------------------------------------------
# Summary subcommand
# ---------------------------------------------------------------------------

def _summary(args: argparse.Namespace) -> None:
    import pandas as pd

    outdir = Path(args.outdir)
    phi_path = outdir / "phi_summary.csv"
    if not phi_path.exists():
        print(f"ERROR: {phi_path} not found. Run 'xai-gwas run' first.")
        sys.exit(1)

    summary = pd.read_csv(phi_path)
    print(f"\nTop {args.top_n} groups by mean |φ|:")
    print(summary.head(args.top_n).to_string(index=False))

    metrics_path = outdir / "cv_metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        print(f"\nCV R² summary:")
        print(f"  Mean={metrics['r2_test'].mean():.4f}  "
              f"SD={metrics['r2_test'].std():.4f}  "
              f"Min={metrics['r2_test'].min():.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        _run(args)
    elif args.command == "summary":
        _summary(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
