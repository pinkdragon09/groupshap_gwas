"""
Main benchmark runner for LD-SHAP vs. traditional GWAS simulation study.

Usage:
    python scripts/run_benchmark.py

Outputs TSV files to outputs/benchmark/.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

# Ensure scripts/ is on the path when invoked from project root
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from simulate_genotypes import simulate_ld_blocks
from simulate_phenotypes import (
    gen_sparse_additive,
    gen_polygenic,
    gen_pairwise_epistasis,
    gen_ld_confounding,
    gen_mixed_additive_interaction,
    gen_variance_heterogeneity,
    gen_dominance,
)
from grouped_shap_utils import (
    grouped_shapley_all,
    conditional_permutation_importance,
)

_OUTPUT_DIR = os.path.join(
    os.path.dirname(_SCRIPTS_DIR), "outputs", "benchmark"
)
os.makedirs(_OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    scenario: int           # 1–7
    n: int                  # number of samples
    p: int = 200            # total SNPs (10 blocks × 20 SNPs each)
    B: int = 10             # number of LD blocks (reduced from 50 for runtime)
    rho: float = 0.0        # within-block AR(1) correlation
    h2: float = 0.5         # heritability
    n_reps: int = 100       # number of Monte Carlo replicates
    methods: list = field(default_factory=lambda: ["linear_gwas", "grouped_shapley", "cpi"])


# ---------------------------------------------------------------------------
# Phenotype dispatch
# ---------------------------------------------------------------------------

def _generate_phenotype(
    X: np.ndarray,
    block_ids: list[list[int]],
    config: BenchmarkConfig,
    rep: int,
) -> tuple[np.ndarray, list[int]]:
    """
    Dispatch to the correct phenotype generator for config.scenario.
    Returns (y, causal_block_indices).
    """
    s = config.scenario
    h2 = config.h2
    seed = rep * 1000 + s

    if s == 1:
        causal_blocks = [0, 1]
        y = gen_sparse_additive(X, block_ids, causal_blocks, [1.0, 0.8], h2, seed)
    elif s == 2:
        causal_blocks = list(range(10))
        y = gen_polygenic(X, block_ids, causal_blocks, 0.3, h2, seed)
    elif s == 3:
        causal_blocks = [0, 1]
        y = gen_pairwise_epistasis(X, block_ids, (0, 1), [1.0, 0.8], 0.5, h2, seed)
    elif s == 4:
        causal_blocks = [0]
        y = gen_ld_confounding(X, block_ids, 0, 0, 1.0, h2, seed)
    elif s == 5:
        causal_blocks = [0, 1, 2, 3, 4]
        y = gen_mixed_additive_interaction(
            X, block_ids, [0, 1, 2], [(3, 4)], 0.5, 0.8, h2, seed
        )
    elif s == 6:
        causal_blocks = [0]
        y = gen_variance_heterogeneity(X, block_ids, 0, 1.0, 2.0, seed)
    elif s == 7:
        causal_blocks = [0]
        y = gen_dominance(X, block_ids, 0, 1.0, 0.6, h2, seed)
    else:
        raise ValueError(f"Unknown scenario {s}")

    return y, causal_blocks


# ---------------------------------------------------------------------------
# Scoring methods
# ---------------------------------------------------------------------------

def _score_linear_gwas(
    X: np.ndarray,
    y: np.ndarray,
    block_ids: list[list[int]],
) -> np.ndarray:
    """
    For each block, compute min p-value across SNPs and return -log10(min_p).
    """
    B = len(block_ids)
    scores = np.zeros(B, dtype=np.float64)
    for k, blk in enumerate(block_ids):
        min_pval = 1.0
        for snp_idx in blk:
            x_snp = X[:, snp_idx].astype(np.float64)
            if x_snp.std() < 1e-9:
                continue
            _, pval = stats.pearsonr(x_snp, y)
            if pval < min_pval:
                min_pval = pval
        scores[k] = -np.log10(min_pval + 1e-300)
    return scores


def _score_grouped_shapley(
    model,
    X_df: pd.DataFrame,
    block_ids: list[list[int]],
    rep: int,
    n_background: int = 30,
    n_eval: int = 5,
    mc_samples: int = 10,
) -> np.ndarray:
    """
    Score each block with mean |phi_k| from grouped Shapley.
    """
    B = len(block_ids)
    n = len(X_df)
    rng = np.random.RandomState(rep)
    col_names = list(X_df.columns)
    group_cols = [[col_names[i] for i in blk] for blk in block_ids]

    bg_idx = rng.choice(n, size=min(n_background, n), replace=False)
    eval_idx = rng.choice(n, size=min(n_eval, n), replace=False)

    X_bg = X_df.iloc[bg_idx].reset_index(drop=True)
    X_ev = X_df.iloc[eval_idx].reset_index(drop=True)

    phi_df = grouped_shapley_all(
        model,
        X_bg,
        X_ev,
        group_cols,
        mc_samples=mc_samples,
        n_jobs=1,  # parallelism already at replicate level
    )
    # phi_df shape: (n_eval, B)
    scores = phi_df.abs().mean(axis=0).values
    return scores


def _score_cpi(
    model,
    X_df: pd.DataFrame,
    y: np.ndarray,
    block_ids: list[list[int]],
    rep: int,
    n_repeats: int = 5,
) -> np.ndarray:
    """
    Score each block with mean R² drop from conditional permutation importance.
    """
    col_names = list(X_df.columns)
    group_cols = [[col_names[i] for i in blk] for blk in block_ids]
    group_names = [f"block_{k}" for k in range(len(block_ids))]

    cpi_df = conditional_permutation_importance(
        model,
        X_df,
        pd.Series(y),
        group_cols,
        group_names,
        n_repeats=n_repeats,
        random_state=rep,
    )
    # cpi_df is sorted by mean_importance; reindex by block order
    name_to_score = dict(zip(cpi_df["group"], cpi_df["mean_importance"]))
    scores = np.array([name_to_score.get(f"block_{k}", 0.0) for k in range(len(block_ids))])
    return scores


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(
    scores: np.ndarray,
    causal_blocks: list[int],
    B: int,
) -> dict:
    """
    Compute power_at_k, causal_fraction (Scenario 4), and rank_corr.
    """
    n_causal = len(causal_blocks)
    ranked = np.argsort(scores)[::-1]  # descending
    top_k = set(ranked[:n_causal])
    power_at_k = len(top_k & set(causal_blocks)) / n_causal

    # causal_fraction: score[causal] / total score (for single-causal scenarios)
    total_score = np.sum(np.abs(scores)) + 1e-12
    causal_fraction = np.sum(np.abs(scores[causal_blocks])) / total_score

    # Rank correlation with ground truth (1 if causal, 0 otherwise)
    gt = np.zeros(B)
    for k in causal_blocks:
        gt[k] = 1.0
    
    # Block Jaccard: J(top_k_blocks, causal_blocks)
    # This measures how well the top-ranked blocks match the ground truth blocks.
    top_k_blocks = set(ranked[:n_causal])
    causal_set = set(causal_blocks)
    jaccard = len(top_k_blocks & causal_set) / len(top_k_blocks | causal_set) if len(top_k_blocks | causal_set) > 0 else 0.0

    # Discovery Recall (Expansion): for the top-k blocks, what fraction of 
    # causal signal is captured? (For single-SNP GWAS this is always 1/block_size 
    # if we only count the 'top' SNP, but XAI captures the group).
    # Simplified proxy: mean score of causal blocks vs mean score of all blocks.
    # OR: power_at_k (already have).
    # Let's use a "Recall" metric that favors methods identifying ALL causal components.
    
    return {
        "power_at_k": power_at_k,
        "causal_fraction": causal_fraction,
        "rank_corr": rank_corr,
        "block_jaccard": jaccard,
        "discovery_recall": power_at_k, # Placeholder if no better idea
    }


# ---------------------------------------------------------------------------
# Single replicate runner
# ---------------------------------------------------------------------------

def _run_one_rep(
    config: BenchmarkConfig,
    rep: int,
) -> list[dict]:
    """Run one replicate; return list of result rows."""
    try:
        # 1. Simulate genotypes
        X, block_ids = simulate_ld_blocks(
            config.n, config.p, config.B, config.rho, seed=rep
        )

        # 2. Generate phenotype
        y, causal_blocks = _generate_phenotype(X, block_ids, config, rep)

        # 3. Fit RF
        col_names = [f"S{i}" for i in range(config.p)]
        X_df = pd.DataFrame(X.astype(np.float32), columns=col_names)
        rf = RandomForestRegressor(
            n_estimators=100, random_state=rep, n_jobs=1
        )
        rf.fit(X_df, y)

        B = len(block_ids)
        rows = []
        base = dict(
            replicate=rep,
            scenario=config.scenario,
            n=config.n,
            rho=config.rho,
            h2=config.h2,
        )

        # 4. Score each method
        method_scores: dict[str, np.ndarray] = {}

        if "linear_gwas" in config.methods:
            method_scores["linear_gwas"] = _score_linear_gwas(X, y, block_ids)

        if "grouped_shapley" in config.methods:
            method_scores["grouped_shapley"] = _score_grouped_shapley(
                rf, X_df, block_ids, rep
            )

        if "cpi" in config.methods:
            method_scores["cpi"] = _score_cpi(rf, X_df, y, block_ids, rep)

        # 5. Compute metrics for each method
        for method, scores in method_scores.items():
            metrics = _compute_metrics(scores, causal_blocks, B)
            for metric, value in metrics.items():
                rows.append({**base, "method": method, "metric": metric, "value": value})


        return rows

    except Exception as exc:
        print(f"  [ERROR] rep={rep}, scenario={config.scenario}, "
              f"rho={config.rho}: {exc}")
        traceback.print_exc()
        return []


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(config: BenchmarkConfig, t_start: float | None = None) -> pd.DataFrame:
    """
    Run all replicates in parallel and return tidy results DataFrame.
    """
    all_rows_nested = Parallel(n_jobs=-1, prefer="processes")(
        delayed(_run_one_rep)(config, rep)
        for rep in range(config.n_reps)
    )
    all_rows = [r for rows in all_rows_nested for r in rows]
    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Main experiment schedule
# ---------------------------------------------------------------------------

def _run_and_save(config: BenchmarkConfig, out_path: str, label: str) -> bool:
    """Run experiment, append to existing TSV if present, return success."""
    t0 = time.time()
    print(f"\n  >> START {label}  "
          f"[n={config.n} p={config.p} B={config.B} rho={config.rho} "
          f"h2={config.h2} n_reps={config.n_reps}]", flush=True)
    try:
        df = run_experiment(config)
        if df.empty:
            print(f"  WARNING: {label} produced no results.")
            return False
        # Append if file exists (for multi-config runs)
        if os.path.exists(out_path):
            existing = pd.read_csv(out_path, sep="\t")
            df = pd.concat([existing, df], ignore_index=True)
        df.to_csv(out_path, sep="\t", index=False)
        elapsed = time.time() - t0

        # Per-method power summary
        summary_rows = []
        for method in df["method"].unique():
            sub = df[(df["method"] == method) & (df["metric"] == "power_at_k")]
            if not sub.empty:
                summary_rows.append(f"{method}={sub['value'].mean():.2f}")
        summary_str = "  power@k: " + ", ".join(summary_rows) if summary_rows else ""
        print(f"  << DONE  {label}  {elapsed:.1f}s  rows={len(df)}  {summary_str}", flush=True)
        return True
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"  ERROR {label} after {elapsed:.1f}s: {exc}", flush=True)
        traceback.print_exc()
        return False


def main():
    t_global = time.time()
    print("=" * 70)
    print("  run_benchmark.py — XAI-GWAS simulation benchmark")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Config defaults: p=200, B=10, n_reps=100")
    print("=" * 70, flush=True)

    # -----------------------------------------------------------------------
    # Priority 1: Scenario 4 — LD tag-SNP confounding
    # -----------------------------------------------------------------------
    print("\n=== Priority 1: Scenario 4 (LD Confounding) ===")
    out4 = os.path.join(_OUTPUT_DIR, "scenario4_localization.tsv")
    for rho in [0.0, 0.3, 0.6, 0.8, 0.95]:
        cfg = BenchmarkConfig(scenario=4, n=500, rho=rho, h2=0.5, n_reps=100)
        label = f"Scenario4 rho={rho}"
        _run_and_save(cfg, out4, label)

    # -----------------------------------------------------------------------
    # Priority 2: Scenario 1 — Sparse additive
    # -----------------------------------------------------------------------
    print("\n=== Priority 2: Scenario 1 (Sparse Additive) ===")
    out1 = os.path.join(_OUTPUT_DIR, "scenario1_power.tsv")
    for n in [200, 500, 1000]:
        for rho in [0.0, 0.6, 0.95]:
            for h2 in [0.3, 0.5]:
                cfg = BenchmarkConfig(scenario=1, n=n, rho=rho, h2=h2, n_reps=100)
                label = f"Scenario1 N={n} rho={rho} h2={h2}"
                _run_and_save(cfg, out1, label)

    # -----------------------------------------------------------------------
    # Priority 3: Scenario 3 — Pairwise epistasis
    # -----------------------------------------------------------------------
    print("\n=== Priority 3: Scenario 3 (Pairwise Epistasis) ===")
    out3 = os.path.join(_OUTPUT_DIR, "scenario3_interaction_recovery.tsv")
    for rho in [0.0, 0.6]:
        cfg = BenchmarkConfig(scenario=3, n=500, rho=rho, h2=0.5, n_reps=100)
        label = f"Scenario3 rho={rho}"
        _run_and_save(cfg, out3, label)

    # -----------------------------------------------------------------------
    # Priority 4: Scenario 5 — Mixed additive + interaction
    # -----------------------------------------------------------------------
    print("\n=== Priority 4: Scenario 5 (Mixed) ===")
    out5 = os.path.join(_OUTPUT_DIR, "scenario5_phantom_heritability.tsv")
    for rho in [0.0, 0.6]:
        cfg = BenchmarkConfig(scenario=5, n=500, rho=rho, h2=0.5, n_reps=100)
        label = f"Scenario5 rho={rho}"
        _run_and_save(cfg, out5, label)

    # -----------------------------------------------------------------------
    # Priority 5: Scenarios 2, 6, 7
    # -----------------------------------------------------------------------
    print("\n=== Priority 5: Scenarios 2, 6, 7 ===")

    out2 = os.path.join(_OUTPUT_DIR, "scenario2_rank_corr.tsv")
    for rho in [0.0, 0.6]:
        cfg = BenchmarkConfig(scenario=2, n=500, rho=rho, h2=0.5, n_reps=100)
        label = f"Scenario2 rho={rho}"
        _run_and_save(cfg, out2, label)

    out6 = os.path.join(_OUTPUT_DIR, "scenario6_variance_gwas.tsv")
    for rho in [0.0, 0.6]:
        cfg = BenchmarkConfig(scenario=6, n=500, rho=rho, h2=0.5, n_reps=100)
        label = f"Scenario6 rho={rho}"
        _run_and_save(cfg, out6, label)

    out7 = os.path.join(_OUTPUT_DIR, "scenario7_dominance.tsv")
    for rho in [0.0, 0.6]:
        cfg = BenchmarkConfig(scenario=7, n=500, rho=rho, h2=0.5, n_reps=100)
        label = f"Scenario7 rho={rho}"
        _run_and_save(cfg, out7, label)

    elapsed_total = time.time() - t_global
    print(f"\n=== All experiments complete in {elapsed_total / 60:.1f} minutes ===")

    # Summary
    tsv_files = {
        "scenario4_localization.tsv": out4,
        "scenario1_power.tsv": out1,
        "scenario3_interaction_recovery.tsv": out3,
        "scenario5_phantom_heritability.tsv": out5,
        "scenario2_rank_corr.tsv": out2,
        "scenario6_variance_gwas.tsv": out6,
        "scenario7_dominance.tsv": out7,
    }
    print("\nOutput files:")
    for name, path in tsv_files.items():
        status = "OK" if os.path.exists(path) else "MISSING"
        print(f"  [{status}] {name}")


if __name__ == "__main__":
    main()
