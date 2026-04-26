#!/usr/bin/env python3
"""
Phase 6 simulation study — FITTED RANDOM-FOREST VARIANT.

Sister script to run_phase6.py. Replaces the OracleModel shim (which
wrapped the data-generating function as the "model") with a fitted
RandomForestRegressor trained on each scenario's (X_train, y_train).
Required to test the research claim that grouped Shapley provides
better attributions than linear GWAS / per-SNP SHAP on a *learned*
nonlinear model — the oracle variant only verifies algorithmic
correctness of the Shapley arithmetic).

RF hyperparameters match scripts/run_benchmark.py Scenario 4:
    n_estimators=100, random_state=seed, n_jobs=-1
    (max_features default=1.0, min_samples_leaf default=1)

Train/test split: 80/20 via sklearn.train_test_split(random_state=seed).
RF is fitted on the train slice; grouped Shapley background and eval
points are sampled from the test slice.

Outputs (do NOT overwrite oracle baselines):
    scenario1_recovery_fitted.png / .csv
    scenario1_stability_fitted.png / .csv
    scenario3_recovery_fitted.png / .csv
    scenario3_stability_fitted.png / .csv
    scenario5_recovery_fitted.png / .csv
    scenario5_stability_fitted.png / .csv
    phase6_simulation_summary_fitted.csv
"""

from __future__ import annotations

import math
from itertools import chain, combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

OUTDIR = Path("outputs/phase6")
N_TRAIN = 2000
N_BACKGROUND = 100
N_EVAL = 20
N_GROUPS = 4
SNPS_PER_GROUP = 8
N_STABILITY_SEEDS = 15
FIGSIZE = (7, 4)

# Random-forest config matched to run_benchmark.py Scenario 4
RF_N_ESTIMATORS = 100
RF_N_JOBS = -1


def _fit_rf(X_train: pd.DataFrame, y_train: np.ndarray, seed: int) -> RandomForestRegressor:
    rf = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        random_state=seed,
        n_jobs=RF_N_JOBS,
    )
    rf.fit(X_train, y_train)
    return rf


# ─────────────────────────────────────────────────────────────────────────────
# Core utilities
# ─────────────────────────────────────────────────────────────────────────────

class OracleModel:
    """Wraps an analytic function as a sklearn-style predictor."""
    def __init__(self, fn):
        self._fn = fn

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._fn(X), dtype=float)


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def make_group_cols(n_groups: int = N_GROUPS, snps_per: int = SNPS_PER_GROUP) -> list[list[str]]:
    return [[f"G{g}_S{s}" for s in range(snps_per)] for g in range(n_groups)]


def generate_independent_genotypes(n: int, group_cols: list[list[str]]) -> pd.DataFrame:
    data = {}
    for cols in group_cols:
        for col in cols:
            data[col] = np.random.binomial(2, 0.3, size=n)
    return pd.DataFrame(data)


def value_function(model, X_background: pd.DataFrame, eval_x: pd.Series,
                   coalition: frozenset, group_cols: list[list[str]]) -> float:
    X_syn = X_background.copy()
    for j in coalition:
        for col in group_cols[j]:
            X_syn[col] = float(eval_x[col])
    return float(model.predict(X_syn).mean())


def all_value_functions(model, X_background: pd.DataFrame, eval_x: pd.Series,
                        group_cols: list[list[str]]) -> dict[frozenset, float]:
    m = len(group_cols)
    return {
        frozenset(S): value_function(model, X_background, eval_x, frozenset(S), group_cols)
        for S in powerset(range(m))
    }


def grouped_shapley_single(v_cache: dict[frozenset, float], k: int, m: int) -> float:
    total = 0.0
    for S in powerset(range(m)):
        if k in S:
            continue
        s = len(S)
        w = math.factorial(s) * math.factorial(m - s - 1) / math.factorial(m)
        total += w * (v_cache[frozenset(S) | frozenset([k])] - v_cache[frozenset(S)])
    return total


def grouped_shapley_interaction(v_cache: dict[frozenset, float],
                                j: int, k: int, m: int) -> float:
    rest = [g for g in range(m) if g not in (j, k)]
    total = 0.0
    for S in powerset(rest):
        s = len(S)
        w = math.factorial(s) * math.factorial(m - s - 2) / math.factorial(m - 1)
        S_jk = frozenset(S) | frozenset([j, k])
        S_j  = frozenset(S) | frozenset([j])
        S_k  = frozenset(S) | frozenset([k])
        S_b  = frozenset(S)
        total += w * (v_cache[S_jk] - v_cache[S_j] - v_cache[S_k] + v_cache[S_b])
    return total


def compute_phi_scores(model, X: pd.DataFrame, group_cols: list[list[str]],
                       seed: int = 0) -> np.ndarray:
    """Return mean |φ_k| for each group over N_EVAL evaluation points."""
    m = len(group_cols)
    X_bg = X.sample(N_BACKGROUND, random_state=seed).reset_index(drop=True)
    eval_pts = X.sample(N_EVAL, random_state=seed + 1000).reset_index(drop=True)
    phi_acc = np.zeros(m)
    for _, eval_x in eval_pts.iterrows():
        v = all_value_functions(model, X_bg, eval_x, group_cols)
        for k in range(m):
            phi_acc[k] += abs(grouped_shapley_single(v, k, m))
    return phi_acc / len(eval_pts)


def compute_interaction_scores(model, X: pd.DataFrame, group_cols: list[list[str]],
                               seed: int = 0) -> dict[tuple[int, int], float]:
    """Return mean |I_{jk}| for all group pairs."""
    m = len(group_cols)
    X_bg = X.sample(N_BACKGROUND, random_state=seed).reset_index(drop=True)
    eval_pts = X.sample(N_EVAL, random_state=seed + 1000).reset_index(drop=True)
    pair_acc: dict[tuple[int, int], float] = {}
    for j in range(m):
        for k in range(j + 1, m):
            pair_acc[(j, k)] = 0.0
    for _, eval_x in eval_pts.iterrows():
        v = all_value_functions(model, X_bg, eval_x, group_cols)
        for j in range(m):
            for k in range(j + 1, m):
                pair_acc[(j, k)] += abs(grouped_shapley_interaction(v, j, k, m))
    return {pair: val / len(eval_pts) for pair, val in pair_acc.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1 — Additive Sparse
# ─────────────────────────────────────────────────────────────────────────────

def _scenario1_oracle(Xd: pd.DataFrame) -> np.ndarray:
    g0 = Xd[[f"G0_S{s}" for s in range(SNPS_PER_GROUP)]].sum(axis=1).values.astype(float)
    g1 = Xd[[f"G1_S{s}" for s in range(SNPS_PER_GROUP)]].sum(axis=1).values.astype(float)
    return 2.0 * g0 + 1.5 * g1   # G2, G3 are structural nulls


def run_scenario1_recovery() -> pd.DataFrame:
    """
    Recovery: vary noise sigma.
    Compare grouped Shapley SNR (causal/null) vs. linear GWAS SNR.
    Truth: f = 2*G0 + 1.5*G1; G2, G3 null. Model: fitted RandomForest.
    """
    group_cols = make_group_cols()
    m = len(group_cols)
    causal_idx = [0, 1]
    null_idx   = [2, 3]
    sigmas = np.array([0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0])
    rows = []

    for sigma in sigmas:
        np.random.seed(42)
        X = generate_independent_genotypes(N_TRAIN, group_cols)
        y_true = _scenario1_oracle(X)
        y = y_true + np.random.normal(0, float(sigma), N_TRAIN)

        X_train, X_test, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        rf = _fit_rf(X_train, y_train, seed=42)

        # Linear GWAS: |corr(y, G_k_sum)| on the full sample
        g_sums = np.array([
            X[[f"G{k}_S{s}" for s in range(SNPS_PER_GROUP)]].sum(axis=1).values.astype(float)
            for k in range(m)
        ])
        lin_scores = np.array([abs(float(np.corrcoef(g_sums[k], y)[0, 1])) for k in range(m)])
        lin_snr = lin_scores[causal_idx].mean() / max(lin_scores[null_idx].mean(), 1e-9)

        # Grouped Shapley on fitted RF, evaluated on held-out X_test
        phi = compute_phi_scores(rf, X_test, group_cols, seed=0)
        shap_snr = phi[causal_idx].mean() / max(phi[null_idx].mean(), 1e-9)

        rows.append({"sigma": float(sigma), "linear_snr": lin_snr, "shap_snr": shap_snr})

    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "scenario1_recovery_fitted.csv", index=False)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(df["sigma"], df["linear_snr"], "o-", color="#d62728", label="Linear GWAS SNR")
    ax.plot(df["sigma"], df["shap_snr"],   "s-", color="#1f77b4", label="Grouped Shapley SNR")
    ax.axhline(1.0, ls="--", color="gray", lw=0.8, label="SNR = 1 (no signal)")
    ax.set_xlabel("Noise σ")
    ax.set_ylabel("Signal-to-Null Ratio")
    ax.set_title("Scenario 1 — Additive Sparse: Recovery vs. Noise Level (fitted RF)")
    ax.legend()
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(OUTDIR / "scenario1_recovery_fitted.png", dpi=150)
    plt.close(fig)
    print(f"  Scenario 1 recovery saved  (σ ∈ [{sigmas[0]}, {sigmas[-1]}])")
    return df


def run_scenario1_stability() -> pd.DataFrame:
    """
    Stability: fix sigma=0.5, sweep N_STABILITY_SEEDS seeds.
    Boxplot of mean |φ_k| per group: causal (G0, G1) vs. null (G2, G3).
    Re-fits RF every seed so variance reflects data resampling AND fit randomness.
    """
    group_cols = make_group_cols()
    m = len(group_cols)
    sigma = 0.5
    phi_records: dict[str, list[float]] = {f"G{k}": [] for k in range(m)}

    for seed in range(N_STABILITY_SEEDS):
        np.random.seed(seed)
        X = generate_independent_genotypes(N_TRAIN, group_cols)
        y = _scenario1_oracle(X) + np.random.normal(0, sigma, N_TRAIN)

        X_train, X_test, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        rf = _fit_rf(X_train, y_train, seed=seed)

        phi = compute_phi_scores(rf, X_test, group_cols, seed=seed)
        for k in range(m):
            phi_records[f"G{k}"].append(float(phi[k]))

    df = pd.DataFrame(phi_records)
    df.to_csv(OUTDIR / "scenario1_stability_fitted.csv", index=False)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    colors = ["#1f77b4", "#1f77b4", "#aec7e8", "#aec7e8"]
    labels = ["G0 (causal)", "G1 (causal)", "G2 (null)", "G3 (null)"]
    bp = ax.boxplot([df[f"G{k}"].values for k in range(m)], patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_xticklabels(labels, rotation=12)
    ax.set_ylabel("Mean |φ_k|")
    ax.set_title(f"Scenario 1 — Additive Sparse: Stability (σ=0.5, {N_STABILITY_SEEDS} seeds, fitted RF)")
    fig.tight_layout()
    fig.savefig(OUTDIR / "scenario1_stability_fitted.png", dpi=150)
    plt.close(fig)
    print(f"  Scenario 1 stability saved ({N_STABILITY_SEEDS} seeds)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 3 — Pairwise Epistasis
# ─────────────────────────────────────────────────────────────────────────────

def _scenario3_oracle_factory(mu0: float, mu1: float, b: float):
    def _fn(Xd: pd.DataFrame) -> np.ndarray:
        g0 = Xd[[f"G0_S{s}" for s in range(SNPS_PER_GROUP)]].sum(axis=1).values.astype(float)
        g1 = Xd[[f"G1_S{s}" for s in range(SNPS_PER_GROUP)]].sum(axis=1).values.astype(float)
        g2 = Xd[[f"G2_S{s}" for s in range(SNPS_PER_GROUP)]].sum(axis=1).values.astype(float)
        return 2.0 * g0 + 1.5 * g1 + b * (g0 - mu0) * (g1 - mu1) + 0.5 * g2
    return _fn


_SCEN3_NOISE_SIGMA = 0.5


def run_scenario3_recovery() -> pd.DataFrame:
    """
    Recovery: vary interaction strength b_{01}.
    Track I_{01} (true epistatic pair) vs. max null-pair interaction.
    Truth: f = 2*G0 + 1.5*G1 + b*(G0-μ0)*(G1-μ1) + 0.5*G2.
    Model: fitted RandomForest on (X, oracle(X) + N(0, 0.5)).
    """
    group_cols = make_group_cols()
    b_values = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
    rows = []

    for b in b_values:
        np.random.seed(42)
        X = generate_independent_genotypes(N_TRAIN, group_cols)
        mu0 = float(X[[f"G0_S{s}" for s in range(SNPS_PER_GROUP)]].sum(axis=1).mean())
        mu1 = float(X[[f"G1_S{s}" for s in range(SNPS_PER_GROUP)]].sum(axis=1).mean())
        oracle_fn = _scenario3_oracle_factory(mu0, mu1, float(b))
        y = oracle_fn(X) + np.random.normal(0, _SCEN3_NOISE_SIGMA, N_TRAIN)

        X_train, X_test, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        rf = _fit_rf(X_train, y_train, seed=42)

        pairs = compute_interaction_scores(rf, X_test, group_cols, seed=0)
        I01 = pairs[(0, 1)]
        null_max = max(pairs[(0, 2)], pairs[(1, 2)], pairs[(0, 3)], pairs[(1, 3)], pairs[(2, 3)])
        rows.append({"b_01": float(b), "I_01": I01, "null_max": null_max})

    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "scenario3_recovery_fitted.csv", index=False)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(df["b_01"], df["I_01"],    "s-", color="#2ca02c", label="|I₀₁| (epistatic pair)")
    ax.plot(df["b_01"], df["null_max"],"^--", color="#d62728", label="max|I_null| (non-epistatic pairs)")
    ax.set_xlabel("Interaction strength b₀₁")
    ax.set_ylabel("Grouped Shapley Interaction Index")
    ax.set_title("Scenario 3 — Pairwise Epistasis: Recovery (fitted RF)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "scenario3_recovery_fitted.png", dpi=150)
    plt.close(fig)
    print(f"  Scenario 3 recovery saved  (b ∈ [{b_values[0]}, {b_values[-1]}])")
    return df


def run_scenario3_stability() -> pd.DataFrame:
    """
    Stability: fix b=0.3, sweep N_STABILITY_SEEDS seeds.
    Boxplot of I_{01} (epistatic pair) vs. I_{02}, I_{12} (null pairs).
    Re-fits RF every seed so variance reflects data resampling AND fit randomness.
    """
    group_cols = make_group_cols()
    b = 0.3
    I01_list, I02_list, I12_list = [], [], []

    for seed in range(N_STABILITY_SEEDS):
        np.random.seed(seed)
        X = generate_independent_genotypes(N_TRAIN, group_cols)
        mu0 = float(X[[f"G0_S{s}" for s in range(SNPS_PER_GROUP)]].sum(axis=1).mean())
        mu1 = float(X[[f"G1_S{s}" for s in range(SNPS_PER_GROUP)]].sum(axis=1).mean())
        oracle_fn = _scenario3_oracle_factory(mu0, mu1, b)
        y = oracle_fn(X) + np.random.normal(0, _SCEN3_NOISE_SIGMA, N_TRAIN)

        X_train, X_test, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        rf = _fit_rf(X_train, y_train, seed=seed)

        pairs = compute_interaction_scores(rf, X_test, group_cols, seed=seed)
        I01_list.append(pairs[(0, 1)])
        I02_list.append(pairs[(0, 2)])
        I12_list.append(pairs[(1, 2)])

    df = pd.DataFrame({"I_01": I01_list, "I_02": I02_list, "I_12": I12_list})
    df.to_csv(OUTDIR / "scenario3_stability_fitted.csv", index=False)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bp = ax.boxplot([df["I_01"].values, df["I_02"].values, df["I_12"].values],
                    patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], ["#2ca02c", "#d62728", "#d62728"]):
        patch.set_facecolor(color)
    ax.set_xticklabels(["I₀₁ (epistatic)", "I₀₂ (null)", "I₁₂ (null)"])
    ax.set_ylabel("Grouped Shapley Interaction Index")
    ax.set_title(f"Scenario 3 — Pairwise Epistasis: Stability (b=0.3, {N_STABILITY_SEEDS} seeds, fitted RF)")
    fig.tight_layout()
    fig.savefig(OUTDIR / "scenario3_stability_fitted.png", dpi=150)
    plt.close(fig)
    print(f"  Scenario 3 stability saved ({N_STABILITY_SEEDS} seeds)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 5 — LD Proxy-Causal
# ─────────────────────────────────────────────────────────────────────────────

def _make_ld_dataframe(n: int, rho: float, seed: int) -> pd.DataFrame:
    """
    Build a 2-group genotype DataFrame where G0 (causal) and G1 (LD proxy)
    have group-mean correlation ~rho.
    """
    np.random.seed(seed)
    z = np.random.normal(0, 1, n)
    e1 = np.random.normal(0, 1, n)
    e2 = np.random.normal(0, 1, n)
    x1 = z + e1 * 0.1                             # causal latent
    x2 = rho * z + np.sqrt(max(1 - rho**2, 1e-9)) * e2   # proxy latent
    # Replicate across SNPS_PER_GROUP with tiny per-SNP jitter
    data = {}
    for s in range(SNPS_PER_GROUP):
        data[f"G0_S{s}"] = x1 + np.random.normal(0, 0.05, n)
        data[f"G1_S{s}"] = x2 + np.random.normal(0, 0.05, n)
    return pd.DataFrame(data)


_GRP_LD = [[f"G0_S{s}" for s in range(SNPS_PER_GROUP)],
           [f"G1_S{s}" for s in range(SNPS_PER_GROUP)]]
_BETA_TRUE = 10.0


def _scenario5_oracle(Xd: pd.DataFrame) -> np.ndarray:
    return _BETA_TRUE * Xd[[f"G0_S{s}" for s in range(SNPS_PER_GROUP)]].mean(axis=1).values


def run_scenario5_recovery() -> pd.DataFrame:
    """
    Recovery: vary LD correlation rho.
    - Linear smearing: |β_proxy_marginal| / |β_causal_marginal|
    - Grouped Shapley isolation: φ_causal / (φ_causal + φ_proxy)
    Truth: f = 10 * mean(G0_SNPs); G1 is pure LD proxy.
    Model: fitted RandomForest on (X_ld, y_train).
    """
    rhos = np.array([0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95])
    rows = []

    for rho in rhos:
        X_ld = _make_ld_dataframe(N_TRAIN, float(rho), seed=42)
        y = _BETA_TRUE * X_ld[[f"G0_S{s}" for s in range(SNPS_PER_GROUP)]].mean(axis=1).values
        y += np.random.RandomState(42).normal(0, 0.5, N_TRAIN)

        g0_mean = X_ld[[f"G0_S{s}" for s in range(SNPS_PER_GROUP)]].mean(axis=1).values
        g1_mean = X_ld[[f"G1_S{s}" for s in range(SNPS_PER_GROUP)]].mean(axis=1).values
        beta_causal = LinearRegression().fit(g0_mean.reshape(-1, 1), y).coef_[0]
        beta_proxy  = LinearRegression().fit(g1_mean.reshape(-1, 1), y).coef_[0]
        smear = abs(beta_proxy) / max(abs(beta_causal), 1e-9)

        X_train, X_test, y_train, _ = train_test_split(
            X_ld, y, test_size=0.2, random_state=42
        )
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        rf = _fit_rf(X_train, y_train, seed=42)

        phi = compute_phi_scores(rf, X_test, _GRP_LD, seed=0)
        phi_frac = float(phi[0]) / max(float(phi[0]) + float(phi[1]), 1e-9)

        rows.append({"rho": float(rho), "linear_smear": smear, "phi_causal_frac": phi_frac})

    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "scenario5_recovery_fitted.csv", index=False)

    fig, ax1 = plt.subplots(figsize=FIGSIZE)
    ax2 = ax1.twinx()
    l1, = ax1.plot(df["rho"], df["linear_smear"],    "o-", color="#d62728",
                   label="Linear smearing |β_proxy| / |β_causal|")
    l2, = ax2.plot(df["rho"], df["phi_causal_frac"], "s-", color="#1f77b4",
                   label="Grouped Shapley φ_causal fraction")
    ax1.set_xlabel("LD correlation ρ")
    ax1.set_ylabel("Smearing ratio (linear GWAS)", color="#d62728")
    ax2.set_ylabel("φ_causal / (φ_causal + φ_proxy)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.set_ylim(0, 1.05)
    ax1.set_title("Scenario 5 — LD Proxy-Causal: Recovery vs. LD Correlation (fitted RF)")
    ax1.legend(handles=[l1, l2], loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTDIR / "scenario5_recovery_fitted.png", dpi=150)
    plt.close(fig)
    print(f"  Scenario 5 recovery saved  (ρ ∈ [{rhos[0]}, {rhos[-1]}])")
    return df


def run_scenario5_stability() -> pd.DataFrame:
    """
    Stability: fix rho=0.8, sweep N_STABILITY_SEEDS seeds.
    Boxplot of φ_causal / (φ_causal + φ_proxy).
    Re-fits RF every seed so variance reflects data resampling AND fit randomness.
    """
    rho = 0.8
    phi_frac_list = []

    for seed in range(N_STABILITY_SEEDS):
        X_ld = _make_ld_dataframe(N_TRAIN, rho, seed=seed)
        y = _BETA_TRUE * X_ld[[f"G0_S{s}" for s in range(SNPS_PER_GROUP)]].mean(axis=1).values
        y += np.random.RandomState(seed).normal(0, 0.5, N_TRAIN)

        X_train, X_test, y_train, _ = train_test_split(
            X_ld, y, test_size=0.2, random_state=seed
        )
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        rf = _fit_rf(X_train, y_train, seed=seed)

        phi = compute_phi_scores(rf, X_test, _GRP_LD, seed=seed)
        phi_frac_list.append(float(phi[0]) / max(float(phi[0]) + float(phi[1]), 1e-9))

    df = pd.DataFrame({"phi_causal_frac": phi_frac_list})
    df.to_csv(OUTDIR / "scenario5_stability_fitted.csv", index=False)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bp = ax.boxplot([df["phi_causal_frac"].values], patch_artist=True, widths=0.4)
    bp["boxes"][0].set_facecolor("#1f77b4")
    ax.axhline(1.0, ls="--", color="green", lw=0.9, label="Perfect isolation (=1.0)")
    ax.axhline(0.5, ls="--", color="gray",  lw=0.9, label="Uninformative (=0.5)")
    ax.set_xticks([1])
    ax.set_xticklabels(["φ_causal / (φ_causal + φ_proxy)"])
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Scenario 5 — LD Proxy-Causal: Stability (ρ=0.8, {N_STABILITY_SEEDS} seeds, fitted RF)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "scenario5_stability_fitted.png", dpi=150)
    plt.close(fig)
    print(f"  Scenario 5 stability saved ({N_STABILITY_SEEDS} seeds)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("Phase 6 simulation study — FITTED RF VARIANT")
    print("=" * 50)

    summary_rows = []

    print("\nScenario 1: Additive Sparse")
    df1r = run_scenario1_recovery()
    df1s = run_scenario1_stability()
    summary_rows.append({
        "scenario": 1,
        "name": "additive_sparse",
        "shap_snr_min": float(df1r["shap_snr"].min()),
        "shap_snr_max": float(df1r["shap_snr"].max()),
        "phi_causal_mean": float(df1s[["G0", "G1"]].values.mean()),
        "phi_null_mean":   float(df1s[["G2", "G3"]].values.mean()),
    })

    print("\nScenario 3: Pairwise Epistasis")
    df3r = run_scenario3_recovery()
    df3s = run_scenario3_stability()
    summary_rows.append({
        "scenario": 3,
        "name": "pairwise_epistasis",
        "I01_max": float(df3r["I_01"].max()),
        "null_max_max": float(df3r["null_max"].max()),
        "stability_I01_mean": float(df3s["I_01"].mean()),
        "stability_null_mean": float(df3s[["I_02", "I_12"]].values.mean()),
    })

    print("\nScenario 5: LD Proxy-Causal")
    df5r = run_scenario5_recovery()
    df5s = run_scenario5_stability()
    rho08_row = df5r[df5r["rho"].round(2) == 0.8]
    phi_frac_rho08 = float(rho08_row["phi_causal_frac"].values[0]) if len(rho08_row) else float("nan")
    summary_rows.append({
        "scenario": 5,
        "name": "ld_proxy_causal",
        "phi_causal_frac_at_rho08": phi_frac_rho08,
        "stability_phi_frac_mean":  float(df5s["phi_causal_frac"].mean()),
        "stability_phi_frac_std":   float(df5s["phi_causal_frac"].std()),
    })

    pd.DataFrame(summary_rows).to_csv(OUTDIR / "phase6_simulation_summary_fitted.csv", index=False)

    print("\n" + "=" * 50)
    print(f"All outputs saved to {OUTDIR}/")
    for p in sorted(OUTDIR.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
