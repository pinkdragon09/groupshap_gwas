"""Grouped Shapley attribution and related estimators.

Core estimators
---------------
- grouped_shapley_all()             — main entry point; returns (n_eval, m_groups) DataFrame
- grouped_interaction_all()         — pairwise Shapley interaction indices
- summarize_grouped_shapley()       — column-level mean/sd summary
- conditional_permutation_importance() — CPI for grouped features
- ale_effect()                      — first-order ALE for a single feature

Low-level building blocks
--------------------------
- value_function_numpy()            — interventional v(S; x)
- grouped_shapley_single_mc()       — φ_k for one eval point
- grouped_shapley_interaction_single_mc() — I_{jk} for one eval point

Properties verified by the simulation checks
--------------------------------------------
- grouped_shapley_all() under a group-additive predictor recovers the
  exact group-additive effects.
- grouped_shapley_interaction_single_mc() vanishes under additive
  predictors and recovers pairwise epistasis support under a KAN with
  activated cross-group edges.
- conditional_permutation_importance() returns zero for conditionally
  null groups.
- ale_effect() recovers the centred additive component under feature
  correlation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


# ---------------------------------------------------------------------------
# Interventional value function
# ---------------------------------------------------------------------------

def value_function_numpy(
    model,
    background_arr: np.ndarray,
    eval_x_arr: np.ndarray,
    coalition: frozenset,
    group_indices: list[list[int]],
    feature_names: list[str],
) -> float:
    """Interventional value function v(S; x).

    Groups in coalition S are fixed to eval_x values; remaining groups are
    marginalised over background_arr rows.  Converts to DataFrame before
    predict() so column-name-aware wrappers work correctly.

    Parameters
    ----------
    model          : sklearn-compatible regressor with predict(DataFrame).
    background_arr : (n_bg, n_features) background dataset array.
    eval_x_arr     : 1-D array of length n_features (single eval point).
    coalition      : frozenset of group indices in S.
    group_indices  : group_indices[j] = list of column indices for group j.
    feature_names  : column names corresponding to background_arr columns.
    """
    assert eval_x_arr.ndim == 1, (
        f"eval_x_arr must be 1-D, got shape {eval_x_arr.shape}"
    )
    X_syn = background_arr.copy()
    for j in coalition:
        X_syn[:, group_indices[j]] = eval_x_arr[group_indices[j]]
    df_syn = pd.DataFrame(X_syn, columns=feature_names)
    return float(model.predict(df_syn).mean())


# ---------------------------------------------------------------------------
# Single-point Monte Carlo estimators
# ---------------------------------------------------------------------------

def grouped_shapley_single_mc(
    model,
    background_arr: np.ndarray,
    eval_x_arr: np.ndarray,
    k: int,
    m: int,
    group_indices: list[list[int]],
    feature_names: list[str],
    n_samples: int = 50,
    rng: np.random.Generator | np.random.RandomState | None = None,
) -> float:
    """Monte Carlo estimate of grouped Shapley value φ_k for one eval point.

    Samples random permutations of all m groups; S = groups appearing before k.
    Reproduces the Shapley weight |S|!(m−|S|−1)!/m! exactly in expectation.
    """
    groups = list(range(m))
    permute = rng.permutation if rng is not None else np.random.permutation
    total = 0.0
    for _ in range(n_samples):
        perm = permute(groups)
        k_idx = int(np.where(perm == k)[0][0])
        S = frozenset(perm[:k_idx])
        v_with = value_function_numpy(model, background_arr, eval_x_arr, S | {k}, group_indices, feature_names)
        v_without = value_function_numpy(model, background_arr, eval_x_arr, S, group_indices, feature_names)
        total += v_with - v_without
    return total / n_samples


def grouped_shapley_interaction_single_mc(
    model,
    background_arr: np.ndarray,
    eval_x_arr: np.ndarray,
    j: int,
    k: int,
    m: int,
    group_indices: list[list[int]],
    feature_names: list[str],
    n_samples: int = 100,
    rng: np.random.Generator | np.random.RandomState | None = None,
) -> float:
    """Monte Carlo estimate of the Shapley Interaction Index I_{jk}.

        I_{jk} = Σ_{S ⊆ M\\{j,k}} w(s,m) [v(S∪{j,k}) − v(S∪{j}) − v(S∪{k}) + v(S)]
        w(s,m) = s!(m−s−2)! / (m−1)!

    Sampling is unbiased: draw s uniformly from {0,...,m−2}, then draw a
    random permutation of M\\{j,k} and take the first s elements as S.
    This recovers the Shapley weight exactly in expectation.
    """
    rest = [g for g in range(m) if g not in (j, k)]
    permute = rng.permutation if rng is not None else np.random.permutation
    if isinstance(rng, np.random.Generator):
        randint = lambda lo, hi: rng.integers(lo, hi)
    elif rng is not None:
        randint = lambda lo, hi: rng.randint(lo, hi)
    else:
        randint = lambda lo, hi: np.random.randint(lo, hi)

    total = 0.0
    for _ in range(n_samples):
        if rest:
            perm = permute(rest)
            s_size = randint(0, len(rest) + 1)
            S = frozenset(perm[:s_size])
        else:
            S = frozenset()
        v_jk = value_function_numpy(model, background_arr, eval_x_arr, S | {j, k}, group_indices, feature_names)
        v_j  = value_function_numpy(model, background_arr, eval_x_arr, S | {j},     group_indices, feature_names)
        v_k  = value_function_numpy(model, background_arr, eval_x_arr, S | {k},     group_indices, feature_names)
        v_0  = value_function_numpy(model, background_arr, eval_x_arr, S,           group_indices, feature_names)
        total += v_jk - v_j - v_k + v_0
    return total / n_samples


# ---------------------------------------------------------------------------
# Batch API
# ---------------------------------------------------------------------------

def _explain_one_point(
    model,
    background_arr: np.ndarray,
    eval_row: np.ndarray,
    m: int,
    group_indices: list[list[int]],
    feature_names: list[str],
    mc_samples: int,
    rng,
) -> list[float]:
    return [
        grouped_shapley_single_mc(
            model, background_arr, eval_row, k, m, group_indices, feature_names, mc_samples, rng=rng
        )
        for k in range(m)
    ]


def grouped_shapley_all(
    model,
    X_background: pd.DataFrame,
    X_eval: pd.DataFrame,
    group_cols: list[list[str]],
    group_names: list[str] | None = None,
    mc_samples: int = 50,
    n_jobs: int = -1,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Compute grouped Shapley values for all eval points via Monte Carlo.

    Parameters
    ----------
    model        : fitted sklearn-compatible regressor.
    X_background : Background dataset for marginalisation.
    X_eval       : Points at which to explain the model.
    group_cols   : group_cols[j] = list of column names in group j.
    group_names  : Optional group labels (defaults to 'group_0', 'group_1', ...).
    mc_samples   : Number of random permutations per (eval point, group).
    n_jobs       : Joblib parallelism (-1 = all cores).
    random_state : Seed for reproducibility (None = non-deterministic).

    Returns
    -------
    DataFrame of shape (n_eval, m_groups) containing φ_k values.
    """
    m = len(group_cols)
    feature_names = list(X_background.columns)
    col_to_idx = {name: i for i, name in enumerate(feature_names)}
    group_indices = [[col_to_idx[c] for c in g] for g in group_cols]
    background_arr = X_background.values
    eval_arr = X_eval.values
    cols = group_names if group_names is not None else [f"group_{k}" for k in range(m)]

    rngs = (
        [None] * len(eval_arr)
        if random_state is None
        else [np.random.default_rng(random_state + i) for i in range(len(eval_arr))]
    )
    rows = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_explain_one_point)(
            model, background_arr, eval_arr[i], m, group_indices, feature_names, mc_samples, rngs[i]
        )
        for i in range(len(eval_arr))
    )
    return pd.DataFrame(rows, columns=cols)


def grouped_interaction_all(
    model,
    X_background: pd.DataFrame,
    X_eval: pd.DataFrame,
    group_cols: list[list[str]],
    group_names: list[str],
    mc_samples: int = 50,
    n_jobs: int = -1,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Compute all pairwise Shapley Interaction Indices between groups.

    Returns DataFrame sorted descending by mean_abs_interaction with columns:
    group_j, group_k, mean_interaction, mean_abs_interaction, sd_interaction.
    """
    m = len(group_cols)
    feature_names = list(X_background.columns)
    col_to_idx = {name: i for i, name in enumerate(feature_names)}
    group_indices = [[col_to_idx[c] for c in g] for g in group_cols]
    background_arr = X_background.values
    eval_arr = X_eval.values

    def _pair_interactions(j: int, k: int) -> dict:
        base_seed = None if random_state is None else random_state + (j * m + k) * len(eval_arr)
        pair_rngs = (
            [None] * len(eval_arr)
            if base_seed is None
            else [np.random.default_rng(base_seed + i) for i in range(len(eval_arr))]
        )
        vals = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(grouped_shapley_interaction_single_mc)(
                model, background_arr, eval_arr[i], j, k, m, group_indices, feature_names, mc_samples, rng=pair_rngs[i]
            )
            for i in range(len(eval_arr))
        )
        return {
            "group_j":            group_names[j],
            "group_k":            group_names[k],
            "mean_interaction":     float(np.mean(vals)),
            "mean_abs_interaction": float(np.mean(np.abs(vals))),
            "sd_interaction":       float(np.std(vals)),
        }

    results = [_pair_interactions(j, k) for j in range(m) for k in range(j + 1, m)]
    return pd.DataFrame(results).sort_values("mean_abs_interaction", ascending=False)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize_grouped_shapley(
    phi_df: pd.DataFrame,
    group_names: list[str] | None = None,
) -> pd.DataFrame:
    """Column-name-driven, NaN-aware summary of grouped Shapley values.

    Parameters
    ----------
    phi_df      : Output of grouped_shapley_all() — (n_eval, m_groups).
    group_names : Accepted for backwards compatibility; ignored when phi_df
                  has named string columns.

    Returns
    -------
    DataFrame with columns: group, mean_phi, mean_abs_phi, sd_abs_phi,
    n_eval_points — sorted descending by mean_abs_phi.

    Note
    ----
    If the same group label refers to different SNP sets across folds (e.g.
    LD blocks computed on different training splits), averaging across folds
    is not semantically valid.  Use per-fold summaries in that case.
    """
    numeric_df = phi_df.select_dtypes(include=[np.number])
    if "fold" in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=["fold"])

    cols = list(numeric_df.columns)
    if group_names is not None and all(isinstance(c, (int, np.integer)) for c in cols):
        rename = {c: group_names[i] for i, c in enumerate(cols) if i < len(group_names)}
        numeric_df = numeric_df.rename(columns=rename)
        cols = list(numeric_df.columns)

    rows = []
    for col in cols:
        vals = numeric_df[col].to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            continue
        rows.append({
            "group":         col,
            "mean_phi":      float(np.mean(finite)),
            "mean_abs_phi":  float(np.mean(np.abs(finite))),
            "sd_abs_phi":    float(np.std(np.abs(finite))),
            "n_eval_points": int(finite.size),
        })
    return pd.DataFrame(rows).sort_values("mean_abs_phi", ascending=False)


# ---------------------------------------------------------------------------
# Conditional Permutation Importance (CPI)
# ---------------------------------------------------------------------------

def conditional_permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    group_cols: list[list[str]],
    group_names: list[str],
    n_repeats: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """Conditional Permutation Importance for grouped features.

    Permutes one group at a time (leaving all others fixed) and measures the
    drop in R².  Repeating n_repeats times gives stable mean and SD.

    Returns
    -------
    DataFrame with columns: group, mean_importance, sd_importance — sorted
    descending by mean_importance.
    """
    from sklearn.metrics import r2_score

    rng = np.random.RandomState(random_state)
    y_arr = np.asarray(y)
    baseline_r2 = r2_score(y_arr, model.predict(X))

    records = []
    for name, cols in zip(group_names, group_cols):
        if not cols:
            continue
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            perm_idx = rng.permutation(len(X))
            X_perm[cols] = X[cols].values[perm_idx]
            drops.append(baseline_r2 - r2_score(y_arr, model.predict(X_perm)))
        records.append({
            "group": name,
            "mean_importance": float(np.mean(drops)),
            "sd_importance":   float(np.std(drops)),
        })

    return (
        pd.DataFrame(records)
        .sort_values("mean_importance", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Accumulated Local Effects (ALE)
# ---------------------------------------------------------------------------

from xai_gwas.attribution.ale import ale_effect
