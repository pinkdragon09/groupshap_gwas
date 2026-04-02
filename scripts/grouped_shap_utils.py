"""
Cooperative-game grouped Shapley utilities.
Optimized for high-dimensional SNP data using NumPy.
"""

from __future__ import annotations

import math
from itertools import chain, combinations

import numpy as np
import pandas as pd


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def value_function_numpy(
    model,
    background_arr: np.ndarray,
    eval_x_arr: np.ndarray,
    coalition: frozenset,
    group_indices: list[list[int]],
    feature_names: list[str],
) -> float:
    """
    Interventional value function v(S; x).
    Groups in coalition S are fixed to eval_x_arr values; remaining groups
    are marginalised over background_arr rows.
    Converts to DataFrame before predict() so pipeline models and column-name-
    aware wrappers (e.g. SklearnWrapperForGroupedShap) work correctly.
    """
    assert eval_x_arr.ndim == 1, (
        f"eval_x_arr must be 1-D (a single row), got shape {eval_x_arr.shape}"
    )
    X_syn = background_arr.copy()
    for j in coalition:
        indices = group_indices[j]
        X_syn[:, indices] = eval_x_arr[indices]
    df_syn = pd.DataFrame(X_syn, columns=feature_names)
    return float(model.predict(df_syn).mean())


def grouped_shapley_single_mc(
    model,
    background_arr: np.ndarray,
    eval_x_arr: np.ndarray,
    k: int,
    m: int,
    group_indices: list[list[int]],
    feature_names: list[str],
    n_samples: int = 50,
) -> float:
    """
    Monte Carlo estimate of grouped Shapley φ_k.
    Samples random permutations of all m groups; S = groups appearing before k.
    This reproduces the Shapley weight |S|!(m-|S|-1)!/m! exactly in expectation.
    """
    groups = list(range(m))
    total_diff = 0.0
    for _ in range(n_samples):
        perm = np.random.permutation(groups)
        k_idx = int(np.where(perm == k)[0][0])
        S = frozenset(perm[:k_idx])
        v_with_k = value_function_numpy(model, background_arr, eval_x_arr, S | {k}, group_indices, feature_names)
        v_without_k = value_function_numpy(model, background_arr, eval_x_arr, S, group_indices, feature_names)
        total_diff += v_with_k - v_without_k
    return total_diff / n_samples


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
) -> float:
    """
    Unbiased Monte Carlo estimate of the Shapley Interaction Index I_{jk}.

        I_{jk} = Σ_{S ⊆ M\\{j,k}} w(s,m) [v(S∪{j,k}) - v(S∪{j}) - v(S∪{k}) + v(S)]
        w(s,m) = s!(m-s-2)! / (m-1)!

    Sampling distribution (proved unbiased):
      Draw s uniformly from {0,...,m-2}  →  P(size=s) = 1/(m-1)
      Draw a random permutation of M\\{j,k} and take its first s elements as S.
      Then P(specific S of size s) = s!(m-2-s)!/((m-1)(m-2)!) = s!(m-s-2)!/(m-1)! = w(s,m).
    """
    rest = [g for g in range(m) if g not in (j, k)]
    total_delta = 0.0
    for _ in range(n_samples):
        if rest:
            perm = np.random.permutation(rest)
            s_size = np.random.randint(0, len(rest) + 1)   # uniform over {0,...,m-2}
            S = frozenset(perm[:s_size])
        else:
            S = frozenset()
        v_jk = value_function_numpy(model, background_arr, eval_x_arr, S | {j, k}, group_indices, feature_names)
        v_j  = value_function_numpy(model, background_arr, eval_x_arr, S | {j},     group_indices, feature_names)
        v_k  = value_function_numpy(model, background_arr, eval_x_arr, S | {k},     group_indices, feature_names)
        v_0  = value_function_numpy(model, background_arr, eval_x_arr, S,           group_indices, feature_names)
        total_delta += v_jk - v_j - v_k + v_0
    return total_delta / n_samples


def grouped_shapley_all(
    model,
    X_background: pd.DataFrame,
    X_eval: pd.DataFrame,
    group_cols: list[list[str]],
    group_names: list[str] | None = None,
    mc_samples: int = 50,
) -> pd.DataFrame:
    """
    Compute grouped Shapley values for all eval points via Monte Carlo.
    Returns DataFrame of shape (n_eval, m_groups).
    """
    m = len(group_cols)
    feature_names = list(X_background.columns)
    col_to_idx = {name: i for i, name in enumerate(feature_names)}
    group_indices = [[col_to_idx[c] for c in g] for g in group_cols]

    background_arr = X_background.values
    eval_arr = X_eval.values

    cols = group_names if group_names is not None else [f"group_{k}" for k in range(m)]
    rows = []
    for i in range(len(eval_arr)):
        print(f"  Explaining evaluation point {i+1}/{len(eval_arr)}...", end="\r")
        vals = [
            grouped_shapley_single_mc(
                model, background_arr, eval_arr[i], k, m, group_indices, feature_names, mc_samples
            )
            for k in range(m)
        ]
        rows.append(vals)
    print()
    return pd.DataFrame(rows, columns=cols)


def grouped_interaction_all(
    model,
    X_background: pd.DataFrame,
    X_eval: pd.DataFrame,
    group_cols: list[list[str]],
    group_names: list[str],
    mc_samples: int = 50,
) -> pd.DataFrame:
    """
    Compute all pairwise Shapley Interaction Index values between groups.
    Returns DataFrame sorted by mean_abs_interaction.
    """
    m = len(group_cols)
    feature_names = list(X_background.columns)
    col_to_idx = {name: i for i, name in enumerate(feature_names)}
    group_indices = [[col_to_idx[c] for c in g] for g in group_cols]

    background_arr = X_background.values
    eval_arr = X_eval.values

    results = []
    for j in range(m):
        for k in range(j + 1, m):
            print(f"  Calculating interaction between {group_names[j]} and {group_names[k]}...")
            pair_interactions = [
                grouped_shapley_interaction_single_mc(
                    model, background_arr, eval_arr[i], j, k, m, group_indices, feature_names, mc_samples
                )
                for i in range(len(eval_arr))
            ]
            results.append({
                "group_j":            group_names[j],
                "group_k":            group_names[k],
                "mean_interaction":     float(np.mean(pair_interactions)),
                "mean_abs_interaction": float(np.mean(np.abs(pair_interactions))),
                "sd_interaction":       float(np.std(pair_interactions)),
            })
    return pd.DataFrame(results).sort_values("mean_abs_interaction", ascending=False)


def summarize_grouped_shapley(
    phi_df: pd.DataFrame,
    group_names: list[str],
) -> pd.DataFrame:
    rows = []
    for k, name in enumerate(group_names):
        col = phi_df.columns[k]
        vals = phi_df[col].values
        rows.append({
            "group":         name,
            "mean_phi":      float(np.mean(vals)),
            "mean_abs_phi":  float(np.mean(np.abs(vals))),
            "sd_abs_phi":    float(np.std(np.abs(vals))),
            "n_eval_points": len(vals),
        })
    return pd.DataFrame(rows).sort_values("mean_abs_phi", ascending=False)


def load_group_manifest(
    manifest_path: str,
) -> tuple[list[str], list[list[str]], pd.DataFrame]:
    """
    Load a group manifest TSV with columns 'feature' and 'group'.

    Returns
    -------
    group_names   : sorted list of unique group labels
    group_cols    : group_cols[j] = list of feature names in group j
    manifest_df   : raw manifest DataFrame (for downstream inspection)
    """
    manifest_df = pd.read_csv(manifest_path, sep="\t")
    if "feature" not in manifest_df.columns or "group" not in manifest_df.columns:
        raise ValueError(
            f"Manifest file '{manifest_path}' must have 'feature' and 'group' columns. "
            f"Found: {list(manifest_df.columns)}"
        )
    group_names = sorted(manifest_df["group"].dropna().unique().tolist())
    group_cols = [
        manifest_df.loc[manifest_df["group"] == g, "feature"].tolist()
        for g in group_names
    ]
    pairs = [(n, c) for n, c in zip(group_names, group_cols) if c]
    if not pairs:
        return [], [], manifest_df
    group_names_out, group_cols_out = zip(*pairs)
    return list(group_names_out), list(group_cols_out), manifest_df


def build_group_cols_from_meta(
    feature_meta: pd.DataFrame,
    group_col: str = "chromosome",
) -> tuple[list[str], list[list[str]]]:
    """
    Build (group_names, group_cols) from feature_meta.
    Only SNP features (feature_type == 'snp') are grouped; covariates are
    excluded and always marginalised in the value function.

    Returns
    -------
    group_names : list[str]   sorted group labels (e.g. ['Chr1', 'Chr2', ...])
    group_cols  : list[list[str]]  group_cols[j] = SNP column names in group j
    """
    snp_meta = feature_meta[feature_meta["feature_type"] == "snp"].copy()
    group_names = sorted(snp_meta[group_col].dropna().unique())
    group_cols = [
        snp_meta.loc[snp_meta[group_col] == g, "feature"].tolist()
        for g in group_names
    ]
    pairs = [(n, c) for n, c in zip(group_names, group_cols) if c]
    if not pairs:
        return [], []
    group_names, group_cols = zip(*pairs)
    return list(group_names), list(group_cols)
