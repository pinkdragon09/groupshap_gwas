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
    Fast interventional value function using NumPy.
    """
    X_syn = background_arr.copy()
    for j in coalition:
        indices = group_indices[j]
        X_syn[:, indices] = eval_x_arr[indices]
    
    # Convert back to DataFrame for the model if necessary
    # (Most models accept NumPy but some pipeline objects need column names)
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
    groups = list(range(m))
    total_diff = 0.0
    
    for _ in range(n_samples):
        perm = np.random.permutation(groups)
        k_idx = np.where(perm == k)[0][0]
        S = frozenset(perm[:k_idx])
        
        v_with_k = value_function_numpy(model, background_arr, eval_x_arr, S | {k}, group_indices, feature_names)
        v_without_k = value_function_numpy(model, background_arr, eval_x_arr, S, group_indices, feature_names)
        total_diff += (v_with_k - v_without_k)
        
    return total_diff / n_samples


def grouped_shapley_all(
    model,
    X_background: pd.DataFrame,
    X_eval: pd.DataFrame,
    group_cols: list[list[str]],
    group_names: list[str] | None = None,
    mc_samples: int = 50,
) -> pd.DataFrame:
    """
    Optimized grouped Shapley using NumPy and Monte Carlo.
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
        vals = [grouped_shapley_single_mc(model, background_arr, eval_arr[i], k, m, group_indices, feature_names, mc_samples) for k in range(m)]
        rows.append(vals)
    print()
    return pd.DataFrame(rows, columns=cols)


def summarize_grouped_shapley(
    phi_df: pd.DataFrame,
    group_names: list[str],
) -> pd.DataFrame:
    rows = []
    for k, name in enumerate(group_names):
        col = phi_df.columns[k]
        vals = phi_df[col].values
        rows.append(
            {
                "group": name,
                "mean_phi": float(np.mean(vals)),
                "mean_abs_phi": float(np.mean(np.abs(vals))),
                "sd_abs_phi": float(np.std(np.abs(vals))),
                "n_eval_points": len(vals),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_abs_phi", ascending=False)
